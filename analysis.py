# -*- coding: utf-8 -*-
"""工程物资领料履约诊断与提前阻断规则。

数据: material_claims.csv (仅使用题目发放数据, 无补造数据)。
成本口径: 每次现场缺料空跑损失 90 工时; 每提前阻断一次调拨花费 15 工时。

运行: python analysis.py   (自动在 ./data/ 与仓库根目录寻找 CSV)
输出: outputs/metrics.json, outputs/unfulfilled_cases.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

# ---------- 业务口径 ----------
COST_EMPTY_RUN = 90     # 每次现场缺料空跑损失工时
COST_BLOCK = 15         # 每提前阻断一次调拨花费工时

CANDIDATE_PATHS = [
    os.path.join("data", "material_claims.csv"),
    "material_claims.csv",
]
OUT_DIR = "outputs"


def _force_utf8_stdout() -> None:
    """Windows 控制台默认 GBK, 强制 UTF-8 避免中文乱码。

    如仍乱码, 请先在终端执行 chcp 65001 切换代码页。
    outputs/ 下的文件始终以 UTF-8 写出, 不受影响。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def resolve_data_path() -> str:
    for p in CANDIDATE_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "未找到 material_claims.csv, 请放在仓库根目录或 data/ 目录下。"
    )


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 1) 时区统一: requested_at 带 +08:00, stock_synced_at 是 UTC(Z), 全部转 UTC
    df["requested_at"] = pd.to_datetime(df["requested_at"], utc=True)
    df["stock_synced_at"] = pd.to_datetime(df["stock_synced_at"], utc=True)

    # 2) 库存同步滞后(分钟): 申请时刻 - 账面库存最后同步时刻
    df["sync_lag_min"] = (
        df["requested_at"] - df["stock_synced_at"]
    ).dt.total_seconds() / 60.0

    # 3) 提交小时(统一到北京时间 UTC+8), 用于时段分析
    df["request_hour"] = (
        df["requested_at"].dt.tz_convert("Asia/Shanghai").dt.hour
    )

    # 4) 去重: 同一 claim_id 保留首次出现
    n_raw = len(df)
    df = df.drop_duplicates(subset="claim_id", keep="first").copy()
    n_dedup = len(df)

    # 5) 剔除 is_fulfilled 为空(空字符串/NaN)
    df["is_fulfilled"] = pd.to_numeric(df["is_fulfilled"], errors="coerce")
    n_before_drop = len(df)
    df = df[df["is_fulfilled"].notna()].copy()
    df["is_fulfilled"] = df["is_fulfilled"].astype(int)
    n_lab = len(df)

    meta = {
        "rows_raw": int(n_raw),
        "rows_after_dedup": int(n_dedup),
        "duplicated_claim_ids": int(n_raw - n_dedup),
        "rows_dropped_no_label": int(n_before_drop - n_lab),
        "rows_valid": int(n_lab),
    }
    return df, meta


def diagnose(df: pd.DataFrame) -> dict:
    """对比履约/落空两组的关键特征, 验证'同步滞后+库存+繁忙度'假设。"""
    g = df.groupby("is_fulfilled")
    summary = {}
    for col in ["stock_qty", "day_orders", "sync_lag_min"]:
        summary[col] = {
            "fulfilled_median": float(g[col].median().get(1, np.nan)),
            "unfulfilled_median": float(g[col].median().get(0, np.nan)),
            "fulfilled_mean": float(g[col].mean().get(1, np.nan)),
            "unfulfilled_mean": float(g[col].mean().get(0, np.nan)),
        }

    by_wh = df.groupby("warehouse_id")["is_fulfilled"].agg(["mean", "count"])
    by_mat = df.groupby("material_type")["is_fulfilled"].agg(["mean", "count"])

    overall_rate = float(df["is_fulfilled"].mean())
    return {
        "overall_fulfill_rate": overall_rate,
        "feature_summary": summary,
        "by_warehouse": by_wh.round(4).to_dict(orient="index"),
        "by_material": by_mat.round(4).to_dict(orient="index"),
        "unfulfilled_count": int((df["is_fulfilled"] == 0).sum()),
        "fulfilled_count": int((df["is_fulfilled"] == 1).sum()),
        "baseline_empty_run_cost_hours": int(
            (df["is_fulfilled"] == 0).sum() * COST_EMPTY_RUN
        ),
    }


def evaluate_flag(df: pd.DataFrame, flagged: pd.Series) -> dict:
    """给定阻断标记, 计算命中/误伤/净节省工时。"""
    n_flag = int(flagged.sum())
    hit = int(((df["is_fulfilled"] == 0) & flagged).sum())
    false_blocks = int(((df["is_fulfilled"] == 1) & flagged).sum())
    n_unf = int((df["is_fulfilled"] == 0).sum())
    net = hit * COST_EMPTY_RUN - n_flag * COST_BLOCK
    return {
        "n_flagged": n_flag,
        "flag_rate": round(n_flag / len(df), 4),
        "hits_true_unfulfilled": hit,
        "false_blocks": false_blocks,
        "precision": round(hit / n_flag, 4) if n_flag else 0.0,
        "recall": round(hit / n_unf, 4) if n_unf else 0.0,
        "net_saved_hours": int(net),
    }


def best_lag_threshold(df: pd.DataFrame) -> tuple:
    """在 sync_lag_min 上网格搜索, 选净节省工时最大的阻断阈值。"""
    best = None
    scan = []
    for v in range(0, int(df["sync_lag_min"].max()) + 1):
        flagged = df["sync_lag_min"] >= v
        if flagged.sum() == 0:
            continue
        ev = evaluate_flag(df, flagged)
        row = {"threshold_min": v, **ev}
        scan.append(row)
        if best is None or ev["net_saved_hours"] > best["net_saved_hours"]:
            best = row
    return best, scan


def time_split_validate(df: pd.DataFrame, thr_min: int) -> dict:
    """按申请时间前 70% 训练选阈值、后 30% 验证, 避免纯样本内过拟合。"""
    d = df.sort_values("requested_at").reset_index(drop=True)
    split = int(len(d) * 0.7)
    train, test = d.iloc[:split], d.iloc[split:]

    # 仅在训练集上重选阈值
    train_best = None
    for v in range(0, int(train["sync_lag_min"].max()) + 1):
        flagged = train["sync_lag_min"] >= v
        if flagged.sum() == 0:
            continue
        ev = evaluate_flag(train, flagged)
        if train_best is None or ev["net_saved_hours"] > train_best["net_saved_hours"]:
            train_best = {"threshold_min": v, **ev}

    v = train_best["threshold_min"]
    test_ev = evaluate_flag(test, test["sync_lag_min"] >= v)
    test_ev.update({
        "threshold_min": v,
        "n_test": int(len(test)),
        "baseline_empty_run_cost_hours": int(
            (test["is_fulfilled"] == 0).sum() * COST_EMPTY_RUN
        ),
    })
    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_best_threshold_min": v,
        "train_net_saved_hours": train_best["net_saved_hours"],
        "test_metrics": test_ev,
    }


def main():
    _force_utf8_stdout()
    os.makedirs(OUT_DIR, exist_ok=True)
    data_path = resolve_data_path()
    df, meta = load_and_clean(data_path)
    diag = diagnose(df)
    best, lag_scan = best_lag_threshold(df)

    # 在全量最优阈值上出一份明细 + 时间切分验证
    flagged = df["sync_lag_min"] >= best["threshold_min"]
    ts = time_split_validate(df, best["threshold_min"])

    metrics = {
        "data_path": data_path,
        "meta": meta,
        "diagnosis": diag,
        "cost_model": {"empty_run_hours": COST_EMPTY_RUN, "block_hours": COST_BLOCK},
        "best_rule": {
            "name": f"sync_lag_min >= {best['threshold_min']}",
            **{k: v for k, v in best.items() if k != "threshold_min"},
            "threshold_min": best["threshold_min"],
        },
        "lag_threshold_scan": lag_scan,
        "time_split_validation": ts,
    }

    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 被阻断且真实落空的样本明细, 供现场复盘
    d = df.copy()
    d["flagged_block"] = flagged
    d[d["is_fulfilled"] == 0].to_csv(
        os.path.join(OUT_DIR, "unfulfilled_cases.csv"), index=False, encoding="utf-8-sig"
    )

    # ---------- 控制台报告 ----------
    print("=" * 60)
    print("数据清洗")
    print("=" * 60)
    for k, v in meta.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("履约概况")
    print("=" * 60)
    print(f"  整体履约率: {diag['overall_fulfill_rate']:.4f}")
    print(f"  落空单数:   {diag['unfulfilled_count']}")
    print(f"  履约单数:   {diag['fulfilled_count']}")
    print(f"  不干预基线空跑成本: {diag['baseline_empty_run_cost_hours']} 工时")

    print("\n" + "=" * 60)
    print("核心特征对比 (落空 vs 履约, 中位数)")
    print("=" * 60)
    for col, s in diag["feature_summary"].items():
        print(f"  {col:14s} 落空={s['unfulfilled_median']:.2f}  履约={s['fulfilled_median']:.2f}")

    print("\n" + "=" * 60)
    print("阻断规则 (成本: 空跑90 / 阻断15 工时)")
    print("=" * 60)
    print(f"  最优规则: sync_lag_min >= {best['threshold_min']}")
    print(f"  触发 {best['n_flagged']} 单 (占 {best['flag_rate']:.1%}), "
          f"命中 {best['hits_true_unfulfilled']}, 误伤 {best['false_blocks']}")
    print(f"  精确率 {best['precision']:.3f}, 召回 {best['recall']:.3f}")
    print(f"  净节省 {best['net_saved_hours']} 工时 "
          f"(基线 {diag['baseline_empty_run_cost_hours']} 工时)")

    print("\n" + "=" * 60)
    print("时间切分验证 (前70%选阈值 / 后30%验证)")
    print("=" * 60)
    te = ts["test_metrics"]
    print(f"  训练集选出阈值: lag>={ts['train_best_threshold_min']} "
          f"(训练净省 {ts['train_net_saved_hours']} 工时)")
    print(f"  测试集: 触发 {te['n_flagged']} 单, 命中 {te['hits_true_unfulfilled']}, "
          f"精确率 {te['precision']:.3f}, 召回 {te['recall']:.3f}, "
          f"净节省 {te['net_saved_hours']} 工时 "
          f"(测试基线 {te['baseline_empty_run_cost_hours']} 工时)")

    print("\n结果已写入 outputs/metrics.json 和 outputs/unfulfilled_cases.csv")


if __name__ == "__main__":
    main()
