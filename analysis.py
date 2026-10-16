# -*- coding: utf-8 -*-
import json
import os
import numpy as np
import pandas as pd

# ---------- 业务口径 ----------
COST_EMPTY_RUN = 90     # 每次现场缺料空跑损失工时
COST_BLOCK = 15         # 每提前阻断一次调拨花费工时

DATA_PATH = "data/material_claims.csv"
OUT_DIR = "outputs"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 1) 时区统一: requested_at 带 +08:00, stock_synced_at 是 UTC
    df["requested_at"] = pd.to_datetime(df["requested_at"], utc=True)
    df["stock_synced_at"] = pd.to_datetime(df["stock_synced_at"], utc=True)

    # 2) 库存同步滞后(分钟): 申请时刻 - 账面库存最后同步时刻
    df["sync_lag_min"] = (
        df["requested_at"] - df["stock_synced_at"]
    ).dt.total_seconds() / 60.0

    # 3) 提交小时(统一到北京时间 UTC+8), 用于刻画时段规律
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
    """对比履约/落空两组的关键特征, 验证'同步滞后+库存+繁忙度'假设"""
    g = df.groupby("is_fulfilled")
    summary = {}
    for col in ["stock_qty", "day_orders", "sync_lag_min"]:
        summary[col] = {
            "fulfilled_median": float(g[col].median().get(1, np.nan)),
            "unfulfilled_median": float(g[col].median().get(0, np.nan)),
            "fulfilled_mean": float(g[col].mean().get(1, np.nan)),
            "unfulfilled_mean": float(g[col].mean().get(0, np.nan)),
        }

    # 分仓/分品类履约率
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
    }


def build_rule(df: pd.DataFrame):
    """
    可执行阻断规则:
    风险分 = 库存紧缺度 + 同步滞后 + 仓库繁忙度
    触发阻断 => 预计落空, 提前调拨(花 15 工时) 而非现场空跑(损失 90)
    用阈值扫描选净收益最大的规则。
    """
    d = df.copy()

    # 标准化三个风险因子(用分位数阈值, 便于一线理解)
    lag_hi = d["sync_lag_min"].quantile(0.75)
    order_hi = d["day_orders"].quantile(0.75)
    stock_lo = d["stock_qty"].quantile(0.25)

    d["risk_score"] = (
        (d["sync_lag_min"] >= lag_hi).astype(int)
        + (d["day_orders"] >= order_hi).astype(int)
        + (d["stock_qty"] <= stock_lo).astype(int)
    )

    # 扫描: 风险分 >= k 时阻断
    best = None
    rows = []
    for k in [1, 2, 3]:
        flagged = d["risk_score"] >= k
        n_flag = int(flagged.sum())
        if n_flag == 0:
            continue
        # 被阻断中真实会落空的数量(命中)
        hit = int(((d["is_fulfilled"] == 0) & flagged).sum())
        # 被阻断但本可履约(误伤)
        miss = int(((d["is_fulfilled"] == 1) & flagged).sum())
        # 未阻断但落空(漏网)
        miss_unfulfilled = int(((d["is_fulfilled"] == 0) & ~flagged).sum())

        # 净收益 = 避免的空跑损失 - 阻断成本
        saved = hit * COST_EMPTY_RUN - n_flag * COST_BLOCK
        rows.append({
            "threshold_k": k,
            "n_flagged": n_flag,
            "flag_rate": round(n_flag / len(d), 4),
            "hits_true_unfulfilled": hit,
            "false_blocks": miss,
            "precision": round(hit / n_flag, 4) if n_flag else 0,
            "recall": round(hit / max(1, (d["is_fulfilled"] == 0).sum()), 4),
            "net_saved_hours": int(saved),
        })
        if best is None or saved > best["net_saved_hours"]:
            best = rows[-1]

    thresholds = {
        "lag_hi_min": float(lag_hi),
        "order_hi": float(order_hi),
        "stock_lo": float(stock_lo),
    }
    return thresholds, rows, best, d


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, meta = load_and_clean(DATA_PATH)
    diag = diagnose(df)
    thresholds, rule_rows, best, d = build_rule(df)

    metrics = {
        "meta": meta,
        "diagnosis": diag,
        "rule_thresholds": thresholds,
        "rule_scan": rule_rows,
        "best_rule": best,
        "cost_model": {"empty_run_hours": COST_EMPTY_RUN, "block_hours": COST_BLOCK},
    }

    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 落空样本明细, 供现场复盘
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

    print("\n" + "=" * 60)
    print("核心特征对比 (落空 vs 履约, 中位数)")
    print("=" * 60)
    for col, s in diag["feature_summary"].items():
        print(f"  {col:14s} 落空={s['unfulfilled_median']:.2f}  履约={s['fulfilled_median']:.2f}")

    print("\n" + "=" * 60)
    print("阻断规则扫描 (成本: 空跑90 / 阻断15 工时)")
    print("=" * 60)
    print(f"  阈值: 滞后>={thresholds['lag_hi_min']:.1f}min, "
          f"当日单量>={thresholds['order_hi']:.0f}, "
          f"库存<={thresholds['stock_lo']:.0f}")
    for r in rule_rows:
        print(f"  k>={r['threshold_k']}: 触发{r['n_flagged']}单 "
              f"命中{r['hits_true_unfulfilled']} 误伤{r['false_blocks']} "
              f"精确率{r['precision']:.2f} 召回{r['recall']:.2f} "
              f"净省{r['net_saved_hours']}工时")
    print(f"\n  最优规则: k>={best['threshold_k']}, 净省 {best['net_saved_hours']} 工时")

    print("\n结果已写入 outputs/metrics.json 和 outputs/unfulfilled_cases.csv")


if __name__ == "__main__":
    main()
