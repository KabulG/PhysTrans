#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np


def acc_day(actual: np.ndarray, pred: np.ndarray, cap: float) -> float:
   
    mask = np.isfinite(actual) & np.isfinite(pred)
    if mask.sum() == 0:
        return np.nan
    a = actual[mask]
    p = pred[mask]
    den = np.where(a >= 0.2 * cap, a, 0.2 * cap)
    se = ((a - p) / den) ** 2
    mse = se.mean()
    acc = (1.0 - np.sqrt(mse)) * 100.0
    return acc


def evaluate(df, time_col, actual_col, pred_cols, cap):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
   
    days = []
    for day, g in df.groupby(pd.Grouper(freq="1D")):
        if len(g) == 0:
            continue
        expected = 96
        actual_vals = g[actual_col].to_numpy()
      
        data_valid = np.isfinite(actual_vals).sum() / expected * 100.0
        day_result = {
            "date": day.date().isoformat(),
            "data_valid_pct": data_valid,
        }
        for name, col in pred_cols.items():
            if col not in g.columns:
                day_result[f"{name}_acc"] = np.nan
                day_result[f"{name}_report_pct"] = 0.0
                continue
            pred_vals = g[col].to_numpy()
            report_pct = np.isfinite(pred_vals).sum() / expected * 100.0
            day_result[f"{name}_report_pct"] = report_pct
            day_result[f"{name}_acc"] = acc_day(actual_vals, pred_vals, cap)
        days.append(day_result)
    return pd.DataFrame(days)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV with time/actual/pred columns")
    parser.add_argument("--time_col", default="date", help="Timestamp column name")
    parser.add_argument("--actual_col", default="actual", help="Actual power column")
    parser.add_argument("--mid_col", default=None, help="Mid-term forecast column (e.g., 4-day ahead)")
    parser.add_argument("--short_col", default=None, help="Short-term forecast column (e.g., day ahead)")
    parser.add_argument("--ultra_col", default=None, help="Ultra-short forecast column (e.g., 4-hour ahead)")
    parser.add_argument("--cap", type=float, required=True, help="Installed capacity")
    parser.add_argument("--out", default="pv_eval.csv", help="Output CSV for daily metrics")
    args = parser.parse_args()

    pred_cols = {}
    if args.mid_col:
        pred_cols["mid"] = args.mid_col
    if args.short_col:
        pred_cols["short"] = args.short_col
    if args.ultra_col:
        pred_cols["ultra"] = args.ultra_col

    df = pd.read_csv(args.csv)
    res = evaluate(df, args.time_col, args.actual_col, pred_cols, args.cap)
    res.to_csv(args.out, index=False)

 
    thresholds = {"mid": 45.0, "short": 65.0, "ultra": 70.0}
    print(res)
    print("\nMonthly summary:")
    for name in pred_cols.keys():
        acc_mean = res[f"{name}_acc"].mean(skipna=True)
        report_mean = res[f"{name}_report_pct"].mean(skipna=True)
        th = thresholds.get(name, None)
        print(f"{name}: ACC_avg={acc_mean:.2f}% (threshold {th}%), report_avg={report_mean:.2f}%")


if __name__ == "__main__":
    main()
