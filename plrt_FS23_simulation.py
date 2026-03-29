import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import inspect
from __future__ import annotations

meta = pd.read_csv("meta_all(508).csv")
key = pd.read_csv("key_all(508)_stodiff_fixed.csv")

def plrt_mlr(storage_pre: float, release_pre: float, inflow: float, sto_diff: float,
             release_roll7: float, inflow_roll7: float, storage_x_inflow: float,
             inflow2: float, release_pre2: float, max_sto: float, rts: float) -> dict:

    vals = [storage_pre, release_pre, inflow, sto_diff,
            release_roll7, inflow_roll7, storage_x_inflow,
            inflow2, release_pre2, max_sto, rts]

    if any(pd.isna(v) for v in vals):
        raise ValueError("❌ MISSING INPUT in plrt_mlr()")

    # S_max thresholds converted from 1000 acre-ft to m3
    THR_SMAX_1 = 154055714.4068      # 124.895 * 1000 acre-ft
    THR_SMAX_2 = 778203692.8560      # 630.9   * 1000 acre-ft
    THR_SMAX_3 = 6410281774.2960     # 5196.9  * 1000 acre-ft

    # (1) rts <= 9.553
    if rts <= 9.553:

        # (2) max_sto <= 124.895 (1000 acre-ft)
        if max_sto <= THR_SMAX_1:
            val = (0.001 + storage_pre*0.067 + release_pre*0.133 +
                   inflow*1.164 + sto_diff*0.074 + release_roll7*(-0.026) +
                   inflow_roll7*(0.087) + storage_x_inflow*(-0.155) +
                   inflow2*0 + release_pre2*0)
            return {"value": val, "mode": 1}

        else:
            val = (0.001 + storage_pre*0.039 + release_pre*0.332 +
                   inflow*0.664 + sto_diff*0.101 + release_roll7*(-0.020) +
                   inflow_roll7*(0.103) + storage_x_inflow*(-0.081) +
                   inflow2*0 + release_pre2*0)
            return {"value": val, "mode": 2}

    # (1) is false -> rts > 9.553
    else:

        # (3) inflow <= 0.209
        if inflow > 0.209:
            val = (-0.058 + storage_pre*0.029 + release_pre*0.782 +
                   inflow*0.024 + sto_diff*0.537 + release_roll7*(0.166) +
                   inflow_roll7*(0.016) + storage_x_inflow*(-0.006) +
                   inflow2*0 + release_pre2*0)
            return {"value": val, "mode": 3}

        else:
            # (4) release_pre <= -0.560
            if release_pre <= -0.560:

                # (5) max_sto <= 630.9 (1000 acre-ft)
                if max_sto <= THR_SMAX_2:
                    val = (-0.064 + storage_pre*0.017 + release_pre*0.495 +
                           inflow*0.039 + sto_diff*0.145 + release_roll7*(0.368) +
                           inflow_roll7*(-0.045) + storage_x_inflow*(0.005) +
                           inflow2*0 + release_pre2*0)
                    return {"value": val, "mode": 4}

                else:
                    val = (-0.096 + storage_pre*0.007 + release_pre*0.259 +
                           inflow*(-0.425) + sto_diff*0.159 + release_roll7*(0.585) +
                           inflow_roll7*(0.048) + storage_x_inflow*(0.295) +
                           inflow2*0 + release_pre2*0)
                    return {"value": val, "mode": 5}

            else:
                # (6) max_sto <= 5196.9 (1000 acre-ft)
                if max_sto <= THR_SMAX_3:
                    val = (-0.056 + storage_pre*0.020 + release_pre*0.656 +
                           inflow*(-0.159) + sto_diff*0.194 + release_roll7*(0.251) +
                           inflow_roll7*(0.072) + storage_x_inflow*(-0.012) +
                           inflow2*0 + release_pre2*0)
                    return {"value": val, "mode": 6}

                else:
                    val = (-0.021 + storage_pre*0.020 + release_pre*0.665 +
                           inflow*(-0.054) + sto_diff*0.003 + release_roll7*(0.284) +
                           inflow_roll7*(0.008) + storage_x_inflow*(0.034) +
                           inflow2*0 + release_pre2*0)
                    return {"value": val, "mode": 7}
                
def _safe_predict(model, X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    try:
        sig = inspect.signature(model.predict)
        if "verbose" in sig.parameters:  # Keras
            y = model.predict(X, verbose=0)
        else:  # scikit-learn 
            y = model.predict(X)
    except (TypeError, ValueError, AttributeError):
        y = model.predict(X)

    return float(np.asarray(y).squeeze()) if X.shape[0] == 1 else np.asarray(y).squeeze()

def _as_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def get_std(mean, stdev, value):
    mean  = _as_float(mean)
    stdev = _as_float(stdev)
    value = _as_float(value)

    if np.isnan(mean):  mean  = 0.0
    if np.isnan(stdev) or stdev == 0:  
        return value
    return (value - mean) / stdev

def get_raw(mean, stdev, value):
    mean  = _as_float(mean)
    stdev = _as_float(stdev)
    value = _as_float(value)

    if np.isnan(mean):  mean  = 0.0
    if np.isnan(stdev) or stdev == 0:  
        return value + mean  

    return stdev * value + mean


KEY_COLS = [
    "release_mean","release_std",
    "storage_mean","storage_std",
    "release_roll7_mean","release_roll7_std",
    "storage_roll7_mean","storage_roll7_std",
    "release_pre_mean","release_pre_std",
    "release_pre2_mean","release_pre2_std",
    "storage_pre_mean","storage_pre_std",
    "sto_diff_mean","sto_diff_std",
    "storage_x_inflow_mean","storage_x_inflow_std",
]

def coerce_numeric(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

coerce_numeric(key, KEY_COLS)

STD_NUM_COLS = [
    "storage_pre","release_pre","inflow","sto_diff","release_roll7","inflow_roll7",
    "storage_x_inflow","inflow2","release_pre2","max_sto","rts","month"
]
RAW_NUM_COLS = ["storage_pre","inflow","release"]  

def build_key_lookup(key_df: pd.DataFrame):

    key_lookup = {}
    for (rid, m), sub in key_df.groupby(["res_id", "month"]):
        row = sub.iloc[0]
        key_lookup.setdefault(rid, {})[m] = {col: row[col] for col in KEY_COLS if col in row}
    return key_lookup

key_lookup = build_key_lookup(key)

def run_plrt(res_id: int,
             std_df: pd.DataFrame,
             raw_df: pd.DataFrame,
             key_lookup: dict,
             frequency: int = 7) -> pd.DataFrame:

    std_data = std_df.loc[std_df["res_id"] == res_id].sort_values("date").reset_index(drop=True)
    raw_data = raw_df.loc[raw_df["res_id"] == res_id].sort_values("date").reset_index(drop=True)

    if std_data.empty or raw_data.empty:
        raise ValueError(f"res_id {res_id}: std/raw empty data")

    if len(std_data) != len(raw_data):
        raise ValueError(f"res_id {res_id}: std/raw different length ({len(std_data)} != {len(raw_data)})")

    if not std_data["date"].equals(raw_data["date"]):
        raise ValueError(f"res_id {res_id}: std/raw date not equal")

    n = len(std_data)

    out = std_data[["res_id", "date"]].copy()
    cols_init = [
        "release_pred_raw","release_pred_std","storage_pred_raw","storage_pred_std",
        "release_pred_roll7_raw","release_pred_pre_raw","release_pred_pre2_raw",
        "storage_pred_roll7_raw","storage_pred_pre_raw","sto_diff_pred_raw","storage_x_inflow_pred_raw",
        "release_pred_roll7_std","release_pred_pre_std","release_pred_pre2_std",
        "storage_pred_roll7_std","storage_pred_pre_std","sto_diff_pred_std","storage_x_inflow_pred_std",
        "plrt_mode"
    ]
    for c in cols_init:
        out[c] = np.nan
  
    for c in cols_init:
        out[c] = out[c].astype("float64")


    for t in range(n):
  
        m = int(std_data.loc[t, "month"])
        if res_id not in key_lookup or m not in key_lookup[res_id]:
            raise KeyError(f"key_lookup has no res_id={res_id}, month={m}")
        key_row = key_lookup[res_id][m]


        if t < 7 or ((t + 1) % frequency == 0):
            res = plrt_mlr(  
                std_data.loc[t, "storage_pre"],     
                std_data.loc[t, "release_pre"],      
                std_data.loc[t, "inflow"],         
                std_data.loc[t, "sto_diff"],          
                std_data.loc[t, "release_roll7"],   
                std_data.loc[t, "inflow_roll7"],    
                std_data.loc[t, "storage_x_inflow"],  
                std_data.loc[t, "inflow2"],          
                std_data.loc[t, "release_pre2"],     
                std_data.loc[t, "max_sto"],          
                std_data.loc[t, "rts"]                
            )
            val  = float(res["value"])
            mode = int(res["mode"])

            out.loc[t, "release_pred_std"] = val
            out.loc[t, "plrt_mode"]        = float(mode)

            release_pred_raw = get_raw(key_row["release_mean"], key_row["release_std"], val)
            #release_pred_raw = max(0.0, min(release_pred_raw, release_max_global))
            out.loc[t, "release_pred_raw"] = float(release_pred_raw)

            storage_pred_raw = (
                float(raw_data.loc[t, "storage_pre"])
                + float(raw_data.loc[t, "inflow"])
                - float(release_pred_raw)
            )
            #storage_pred_raw = max(0.0, min(storage_pred_raw, float(std_data.loc[t, "max_sto"]) * 0.9))
            out.loc[t, "storage_pred_raw"] = float(storage_pred_raw)

            out.loc[t, "storage_pred_std"] = get_std(
                key_row["storage_mean"], key_row["storage_std"], storage_pred_raw
            )

        else:
            r7r = float(np.mean(out.loc[t-7:t-1, "release_pred_raw"]))
            s7r = float(np.mean(out.loc[t-7:t-1, "storage_pred_raw"]))

            rpr = float(out.loc[t-1, "release_pred_raw"])
            rpr2 = rpr ** 2
            spr = float(out.loc[t-1, "storage_pred_raw"])
            st_diff = spr - s7r
            sxi = spr * float(raw_data.loc[t, "inflow"])

            out.loc[t, "release_pred_roll7_raw"]    = r7r
            out.loc[t, "storage_pred_roll7_raw"]    = s7r
            out.loc[t, "release_pred_pre_raw"]      = rpr
            out.loc[t, "release_pred_pre2_raw"]     = rpr2
            out.loc[t, "storage_pred_pre_raw"]      = spr
            out.loc[t, "sto_diff_pred_raw"]         = st_diff
            out.loc[t, "storage_x_inflow_pred_raw"] = sxi

            out.loc[t, "release_pred_roll7_std"] = get_std(
                key_row["release_roll7_mean"], key_row["release_roll7_std"], r7r
            )
            out.loc[t, "storage_pred_roll7_std"] = get_std(
                key_row["storage_roll7_mean"], key_row["storage_roll7_std"], s7r
            )
            out.loc[t, "release_pred_pre_std"] = get_std(
                key_row["release_pre_mean"], key_row["release_pre_std"], rpr
            )
            out.loc[t, "release_pred_pre2_std"] = get_std(
                key_row["release_pre2_mean"], key_row["release_pre2_std"], rpr2
            )
            out.loc[t, "storage_pred_pre_std"] = get_std(
                key_row["storage_pre_mean"], key_row["storage_pre_std"], spr
            )
            out.loc[t, "sto_diff_pred_std"] = get_std(
                key_row["sto_diff_mean"], key_row["sto_diff_std"], st_diff
            )
            out.loc[t, "storage_x_inflow_pred_std"] = get_std(
                key_row["storage_x_inflow_mean"], key_row["storage_x_inflow_std"], sxi
            )

            res = plrt_mlr( 
                out.loc[t, "storage_pred_pre_std"],     
                out.loc[t, "release_pred_pre_std"],     
                std_data.loc[t, "inflow"],             
                out.loc[t, "sto_diff_pred_std"],        
                out.loc[t, "release_pred_roll7_std"], 
                std_data.loc[t, "inflow_roll7"],        
                out.loc[t, "storage_x_inflow_pred_std"],
                std_data.loc[t, "inflow2"],            
                out.loc[t, "release_pred_pre2_std"],   
                std_data.loc[t, "max_sto"],            
                std_data.loc[t, "rts"]              
            )
            val  = float(res["value"])
            mode = int(res["mode"])

            out.loc[t, "release_pred_std"] = val
            out.loc[t, "plrt_mode"]        = float(mode)

            release_pred_raw = get_raw(key_row["release_mean"], key_row["release_std"], val)
            #release_pred_raw = max(0.0, min(release_pred_raw, release_max_global))
            out.loc[t, "release_pred_raw"] = float(release_pred_raw)

            storage_pred_raw = (
                float(raw_data.loc[t, "storage_pre"])
                + float(raw_data.loc[t, "inflow"])
                - float(release_pred_raw)
            )
            #storage_pred_raw = max(0.0, min(storage_pred_raw, float(std_data.loc[t, "max_sto"]) * 0.9))
            out.loc[t, "storage_pred_raw"] = float(storage_pred_raw)

            out.loc[t, "storage_pred_std"] = get_std(
                key_row["storage_mean"], key_row["storage_std"], storage_pred_raw
            )

    return pd.concat([std_data, out.drop(columns=["res_id", "date"])], axis=1)


def run_plrt_bc(res_id: int,
                std_df: pd.DataFrame,
                raw_df: pd.DataFrame,
                key_lookup: dict,
                frequency: int = 7) -> pd.DataFrame:

    std_data = std_df.loc[std_df["res_id"] == res_id].sort_values("date").reset_index(drop=True)
    raw_data = raw_df.loc[raw_df["res_id"] == res_id].sort_values("date").reset_index(drop=True)

    if std_data.empty or raw_data.empty:
        raise ValueError(f"res_id {res_id}: std/raw no data")

    if len(std_data) != len(raw_data):
        raise ValueError(f"res_id {res_id}: std/raw length different ({len(std_data)} != {len(raw_data)})")

    if not std_data["date"].equals(raw_data["date"]):
        raise ValueError(f"res_id {res_id}: std/raw date different")

    n = len(std_data)

    out = std_data[["res_id", "date"]].copy()
    cols_init = [
        "release_pred_raw","release_pred_std","storage_pred_raw","storage_pred_std",
        "release_pred_roll7_raw","release_pred_pre_raw","release_pred_pre2_raw",
        "storage_pred_roll7_raw","storage_pred_pre_raw","sto_diff_pred_raw","storage_x_inflow_pred_raw",
        "release_pred_roll7_std","release_pred_pre_std","release_pred_pre2_std",
        "storage_pred_roll7_std","storage_pred_pre_std","sto_diff_pred_std","storage_x_inflow_pred_std",
        "plrt_mode"
    ]
    for c in cols_init:
        out[c] = np.nan

    for c in cols_init:
        out[c] = out[c].astype("float64")


    release_max_global = float(raw_data["release"].max(skipna=True))*1.1 if "release" in raw_data.columns else np.inf

    for t in range(n):
        m = int(std_data.loc[t, "month"])
        if res_id not in key_lookup or m not in key_lookup[res_id]:
            raise KeyError(f"key_lookup has no res_id={res_id}, month={m}")
        key_row = key_lookup[res_id][m]


        if t < 7 or ((t + 1) % frequency == 0):
            res = plrt_mlr( 
                std_data.loc[t, "storage_pre"],    
                std_data.loc[t, "release_pre"],     
                std_data.loc[t, "inflow"],         
                std_data.loc[t, "sto_diff"],       
                std_data.loc[t, "release_roll7"],  
                std_data.loc[t, "inflow_roll7"],     
                std_data.loc[t, "storage_x_inflow"], 
                std_data.loc[t, "inflow2"],          
                std_data.loc[t, "release_pre2"],     
                std_data.loc[t, "max_sto"],         
                std_data.loc[t, "rts"]           
            )
            val  = float(res["value"])
            mode = int(res["mode"])

            out.loc[t, "release_pred_std"] = val
            out.loc[t, "plrt_mode"]        = float(mode)

            release_pred_raw = get_raw(key_row["release_mean"], key_row["release_std"], val)
            release_pred_raw = max(0.0, min(release_pred_raw, release_max_global))
            out.loc[t, "release_pred_raw"] = float(release_pred_raw)

            storage_pred_raw = (
                float(raw_data.loc[t, "storage_pre"])
                + float(raw_data.loc[t, "inflow"])
                - float(release_pred_raw)
            )
            storage_pred_raw = max(0.0, min(storage_pred_raw, float(std_data.loc[t, "max_sto"]) * 0.9))
            out.loc[t, "storage_pred_raw"] = float(storage_pred_raw)

            out.loc[t, "storage_pred_std"] = get_std(
                key_row["storage_mean"], key_row["storage_std"], storage_pred_raw
            )

        else:
            r7r = float(np.mean(out.loc[t-7:t-1, "release_pred_raw"]))
            s7r = float(np.mean(out.loc[t-7:t-1, "storage_pred_raw"]))

            rpr = float(out.loc[t-1, "release_pred_raw"])
            rpr2 = rpr ** 2
            spr = float(out.loc[t-1, "storage_pred_raw"])
            st_diff = spr - s7r
            sxi = spr * float(raw_data.loc[t, "inflow"])

            out.loc[t, "release_pred_roll7_raw"]    = r7r
            out.loc[t, "storage_pred_roll7_raw"]    = s7r
            out.loc[t, "release_pred_pre_raw"]      = rpr
            out.loc[t, "release_pred_pre2_raw"]     = rpr2
            out.loc[t, "storage_pred_pre_raw"]      = spr
            out.loc[t, "sto_diff_pred_raw"]         = st_diff
            out.loc[t, "storage_x_inflow_pred_raw"] = sxi

            out.loc[t, "release_pred_roll7_std"] = get_std(
                key_row["release_roll7_mean"], key_row["release_roll7_std"], r7r
            )
            out.loc[t, "storage_pred_roll7_std"] = get_std(
                key_row["storage_roll7_mean"], key_row["storage_roll7_std"], s7r
            )
            out.loc[t, "release_pred_pre_std"] = get_std(
                key_row["release_pre_mean"], key_row["release_pre_std"], rpr
            )
            out.loc[t, "release_pred_pre2_std"] = get_std(
                key_row["release_pre2_mean"], key_row["release_pre2_std"], rpr2
            )
            out.loc[t, "storage_pred_pre_std"] = get_std(
                key_row["storage_pre_mean"], key_row["storage_pre_std"], spr
            )
            out.loc[t, "sto_diff_pred_std"] = get_std(
                key_row["sto_diff_mean"], key_row["sto_diff_std"], st_diff
            )
            out.loc[t, "storage_x_inflow_pred_std"] = get_std(
                key_row["storage_x_inflow_mean"], key_row["storage_x_inflow_std"], sxi
            )

            res = plrt_mlr( 
                out.loc[t, "storage_pred_pre_std"],
                out.loc[t, "release_pred_pre_std"],  
                std_data.loc[t, "inflow"],            
                out.loc[t, "sto_diff_pred_std"],      
                out.loc[t, "release_pred_roll7_std"],   
                std_data.loc[t, "inflow_roll7"],       
                out.loc[t, "storage_x_inflow_pred_std"],
                std_data.loc[t, "inflow2"],            
                out.loc[t, "release_pred_pre2_std"],  
                std_data.loc[t, "max_sto"],             
                std_data.loc[t, "rts"]               
            )
            val  = float(res["value"])
            mode = int(res["mode"])

            out.loc[t, "release_pred_std"] = val
            out.loc[t, "plrt_mode"]        = float(mode)

            release_pred_raw = get_raw(key_row["release_mean"], key_row["release_std"], val)
            release_pred_raw = max(0.0, min(release_pred_raw, release_max_global))
            out.loc[t, "release_pred_raw"] = float(release_pred_raw)

            storage_pred_raw = (
                float(raw_data.loc[t, "storage_pre"])
                + float(raw_data.loc[t, "inflow"])
                - float(release_pred_raw)
            )
            storage_pred_raw = max(0.0, min(storage_pred_raw, float(std_data.loc[t, "max_sto"]) * 0.9))
            out.loc[t, "storage_pred_raw"] = float(storage_pred_raw)

            out.loc[t, "storage_pred_std"] = get_std(
                key_row["storage_mean"], key_row["storage_std"], storage_pred_raw
            )

    return pd.concat([std_data, out.drop(columns=["res_id", "date"])], axis=1)


def get_r2(actual, pred):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)

    if a.size < 10:
        return np.nan
    if np.isnan(a).any() or np.isnan(p).any():
        return np.nan

    ss_res = np.sum((a - p) ** 2)

    mean_a = np.mean(a)
    ss_tot = np.sum((a - mean_a) ** 2)

    if ss_tot == 0:
        return np.nan
    return 1.0 - (ss_res / ss_tot)

STD_DIR  = "std_total(508)_stodiff_fixed"
META_PATH = "meta_all(508).csv"
RAW_DIR  = "raw_total(508)_stodiff_fixed"
KEY_PATH = "key_all(508)_stodiff_fixed.csv"   
FREQS    = [1, 2, 7, 15, 30]
METHODS  = {
    "plrt": run_plrt,               
    "plrt_bc": run_plrt_bc,        
}
OUT_BASE = ""       

os.makedirs(OUT_BASE, exist_ok=True)
os.makedirs(f"{OUT_BASE}/predictions", exist_ok=True)
os.makedirs(f"{OUT_BASE}/plots", exist_ok=True)


# Simulation Test
rid = 41 

raw = pd.read_csv(f"raw_total(508)_stodiff_fixed/{rid}.csv")
std = pd.read_csv(f"std_total(508)_stodiff_fixed/{rid}.csv")

coerce_numeric(key, KEY_COLS)
coerce_numeric(std, STD_NUM_COLS)
coerce_numeric(raw, RAW_NUM_COLS)

for df in (std, raw, key):
    if "date" in df.columns and not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "month" not in df.columns and "date" in df.columns:
        df["month"] = df["date"].dt.month

key_lookup = build_key_lookup(key)

pred_df = run_plrt(rid, std_df=std, raw_df=raw, key_lookup=key_lookup, frequency=1)

obs = raw.loc[raw["res_id"] == rid].sort_values("date")["release"].values
pred = pred_df["release_pred_raw"].values
r2 = get_r2(obs, pred)
print("R^2 =", r2)

pred_df = run_plrt_bc(rid, std_df=std, raw_df=raw, key_lookup=key_lookup, frequency=2)

obs = raw.loc[raw["res_id"] == rid].sort_values("date")["release"].values
pred = pred_df["release_pred_raw"].values
r2 = get_r2(obs, pred)
print("R^2 =", r2)

# Final model simulation
import os, glob
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm  

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

os.makedirs(f"{OUT_BASE}/predictions", exist_ok=True)
os.makedirs(f"{OUT_BASE}/plots", exist_ok=True)

def preprocess_pair(std_fp: str, raw_fp: str):
    std = pd.read_csv(std_fp)
    raw = pd.read_csv(raw_fp)

    coerce_numeric(std, STD_NUM_COLS)
    coerce_numeric(raw, RAW_NUM_COLS)

    for df in (std, raw):
        if "date" in df.columns and not np.issubdtype(df["date"].dtype, np.datetime64):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "month" not in df.columns and "date" in df.columns:
            df["month"] = df["date"].dt.month

    std = std.sort_values("date").reset_index(drop=True)
    raw = raw.sort_values("date").reset_index(drop=True)

    if len(std) != len(raw):
        merged = std.merge(raw, on=["res_id","date"], suffixes=("_std","_raw"))
        if merged.empty:
            return None, None
        def pick(df, suffix):
            cols = [c for c in df.columns if c.endswith(suffix)]
            out = df[["res_id","date"] + cols].copy()
            out.columns = [c.replace(suffix, "") for c in out.columns]
            return out
        std = pick(merged, "_std")
        raw = pick(merged, "_raw")

    if "release" not in raw.columns:
        return None, None
    return std, raw

def process_one_rid(rid: int):
    try:
        std_fp = os.path.join(STD_DIR, f"{rid}.csv")
        raw_fp = os.path.join(RAW_DIR, f"{rid}.csv")
        if not (os.path.exists(std_fp) and os.path.exists(raw_fp)):
            return []

        std, raw = preprocess_pair(std_fp, raw_fp)
        if std is None or raw is None:
            print(f"[SKIP] rid={rid}")
            return []

        need_months = set(pd.to_datetime(std["date"]).dt.month.unique().astype(int))
        have_months = set(key_lookup.get(rid, {}).keys())
        if not need_months.issubset(have_months):
            print(f"[SKIP] rid={rid} no key_lookup: {sorted(list(need_months-have_months))}")
            return []

        obs_series = raw["release"].astype(float).values
        dates = raw["date"].values
        r2_rows_local = []

        for method_name, method_fn in METHODS.items():
            pred_collect = pd.DataFrame({"date": dates, "obs": obs_series})
            for freq in FREQS:
                try:
                    pred_df = method_fn(rid, std_df=std, raw_df=raw,
                                        key_lookup=key_lookup, frequency=freq)
                    if "release_pred_raw" not in pred_df.columns:
                        print(f"[ERROR] rid={rid} {method_name} f={freq}: no release_pred_raw")
                        pred_collect[f"pred_f{freq}"] = np.nan
                        r2_rows_local.append({"res_id": rid, "method": method_name,
                                              "frequency": freq, "R2": np.nan})
                        continue
                    pred_vals = pred_df["release_pred_raw"].astype(float).values
                    pred_collect[f"pred_f{freq}"] = pred_vals
                    r2 = get_r2(obs_series, pred_vals)
                    r2_rows_local.append({"res_id": rid, "method": method_name,
                                          "frequency": freq, "R2": r2})
                except Exception as e:
                    print(f"[ERROR] rid={rid} {method_name} f={freq}: {e}")
                    pred_collect[f"pred_f{freq}"] = np.nan
                    r2_rows_local.append({"res_id": rid, "method": method_name,
                                          "frequency": freq, "R2": np.nan})

            out_dir = os.path.join(OUT_BASE, "predictions", method_name)
            os.makedirs(out_dir, exist_ok=True)
            pred_collect.to_csv(os.path.join(out_dir, f"{rid}.csv"), index=False)

        return r2_rows_local
    except Exception as e:
        print(f"[FATAL] rid={rid}: {e}")
        return []


std_files = sorted(glob.glob(os.path.join(STD_DIR, "*.csv")))
std_resids = [int(Path(fp).stem) for fp in std_files if Path(fp).stem.isdigit()]
key_resids = key['res_id'].unique().astype(int)


meta = pd.read_csv(META_PATH)
meta_resids = meta['ID'].unique().astype(int)
N_WORKERS = min(16, os.cpu_count() or 4)

RESID_SUBSET = [116, 353, 305, 387, 88, 10021, 89, 10031, 99, 80,
                97, 92, 90, 10025, 10032, 10027, 10023, 94,
                10026, 10030, 85]

rid_list = sorted(set(RESID_SUBSET).intersection(std_resids).intersection(meta_resids))



if __name__ == "__main__":
    print(f"[INFO] {len(rid_list)} reservoir (workers={N_WORKERS})")

    r2_rows = []
    with Pool(processes=N_WORKERS) as pool:
        for res in tqdm(pool.imap_unordered(process_one_rid, rid_list, chunksize=2),
                        total=len(rid_list), desc="Simulating reservoirs", ncols=90):
            if res:
                r2_rows.extend(res)

    r2_df = pd.DataFrame(r2_rows)
    r2_csv = os.path.join(OUT_BASE, "r2_summary.csv")
    r2_df.to_csv(r2_csv, index=False)
    print(f"[SAVE] {r2_csv} (shape={r2_df.shape})")
