                      
"""
FedHGF ablation runner.

Examples:
  python run_ablation.py --datasets wadi hai batadal_small --seeds 42,123,2024 --device cuda
  python run_ablation.py --datasets wadi --variants "Flow Only" "Full FedHGF"

Outputs:
  - Console text table
  - LaTeX table
  - results_ablation/ablation_<timestamp>.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_this = Path(__file__).resolve().parent
_root = _this.parent
sys.path.insert(0, str(_this))
sys.path.insert(1, str(_root))

from run_fair_comparison import load_clients, run_fedhgf, set_seed

                                                                                
                                                               
                                                                                
DATASET_BEST_CFG: Dict[str, dict] = {
                                                                          
    "wadi": {
        "w_fusion": (0.20, 0.55, 0.25),
        "fusion_mode": "fixed",
        "cfg_extra_base": {},
    },
                                                                           
    "batadal": {
        "w_fusion": (0.20, 0.55, 0.25),
        "fusion_mode": "fixed",
        "cfg_extra_base": {},
    },
                                                                               
    "batadal_small": {
        "w_fusion": (0.20, 0.55, 0.25),
        "fusion_mode": "fixed",
        "label_mode": "last",
        "cfg_extra_base": {},
    },
                                                                           
    "hai": {
        "w_fusion": (0.20, 0.55, 0.25),
        "fusion_mode": "fixed",
        "cfg_extra_base": {},
    },
                                                                          
    "swat": {
        "w_fusion": (0.20, 0.55, 0.25),
        "fusion_mode": "fixed",
        "cfg_extra_base": {},
    },
}

                                                      
DATASET_DISPLAY: Dict[str, str] = {
    "swat":          "SWaT",
    "wadi":          "WADI",
    "hai":           "HAI 21.03",
    "batadal":       "BATADAL",
    "batadal_small": "BATADAL",
}


                                                                                
                                                         
                                                                                
def get_ablation_variants(dataset: str) -> "OrderedDict[str, dict]":
    """
    Returns an ordered dict of {variant_label: cfg_override_dict}.

    7-variant ablation design covering 4 contribution lines:
    ─────────────────────────────────────────────────────────────────
    Line 1 — Local flow density scoring:
      Flow Only       : w=(0,1,0), full training, only s2 at inference
                        → is flow density the core anomaly evidence?
      w/o Flow Score  : use_flow=False, w redistributed to s1+s3
                        → how much does the model lose without flow?

    Line 2 — Graph structure (encoder + scoring, separately):
      w/o Graph Score : keep graph encoder, remove s3 residual score
                        → is graph residual scoring branch useful?
      w/o Graph Enc   : remove graph from encoder, keep s3 residual score
                        → does graph attention help representation learning?

    Line 3 — Heterogeneous auxiliary variables:
      w/o Cross Block : B_k=C_k=0, keep aux features but no coupling
                        → is anchor-aux coupling structure effective?
      Anchor Only     : strip all aux features, anchor nodes only
                        → are heterogeneous aux variables valuable?

    Line 4 — Latent collapse prevention:
      w/o Var Floor   : lambda_v=0, remove variance floor regularizer
                        → does var-floor prevent center-alignment collapse?

    Special keys in the override dict:
      _anchor_only: bool  → strip aux features before running
    """
    best = DATASET_BEST_CFG.get(dataset, {})
    w    = best.get("w_fusion", (0.20, 0.55, 0.25))
    fm   = best.get("fusion_mode", "fixed")
    base = dict(best.get("cfg_extra_base", {}))                        

    w1, w2, w3 = w

                                                
    ws12 = w1 + w2 or 1e-8
    ws13 = w1 + w3 or 1e-8
    w_no_s3  = (round(w1 / ws12, 4), round(w2 / ws12, 4), 0.0)
    w_no_s2  = (round(w1 / ws13, 4), 0.0, round(w3 / ws13, 4))

    def _merge(**kw):
        """Merge base cfg with variant-specific overrides."""
        d = dict(base)
        d.update(kw)
        return d

    return OrderedDict([
                                                                            
                                                                             
                                                              
        ("Flow Only", _merge(
            w_fusion=(0.0, 1.0, 0.0),
            fusion_mode="fixed",
        )),
                                                                         
        ("w/o Flow Score", _merge(
            use_flow=False,
            w_fusion=w_no_s2,
            fusion_mode="fixed",
        )),

                                                                            
                                                                    
                                                              
        ("w/o Graph Score", _merge(
            use_graph_residual=False,
            w_fusion=w_no_s3,
            fusion_mode="fixed",
        )),
                                                                   
                                                                      
                                                                     
        ("w/o Graph Enc", _merge(
            graph_in_encoder=False,
            lambda_g=0.0,
            w_fusion=w,
            fusion_mode=fm,
        )),

                                                                            
                                                                  
                                                                    
                                                         
        ("w/o Cross Block", _merge(
            use_data_driven_cross_block=False,
            w_fusion=w,
            fusion_mode=fm,
        )),
                                                                        
                                                                       
        ("Anchor Only", _merge(
            _anchor_only=True,
            w_fusion=w,
            fusion_mode=fm,
        )),

                                                                           
                                                                          
                                                                         
                                       
        ("w/o Var Floor", _merge(
            lambda_v=0.0,
            w_fusion=w,
            fusion_mode=fm,
        )),

                                                                           
        ("Full FedHGF", _merge(
            w_fusion=w,
            fusion_mode=fm,
        )),
    ])


                                                                                
                                                            
                                                                                
def make_anchor_only_clients(clients: list, n_anchor: int) -> list:
    """
    Return a new client list where each client's X arrays are sliced to the
    first n_anchor feature columns (anchor features only).

    Assumes the first n_anchor columns in the feature dimension correspond to
    anchor variables (standard layout from load_clients).
    """
    result = []
    for c in clients:
        cc = dict(c)
        cc["n_k"]       = n_anchor
        cc["aux_names"] = []
        for key in ("X_train", "X_cal", "X_test"):
            if key in cc and cc[key] is not None:
                x = cc[key]                         
                cc[key] = x[..., :n_anchor, :]
        result.append(cc)
    return result


                                                                                
                                
                                                                                
def run_variant(
    clients: list,
    n_anchor: int,
    dataset: str,
    seed: int,
    device: str,
    variant_name: str,
    variant_cfg: dict,
) -> dict:
    """Run one ablation variant; return overall metrics dict."""
                                                                
    anchor_only = variant_cfg.pop("_anchor_only", False)

    w_fusion    = variant_cfg.pop("w_fusion",    None)
    fusion_mode = variant_cfg.pop("fusion_mode", None)
    cfg_extra   = dict(variant_cfg)                               

    actual_clients = clients
    if anchor_only:
        actual_clients = make_anchor_only_clients(clients, n_anchor)
        cfg_extra.setdefault("use_data_driven_cross_block", False)

    print(f"\n  -- Variant: {variant_name!r} "
          f"| w={w_fusion}  fm={fusion_mode}  extra={cfg_extra} --")

    overall, _ = run_fedhgf(
        actual_clients, n_anchor, dataset, seed,
        fedhgf_full=False, device=device,
        w_fusion=w_fusion,
        fusion_mode=fusion_mode,
        cfg_extra=cfg_extra if cfg_extra else None,
    )
    return overall


                                                                                
                             
                                                                                
def _fmt(v, decimals=4):
    """Format a float or list-of-floats (mean ± std)."""
    if v is None:
        return "---"
    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            return f"{v[0]:.{decimals}f}"
        mu  = float(np.mean(v))
        std = float(np.std(v))
        return f"{mu:.{decimals}f}±{std:.{decimals}f}"
    return f"{float(v):.{decimals}f}"


def print_text_table(results: dict, datasets: list, variants: list):
    """
    results[dataset][variant] = {"auroc": float, "f1": float}
    """
    col_w = 12
    header = f"{'Setting':<28}"
    for ds in datasets:
        dname = DATASET_DISPLAY.get(ds, ds.upper())
        header += f"  {dname:>{col_w*2+2}}"
    print("\n" + "=" * (28 + len(datasets) * (col_w * 2 + 4)))
    print("  FedHGF Ablation Study")
    print("=" * (28 + len(datasets) * (col_w * 2 + 4)))
    print(f"  {'Setting':<26}", end="")
    for ds in datasets:
        print(f"  {'AUROC':>{col_w}}  {'F1':>{col_w}}", end="")
    print()
    print("-" * (28 + len(datasets) * (col_w * 2 + 4)))
    for v in variants:
        print(f"  {v:<26}", end="")
        for ds in datasets:
            r = results.get(ds, {}).get(v)
            if r:
                auroc_s = _fmt(r.get("auroc"))
                f1_s    = _fmt(r.get("f1"))
            else:
                auroc_s = f1_s = "---"
            print(f"  {auroc_s:>{col_w}}  {f1_s:>{col_w}}", end="")
        print()
    print("=" * (28 + len(datasets) * (col_w * 2 + 4)))


def print_latex_table(results: dict, datasets: list, variants: list):
    """Print a LaTeX tabular block compatible with the paper format."""
    ds_headers = " & ".join(
        f"\\multicolumn{{2}}{{c|}}{{{DATASET_DISPLAY.get(ds, ds.upper())}}}"
        if i < len(datasets) - 1 else
        f"\\multicolumn{{2}}{{c}}{{{DATASET_DISPLAY.get(ds, ds.upper())}}}"
        for i, ds in enumerate(datasets)
    )
    col_spec = "l|" + "cc|" * (len(datasets) - 1) + "cc"
    auroc_f1 = " & ".join(["AUROC & F1"] * len(datasets))

    print("\n% -- LaTeX Table (copy into paper) ------------------------------")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{Ablation study of FedHGF.}")
    print(r"\label{tab:ablation_FedHGF_main}")
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print(r"\toprule")
    print(f"\\multirow{{2}}{{*}}{{Setting}}")
    for i, ds in enumerate(datasets):
        sep = "c|" if i < len(datasets) - 1 else "c"
        print(f"& \\multicolumn{{2}}{{{sep}}}{{{DATASET_DISPLAY.get(ds, ds.upper())}}}")
    print(r"\\")
    cmidr = " ".join(
        f"\\cmidrule(lr){{{2+i*2}-{3+i*2}}}"
        for i in range(len(datasets))
    )
    print(cmidr)
    print(f"& {auroc_f1} \\\\")
    print(r"\midrule")

    sep_before = {"w/o Graph Score", "w/o Cross Block", "w/o Var Floor", "Full FedHGF"}

    for v in variants:
        if v in sep_before:
            print(r"\midrule")
        bold = v == "Full FedHGF"
        label = f"\\textbf{{{v}}}" if bold else v
        row = f"{label}"
        for ds in datasets:
            r = results.get(ds, {}).get(v)
            if r:
                auroc_v = r.get("auroc")
                f1_v    = r.get("f1")
                auroc_s = f"{float(np.mean(auroc_v) if isinstance(auroc_v, list) else auroc_v):.4f}"
                f1_s    = f"{float(np.mean(f1_v) if isinstance(f1_v, list) else f1_v):.4f}"
                if bold:
                    auroc_s = f"\\textbf{{{auroc_s}}}"
                    f1_s    = f"\\textbf{{{f1_s}}}"
            else:
                auroc_s = f1_s = "---"
            row += f"\n  & {auroc_s} & {f1_s}"
        row += " \\\\"
        print(row)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")
    print("% ---------------------------------------------------------------")


def save_csv(results: dict, datasets: list, variants: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts  = time.strftime("%Y%m%d_%H%M%S")
    fpath = out_dir / f"ablation_{ts}.csv"
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Setting"]
        for ds in datasets:
            header += [f"{ds}_AUROC", f"{ds}_F1"]
        writer.writerow(header)
        for v in variants:
            row = [v]
            for ds in datasets:
                r = results.get(ds, {}).get(v)
                if r:
                    auroc_v = r.get("auroc")
                    f1_v    = r.get("f1")
                    row += [
                        float(np.mean(auroc_v) if isinstance(auroc_v, list) else auroc_v),
                        float(np.mean(f1_v)    if isinstance(f1_v,    list) else f1_v),
                    ]
                else:
                    row += ["", ""]
            writer.writerow(row)
    print(f"\n  Saved CSV: {fpath}")
    return fpath


                                                                                
      
                                                                                
def main():
    ap = argparse.ArgumentParser(description="FedHGF Ablation Study Runner")
    ap.add_argument("--datasets", nargs="+", default=["swat", "wadi", "hai", "batadal_small"],
                    help="Datasets to run (e.g. swat wadi hai batadal_small)")
    ap.add_argument("--seeds",  type=str, default="42",
                    help="Comma-separated seeds (e.g. 42,123,2024)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="Variant names to run (default: all). "
                         "Example: --variants 'Flow Only' 'Full FedHGF'")
    ap.add_argument("--latex-only", action="store_true",
                    help="Skip running; only print LaTeX table (requires prior results in memory)")
    args = ap.parse_args()

    seeds   = [int(s) for s in args.seeds.split(",")]
    datasets = [ds.lower() for ds in args.datasets]

    print("=" * 70)
    print("  FedHGF Ablation Study")
    print(f"  datasets={datasets}  seeds={seeds}  device={args.device}")
    print("=" * 70)

                                                                  
    results: Dict[str, Dict[str, Dict]] = {ds: {} for ds in datasets}

    for ds in datasets:
        print(f"\n{'='*70}")
        print(f"  Dataset: {DATASET_DISPLAY.get(ds, ds.upper())}")
        print(f"{'='*70}")

        all_variants = get_ablation_variants(ds)

                                           
        if args.variants:
            all_variants = OrderedDict(
                (k, v) for k, v in all_variants.items()
                if k in args.variants
            )
        variant_names = list(all_variants.keys())

        for vname in variant_names:
            auroc_seeds, f1_seeds = [], []
            for seed in seeds:
                set_seed(seed)
                ds_label_mode = DATASET_BEST_CFG.get(ds, {}).get("label_mode")
                clients, n_anchor, anchor_cols = load_clients(
                    ds, seed, label_mode=ds_label_mode)
                print(f"\n  seed={seed}: {len(clients)} clients, n_anchor={n_anchor}")

                                                                               
                vcfg = deepcopy(all_variants[vname])
                t0   = time.time()
                try:
                    overall = run_variant(
                        clients, n_anchor, ds, seed, args.device,
                        variant_name=vname, variant_cfg=vcfg,
                    )
                    auroc_seeds.append(overall.get("auroc", 0.0))
                    f1_seeds.append(overall.get("f1",    0.0))
                    elapsed = time.time() - t0
                    print(f"  [OK] {vname!r} seed={seed} "
                          f"AUROC={auroc_seeds[-1]:.4f}  F1={f1_seeds[-1]:.4f}  "
                          f"({elapsed:.0f}s)")
                except Exception as e:
                    print(f"  [FAIL] {vname!r} seed={seed} FAILED: {e}")

            if auroc_seeds:
                results[ds][vname] = {
                    "auroc": auroc_seeds if len(auroc_seeds) > 1 else auroc_seeds[0],
                    "f1":    f1_seeds    if len(f1_seeds)    > 1 else f1_seeds[0],
                }

                                                                            
    all_variant_names = list(get_ablation_variants(datasets[0]).keys())
    if args.variants:
        all_variant_names = [v for v in all_variant_names if v in args.variants]

    print_text_table(results, datasets, all_variant_names)
    print_latex_table(results, datasets, all_variant_names)

    out_dir = _this / "results_ablation"
    save_csv(results, datasets, all_variant_names, out_dir)


if __name__ == "__main__":
    main()
