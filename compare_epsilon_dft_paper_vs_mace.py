"""
Compare ε(iξ) errors vs DFT (KKR ground truth): Gudarzi MHO vs FrequencyMACE.

Reports two evaluation regimes:

  A) Full KKR grid — includes low-ξ where bandgap-only MHO is not designed to match.
  B) Fair region ξ > Eg — electronic-dominated frequencies only (fair for MHO).

MHO variants:
  - Eg only     : C_UV = A·Eg^{-1.2}, ω_UV from Eg (weakest baseline)
  - Eg + ε∞ DFT : C_UV = ε_DFT(iξ_min) − 1, ω_UV from Eg (fair crossover: one DFT point)

Usage:
    python3 compare_epsilon_dft_paper_vs_mace.py --model_path models/mace_kkr_v1.pt
"""

import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from compare_hamaker_constants import get_split_indices
from gudarzi_mho import alpha_from_eq7, eps_mho_from_bandgap
from lmdb_dataset import LMDBCrystalDataset, get_collate_fn
from model_forward import model_forward
from model_loader import load_model

PERCENTILES = [5, 25, 50, 75, 95]


def load_mace(path, device):
    ckpt = torch.load(path, map_location=device)
    model, model_type = load_model(ckpt, device)
    return ckpt, model, model_type


@torch.no_grad()
def predict_mace(model, model_type, loader, device, use_log):
    pred_list, true_list = [], []
    for inputs, target, _ in loader:
        target = target.to(device)
        pred_lin = model_forward(model, model_type, inputs, device, use_log=use_log)
        pred_list.append(pred_lin.cpu().numpy())
        true_list.append(target[:, 0, :].cpu().numpy())
    return np.concatenate(pred_list), np.concatenate(true_list)


def per_crystal_mae(pred, true, xi=None, gaps=None):
    if xi is None:
        return np.mean(np.abs(pred - true), axis=1)
    out = np.empty(len(pred), dtype=np.float64)
    for i in range(len(pred)):
        mask = xi > gaps[i]
        if not mask.any():
            out[i] = np.nan
        else:
            out[i] = np.mean(np.abs(pred[i, mask] - true[i, mask]))
    return out


def per_crystal_abs_rel_pct(pred, true, xi=None, gaps=None):
    if xi is None:
        return 100.0 * np.mean(np.abs(pred - true) / (np.abs(true) + 1e-30), axis=1)
    out = np.empty(len(pred), dtype=np.float64)
    for i in range(len(pred)):
        mask = xi > gaps[i]
        if not mask.any():
            out[i] = np.nan
        else:
            t = true[i, mask]
            p = pred[i, mask]
            out[i] = 100.0 * np.mean(np.abs(p - t) / (np.abs(t) + 1e-30))
    return out


def per_crystal_log_mae(pred, true, xi=None, gaps=None):
    if xi is None:
        lp = np.log(np.clip(pred, 1e-30, None))
        lt = np.log(np.clip(true, 1e-30, None))
        return np.mean(np.abs(lp - lt), axis=1)
    out = np.empty(len(pred), dtype=np.float64)
    for i in range(len(pred)):
        mask = xi > gaps[i]
        if not mask.any():
            out[i] = np.nan
        else:
            lp = np.log(np.clip(pred[i, mask], 1e-30, None))
            lt = np.log(np.clip(true[i, mask], 1e-30, None))
            out[i] = np.mean(np.abs(lp - lt))
    return out


def summarize(name, values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"name": name, "n": 0}
    pct = np.percentile(values, PERCENTILES)
    return {
        "name": name,
        "n": len(values),
        "mean": float(values.mean()),
        "percentiles": {p: float(v) for p, v in zip(PERCENTILES, pct)},
    }


def print_table(rows, metric_label):
    rows = [r for r in rows if r.get("n", 0) > 0]
    if not rows:
        print(f"\n{metric_label}: no valid samples")
        return
    cols = [r["name"] for r in rows]
    w = max(28, max(len(c) for c in cols) + 2)
    print(f"\n{metric_label}")
    print(f"  {'stat':<12}" + "".join(f"{c:>{w}}" for c in cols))
    print(f"  {'-'*12}" + "-" * w * len(cols))
    print(f"  {'mean':<12}" + "".join(f"{r['mean']:>{w}.4f}" for r in rows))
    for p in PERCENTILES:
        line = f"  {f'p{p}':<12}"
        for r in rows:
            line += f"{r['percentiles'][p]:>{w}.4f}"
        print(line)
    print(f"  {'n':<12}" + "".join(f"{r['n']:>{w}d}" for r in rows))


def build_metric_tables(methods, true, xi, gaps):
    """methods: list of (label, pred_array). Returns three table row lists."""
    mae_rows, rel_rows, log_rows = [], [], []
    for label, pred, use_xi_mask in methods:
        if pred is None:
            continue
        kw = {"xi": xi, "gaps": gaps} if use_xi_mask else {}
        mae_rows.append(summarize(label, per_crystal_mae(pred, true, **kw)))
        rel_rows.append(summarize(label, per_crystal_abs_rel_pct(pred, true, **kw)))
        log_rows.append(summarize(label, per_crystal_log_mae(pred, true, **kw)))
    return mae_rows, rel_rows, log_rows


def print_section(title, methods, true, xi, gaps, use_xi_mask):
    mae, rel, log = build_metric_tables(
        [(lbl, p, use_xi_mask) for lbl, p in methods],
        true, xi, gaps,
    )
    print(f"\n{'=' * 72}")
    print(title)
    print("=" * 72)
    print_table(mae, "MAE on ε(iξ)  (lower is better)")
    print_table(rel, "Mean |pred − DFT| / |DFT| × 100  (%)")
    print_table(log, "Mean |log ε_pred − log ε_DFT|  (lower is better)")


def main():
    parser = argparse.ArgumentParser(
        description="ε(iξ) error vs DFT: Gudarzi MHO vs MACE (full grid + fair ξ>Eg)",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, default=None)
    parser.add_argument("--lmdb_path", type=str, default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gap_field", type=str, default="opt_gap", choices=("opt_gap", "mbj_gap"))
    parser.add_argument("--metal_gap_threshold", type=float, default=0.1)
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Fixed α for all materials; default uses Eq. 7 with n=√ε∞",
    )
    parser.add_argument("--a_pref", type=float, default=20.0)
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, model, model_type = load_mace(args.model_path, device)
    use_log = ckpt.get("use_log", False)
    use_mace = model_type == "MACE"
    target_components = tuple(ckpt.get("target_components", ("xx",)))

    meta_pred = None
    meta_name = None
    ckpt_m = None
    model_type_m = None
    if args.meta_path:
        ckpt_m, model_m, model_type_m = load_mace(args.meta_path, device)
        meta_name = ckpt_m.get("model_name", os.path.basename(args.meta_path))

    dataset = LMDBCrystalDataset(
        args.lmdb_path,
        target_components=target_components,
        use_kkr=True,
        metal_gap_threshold=ckpt.get("metal_gap_threshold"),
        use_mace=use_mace,
        mace_r_max=ckpt.get("mace_r_max"),
    )
    xi = np.asarray(dataset.kkr_xi_array, dtype=np.float64)
    split_idx = get_split_indices(ckpt, len(dataset), args.split)
    loader = DataLoader(
        Subset(dataset, split_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=get_collate_fn(use_mace=use_mace),
    )

    model_name = ckpt.get("model_name", os.path.basename(args.model_path))
    print("=" * 72)
    print(f"DFT reference : KKR ε(iξ) from {args.lmdb_path}")
    print(f"MHO Eg only   : C_UV = A·Eg^{{-1.2}}, ω_UV = B·Eg^{{0.736}}")
    print(f"MHO Eg+ε∞ DFT : C_UV = ε_DFT(ξ_min)−1, ω_UV from Eg  [fair crossover]")
    alpha_desc = f"fixed {args.alpha}" if args.alpha is not None else "Eq. 7 (n=√ε∞)"
    print(f"  gap={args.gap_field}  α={alpha_desc}  A={args.a_pref}  exclude gap < {args.metal_gap_threshold} eV")
    print(f"Backend        : {model_type}")
    print(f"MACE/CGCNN     : {args.model_path} ({model_name})")
    if args.meta_path:
        print(f"MACE meta     : {args.meta_path} ({meta_name})")
    print(f"Split         : {args.split}  n={len(split_idx)}")
    print("=" * 72)

    pred_mace, true = predict_mace(model, model_type, loader, device, use_log)
    if args.meta_path:
        meta_pred, _ = predict_mace(
            model_m, model_type_m, loader, device, ckpt_m.get("use_log", use_log),
        )

    mpids, formulas, gaps = [], [], []
    pred_eg_list, pred_epsinf_list, valid_mask = [], [], []

    for i, global_i in enumerate(split_idx):
        info = dataset.get_sample_info(global_i)
        mpids.append(info["mpid"])
        formulas.append(info["formula"])
        gap_raw = info.get(args.gap_field)
        gap = float(gap_raw) if gap_raw is not None else float("nan")
        gaps.append(gap)

        ok = np.isfinite(gap) and gap >= args.metal_gap_threshold
        if not ok:
            valid_mask.append(False)
            continue
        try:
            c_uv_eg = args.a_pref * (gap ** -1.2)
            eps_static = float(true[i, 0])
            alpha_eg = (
                args.alpha
                if args.alpha is not None
                else alpha_from_eq7(gap, c_uv_eg, formula=info["formula"])
            )
            alpha_epsinf = (
                args.alpha
                if args.alpha is not None
                else alpha_from_eq7(
                    gap, eps_static - 1.0, formula=info["formula"], eps_static=eps_static,
                )
            )
            pred_eg_list.append(
                eps_mho_from_bandgap(
                    xi, gap, formula=info["formula"],
                    alpha=alpha_eg, a_pref=args.a_pref,
                )
            )
            c_uv = eps_static - 1.0
            pred_epsinf_list.append(
                eps_mho_from_bandgap(
                    xi, gap, formula=info["formula"],
                    alpha=alpha_epsinf, c_uv=c_uv,
                )
            )
            valid_mask.append(True)
        except ValueError:
            valid_mask.append(False)

    dataset.close()

    valid_mask = np.array(valid_mask, dtype=bool)
    n_valid = int(valid_mask.sum())
    print(f"\nMHO on {n_valid}/{len(split_idx)} crystals (gap ≥ {args.metal_gap_threshold} eV)")

    pred_eg = np.stack(pred_eg_list)
    pred_epsinf = np.stack(pred_epsinf_list)
    true_v = true[valid_mask]
    gaps_v = np.array(gaps)[valid_mask]
    pred_mace_v = pred_mace[valid_mask]
    pred_meta_v = meta_pred[valid_mask] if meta_pred is not None else None

    def method_list():
        m = [
            ("MHO (Eg only)", pred_eg),
            ("MHO (Eg + ε∞ DFT)", pred_epsinf),
            (f"MACE ({model_name})", pred_mace_v),
        ]
        if pred_meta_v is not None:
            m.append((f"MACE meta ({meta_name})", pred_meta_v))
        return m

    methods = method_list()

    print_section(
        "FULL KKR grid (all ξ) — bandgap-only MHO is weak here by design",
        methods, true_v, xi, gaps_v, use_xi_mask=False,
    )

    print_section(
        "FAIR: ξ > Eg only — electronic region; MHO + ε∞ crossover vs MACE",
        methods, true_v, xi, gaps_v, use_xi_mask=True,
    )

    n_xi_fair = [int((xi > g).sum()) for g in gaps_v]
    print(f"\n  ξ points per crystal above Eg: mean={np.mean(n_xi_fair):.0f}  "
          f"min={min(n_xi_fair)}  max={max(n_xi_fair)}  (of {len(xi)} total)")

    if args.out_csv:
        valid_indices = np.where(valid_mask)[0]
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "mpid", "formula", "gap_ev", "n_xi_above_eg",
                "mae_mho_eg_full", "mae_mho_epsinf_full", "mae_mace_full",
                "mae_mho_eg_xi_gt_eg", "mae_mho_epsinf_xi_gt_eg", "mae_mace_xi_gt_eg",
            ])
            for vi, idx in enumerate(valid_indices):
                w.writerow([
                    mpids[idx], formulas[idx], gaps_v[vi], int((xi > gaps_v[vi]).sum()),
                    per_crystal_mae(pred_eg[vi:vi + 1], true_v[vi:vi + 1])[0],
                    per_crystal_mae(pred_epsinf[vi:vi + 1], true_v[vi:vi + 1])[0],
                    per_crystal_mae(pred_mace_v[vi:vi + 1], true_v[vi:vi + 1])[0],
                    per_crystal_mae(pred_eg[vi:vi + 1], true_v[vi:vi + 1], xi=xi, gaps=gaps_v[vi:vi + 1])[0],
                    per_crystal_mae(pred_epsinf[vi:vi + 1], true_v[vi:vi + 1], xi=xi, gaps=gaps_v[vi:vi + 1])[0],
                    per_crystal_mae(pred_mace_v[vi:vi + 1], true_v[vi:vi + 1], xi=xi, gaps=gaps_v[vi:vi + 1])[0],
                ])
        print(f"\nSaved {args.out_csv}")


if __name__ == "__main__":
    main()
