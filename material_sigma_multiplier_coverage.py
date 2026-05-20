"""
Per-material sigma-band coverage (all frequencies at once).

For each crystal, the model predicts ε(iξ) and (b, c) with
  σ(ξ) = b / (ξ + c)^α.

Residual space (must match how you interpret the model):

  **linear** — r(ξ) = |ε̂(ξ) − ε(ξ)| with ε̂ in linear space (exp of network
  output when the checkpoint was trained with use_log). Compare to σ(ξ) from
  (b, c) in the same linear ε units. This matches SpectrumAndPowerLawABCLoss’s
  sigma term, which always uses linear |ε̂ − ε|.

  **log** — r(ξ) = |log ε̂ − log ε| (natural log, with a floor on ε to avoid
  −inf). The power-law σ is still defined in linear ε units, so we compare to
  an effective log-scale band width σ_eff(ξ) = σ(ξ) / max(ε(ξ), ε_floor)
  (first-order: multiplicative uncertainty).

  **auto** — log if checkpoint has use_log else linear.

If your checkpoint omits use_log, pass --error_space explicitly or
--use_log / --no_use_log to match training.

Usage:
  python3 material_sigma_multiplier_coverage.py --model_path models/mace_kkr_with_abc_v3.pt
  python3 material_sigma_multiplier_coverage.py --model_name mace_kkr_with_abc_v3
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lmdb_dataset import LMDBCrystalDataset, collate_pool
from mace_model_error_params import FrequencyMACEWithABC


def load_model_from_checkpoint(ckpt, device):
    model = FrequencyMACEWithABC(
        orig_atom_fea_len=ckpt["orig_atom_fea_len"],
        nbr_fea_len=ckpt["nbr_fea_len"],
        atom_fea_len=ckpt.get("atom_fea_len", 128),
        n_layers=ckpt.get("n_layers", 4),
        h_fea_len=ckpt.get("h_fea_len", 256),
        n_h=ckpt.get("n_h", 1),
        n_frequencies=ckpt["n_frequencies"],
        n_output_channels=ckpt["n_output_channels"],
        dropout=ckpt.get("dropout", 0.1),
        abc_hidden=ckpt.get("abc_hidden", 128),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def sigma_curve(xi, b, c, alpha, sigma_floor):
    """σ(ξ) = b / (ξ + c)^α, shape (F,) for scalars b, c."""
    xi = np.asarray(xi, dtype=np.float64)
    c = max(float(c), sigma_floor)
    b = float(b)
    denom = (xi + c) ** float(alpha)
    s = b / np.maximum(denom, sigma_floor)
    return np.maximum(s, sigma_floor)


def main():
    parser = argparse.ArgumentParser(
        description="k* = max |ε_pred−ε_true|/σ per material; coverage vs 1σ, 1.5σ, …"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to .pt checkpoint (FrequencyMACEWithABC)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Basename under models/ if --model_path not set (e.g. mace_kkr_with_abc_v3)",
    )
    parser.add_argument("--lmdb_path", type=str, default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("test", "val", "train", "all"),
        help="Which indices from checkpoint to use (default: test)",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="1,1.5,2,2.5,3,4,5",
        help="Comma-separated multipliers m (material covered if k* <= m)",
    )
    parser.add_argument(
        "--sigma_floor",
        type=float,
        default=1e-12,
        help="Absolute floor on σ to avoid div by zero",
    )
    parser.add_argument(
        "--sigma_floor_rel",
        type=float,
        default=0.01,
        help="Per-crystal relative sigma floor as a fraction of σ at ξ=0 "
             "(b/c^α). Frequencies where σ < sigma_floor_rel * σ_peak are "
             "excluded from k* — avoids huge ratios in the high-ξ tail where "
             "σ≈0 but tiny numerical residuals remain. Set 0 to disable.",
    )
    parser.add_argument(
        "--metal_eps_threshold",
        type=float,
        default=float("inf"),
        help="If finite, drop crystals with eps_true(lowest ξ) above this (metal proxy; "
        "default inf = keep all)",
    )
    parser.add_argument(
        "--error_space",
        type=str,
        choices=("auto", "linear", "log"),
        default="auto",
        help="auto: log residuals if ckpt use_log else linear (see docstring)",
    )
    parser.add_argument(
        "--eps_floor",
        type=float,
        default=1e-12,
        help="Floor on ε in log() and log-space σ_eff denominator",
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--use_log",
        action="store_true",
        help="Treat model output as log(ε) even if missing from checkpoint",
    )
    g.add_argument(
        "--no_use_log",
        action="store_true",
        help="Treat model output as linear ε even if ckpt says use_log",
    )
    args = parser.parse_args()

    if args.model_path:
        path = args.model_path
    elif args.model_name:
        path = os.path.join("models", args.model_name if args.model_name.endswith(".pt") else f"{args.model_name}.pt")
    else:
        raise SystemExit("Provide --model_path or --model_name")

    if not os.path.isfile(path):
        raise SystemExit(f"Checkpoint not found: {path}")

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    thresholds = sorted(set(thresholds))

    ckpt = torch.load(path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(ckpt, device)

    use_log = ckpt.get("use_log", False)
    if args.use_log:
        use_log = True
    if args.no_use_log:
        use_log = False

    # σ(ξ) = b/(ξ+c)^α is always trained against linear |ε̂ − ε|, regardless of
    # use_log. "auto" therefore always means linear — matching what the plot scripts show.
    if args.error_space == "auto":
        error_space = "linear"
    else:
        error_space = args.error_space

    alpha = float(ckpt.get("alpha", 3.0))
    sigma_transform = ckpt.get("sigma_transform", "log1p" if use_log else "none")
    target_components = ckpt.get("target_components", ("xx",))
    metal_gap_threshold = ckpt.get("metal_gap_threshold", None)

    dataset = LMDBCrystalDataset(
        args.lmdb_path,
        target_components=target_components,
        use_kkr=True,
        metal_gap_threshold=metal_gap_threshold,
    )
    xi = np.asarray(dataset.frequency_array, dtype=np.float64)

    if args.split == "all":
        indices = list(range(len(dataset)))
    else:
        key = {"test": "test_idx", "val": "val_idx", "train": "train_idx"}[args.split]
        indices = ckpt.get(key)
        if indices is None:
            raise SystemExit(f"Checkpoint has no {key}; use --split all or retrain with split saved.")

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pool,
    )

    k_star_list = []
    n_skipped_metal = 0

    with torch.no_grad():
        for inputs, target, _ in loader:
            af, nf, nfi, cai = inputs
            af = af.to(device)
            nf = nf.to(device)
            nfi = nfi.to(device)
            cai = [x.to(device) for x in cai]
            target = target.to(device)

            eps_out, bc_out = model(af, nf, nfi, cai)
            eps_out_np = eps_out.cpu().numpy()
            eps_true_np = target.cpu().numpy()
            bc_np = bc_out.cpu().numpy()

            for i in range(eps_out_np.shape[0]):
                if np.isfinite(args.metal_eps_threshold):
                    if eps_true_np[i, 0, 0] > args.metal_eps_threshold:
                        n_skipped_metal += 1
                        continue

                true = eps_true_np[i]
                raw = eps_out_np[i]
                if use_log:
                    pred_lin = np.exp(raw)
                else:
                    pred_lin = raw

                b_i, c_i = float(bc_np[i, 0]), float(bc_np[i, 1])
                sig_f = sigma_curve(xi, b_i, c_i, alpha, args.sigma_floor)

                # per-crystal relative floor: ignore high-ξ tail where σ≈0
                if args.sigma_floor_rel > 0:
                    sig_peak = b_i / max(c_i, args.sigma_floor) ** alpha
                    eff_floor = max(args.sigma_floor, args.sigma_floor_rel * sig_peak)
                else:
                    eff_floor = args.sigma_floor
                freq_mask = sig_f >= eff_floor

                if not np.any(freq_mask):
                    k_star_list.append(0.0)
                    continue

                if error_space == "linear":
                    resid = np.abs(pred_lin - true)
                else:
                    log_p = np.log(np.maximum(pred_lin, args.eps_floor))
                    log_t = np.log(np.maximum(true, args.eps_floor))
                    resid = np.abs(log_p - log_t)

                ratio = resid[..., freq_mask] / sig_f[freq_mask]
                k_star_list.append(float(np.max(ratio)))

    k_star = np.asarray(k_star_list, dtype=np.float64)
    n = len(k_star)
    n_bad = int(np.sum(~np.isfinite(k_star)))

    display_name = ckpt.get("model_name", os.path.basename(path))
    print(f"Checkpoint : {path}")
    print(f"Model name : {display_name}")
    print(f"Split      : {args.split}  (n={len(indices)} in subset, n_eval={n})")
    print(f"alpha            : {alpha}")
    print(f"use_log (ckpt+ov): {use_log}")
    print(f"sigma_transform  : {sigma_transform}  (training; sigma term uses linear |Δε|)")
    print(f"error_space      : {error_space}")
    print(f"sigma_floor_rel  : {args.sigma_floor_rel}  (ignore ξ where σ < this × σ_peak)")
    if n_skipped_metal:
        print(f"Skipped (metal filter): {n_skipped_metal}")
    if n_bad:
        print(f"Non-finite k* count: {n_bad}")

    k_ok = k_star[np.isfinite(k_star)]
    if len(k_ok) == 0:
        dataset.close()
        raise SystemExit("No valid materials after filtering.")

    if error_space == "linear":
        k_desc = "k* = max |ε̂−ε| / σ(ξ)"
    else:
        k_desc = "k* = max |log ε̂ − log ε| / σ(ξ)"

    print(f"\n--- {k_desc}  per material ---")
    pct = [0, 5, 25, 50, 75, 90, 95, 99, 100]
    qs = np.percentile(k_ok, pct)
    for p, q in zip(pct, qs):
        print(f"  p{p:3d}: {q:.4g}")

    band = "m·σ(ξ)"
    print(f"\n--- Fraction of materials with ALL ξ covered ({band}) ---")
    for m in thresholds:
        frac = float(np.mean(k_ok <= m))
        print(f"  m = {m:4g}  →  {100.0 * frac:6.2f}%  covered  ({int(np.sum(k_ok <= m))} / {len(k_ok)})")

    dataset.close()


if __name__ == "__main__":
    main()
