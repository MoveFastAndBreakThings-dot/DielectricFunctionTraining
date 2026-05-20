"""
Plot predicted vs actual ε(iξ) for a single material identified by its MP-ID.

Edit the two constants below, then run:
    python3 plot_single_material.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lmdb_dataset import LMDBCrystalDataset, collate_pool

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG  –  only these two lines need changing
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/mace_kkr_v2.pt"
MPID       = "mp-1234"
# ─────────────────────────────────────────────────────────────────────────────

LMDB_PATH = "processed_data/crystal_graphs.lmdb"
OUT_DIR   = "plots_single"


def _load_model(ckpt, device):
    """Load whichever model variant is stored in the checkpoint."""
    has_abc = "abc_hidden" in ckpt or any(
        k.startswith("fc_abc") for k in ckpt.get("model_state_dict", {})
    )
    if has_abc:
        from mace_model_error_params import FrequencyMACEWithABC
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
    else:
        from mace_model import FrequencyMACE
        model = FrequencyMACE(
            orig_atom_fea_len=ckpt["orig_atom_fea_len"],
            nbr_fea_len=ckpt["nbr_fea_len"],
            atom_fea_len=ckpt.get("atom_fea_len", 128),
            n_layers=ckpt.get("n_layers", 4),
            h_fea_len=ckpt.get("h_fea_len", 256),
            n_h=ckpt.get("n_h", 1),
            n_frequencies=ckpt["n_frequencies"],
            n_output_channels=ckpt["n_output_channels"],
            dropout=ckpt.get("dropout", 0.1),
        ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, has_abc


def _build_mpid_index(dataset, cache_path):
    """Scan dataset once and write mpid -> dataset index to a JSON cache."""
    import json
    print(f"Building MPID index for {len(dataset)} samples (one-time, will be cached)…")
    mapping = {}
    for i in range(len(dataset)):
        mapping[dataset.get_sample_info(i)["mpid"]] = i
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(dataset)}")
    with open(cache_path, "w") as f:
        json.dump(mapping, f)
    print(f"  Index cached to {cache_path}")
    return mapping


def find_mpid_index(dataset, mpid, lmdb_path):
    import json
    cache_path = lmdb_path + ".mpid_index.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            mapping = json.load(f)
    else:
        mapping = _build_mpid_index(dataset, cache_path)
    return mapping.get(mpid)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(MODEL_PATH, map_location=device)
    model_name = ckpt.get("model_name", os.path.splitext(os.path.basename(MODEL_PATH))[0])
    use_log = ckpt.get("use_log", False)
    target_components = ckpt.get("target_components", ("xx",))

    model, has_abc = _load_model(ckpt, device)

    dataset = LMDBCrystalDataset(
        LMDB_PATH,
        target_components=target_components,
        use_kkr=True,
        metal_gap_threshold=ckpt.get("metal_gap_threshold"),
    )
    xi = np.asarray(dataset.kkr_xi_array, dtype=np.float64)

    print(f"Model : {model_name}  (use_log={use_log}, abc={has_abc})")
    print(f"MPID  : {MPID}")
    idx = find_mpid_index(dataset, MPID, LMDB_PATH)
    if idx is None:
        print(f"ERROR: '{MPID}' not found in {LMDB_PATH}")
        dataset.close()
        return

    info = dataset.get_sample_info(idx)
    formula = info["formula"]
    print(f"Found at index {idx}  ({formula})")

    loader = DataLoader(
        Subset(dataset, [idx]),
        batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_pool,
    )

    with torch.no_grad():
        (af, nf, nfi, cai), target, _ = next(iter(loader))
        af = af.to(device); nf = nf.to(device)
        nfi = nfi.to(device); cai = [x.to(device) for x in cai]
        target = target.to(device)

        out = model(af, nf, nfi, cai)
        eps_raw = out[0] if has_abc else out
        eps_pred = np.exp(eps_raw.cpu().numpy()[0, 0]) if use_log else eps_raw.cpu().numpy()[0, 0]

    eps_true = target.cpu().numpy()[0, 0]
    mae = float(np.mean(np.abs(eps_pred - eps_true)))

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, log_x in zip(axes, [True, False]):
        ax.plot(xi, eps_pred, color="C0", lw=1.8, label="Predicted")
        ax.plot(xi, eps_true, color="C1", lw=1.8, ls="--", label="Actual")
        if log_x:
            ax.set_xscale("log")
            ax.set_title("Log ξ axis")
        else:
            ax.set_title("Linear ξ axis")
        ax.set_xlabel("ξ (eV)")
        ax.set_ylabel("ε(iξ)")
        ax.set_ylim(bottom=0.9)
        ax.legend()

    fig.suptitle(
        f"{MPID}  {formula}  |  MAE = {mae:.4f}  |  model: {model_name}",
        fontsize=11,
    )
    plt.tight_layout()

    safe_mpid = MPID.replace("/", "-").replace(" ", "_")
    out_path = os.path.join(OUT_DIR, f"{safe_mpid}_{model_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    dataset.close()
    print(f"Saved: {out_path}  (MAE={mae:.4f})")


if __name__ == "__main__":
    main()
