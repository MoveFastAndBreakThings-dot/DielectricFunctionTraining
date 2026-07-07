"""
Meta-train (fine-tune) a pretrained FrequencyMACE to improve pairwise Hamaker prediction.

Loss is computed end-to-end: crystal graphs → ε(iξ) → differentiable Matsubara
pair Hamaker A_12, compared to ground truth from build_hamaker_pair_dataset.py.

Usage:
    python3 build_hamaker_pair_dataset.py
    python3 train_mace_meta_hamaker.py --backbone models/mace_kkr_v1.pt
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from hamaker_pair_dataset import HamakerPairDataset, get_hamaker_collate, load_pair_splits
from hamaker_torch import HamakerPairLoss
from lmdb_dataset import LMDBCrystalDataset
from model_forward import model_forward_train
from model_loader import load_model

MODEL_NAME = "mace_meta_hamaker_v1"


def set_trainable(model, freeze_encoder=False):
  if not freeze_encoder:
    for p in model.parameters():
      p.requires_grad = True
    return
  for name, p in model.named_parameters():
    p.requires_grad = name.startswith("fc_freq")


def run_pairs(model, model_type, inputs_a, inputs_b, device, use_log):
  """Forward both sides of each pair; return linear ε(iξ) for channel 0."""
  out_a = model_forward_train(model, model_type, inputs_a, device)[:, 0, :]
  out_b = model_forward_train(model, model_type, inputs_b, device)[:, 0, :]
  if use_log:
    return torch.exp(out_a), torch.exp(out_b)
  return out_a, out_b


def train_epoch(model, model_type, loader, criterion, optimizer, device, use_log, grad_clip):
  model.train()
  total, n = 0.0, 0
  for batch_idx, (inputs_a, inputs_b, a_true) in enumerate(loader):
    a_true = a_true.to(device)
    eps_a, eps_b = run_pairs(model, model_type, inputs_a, inputs_b, device, use_log)
    loss = criterion(eps_a, eps_b, a_true)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
      [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip,
    )
    optimizer.step()

    bs = a_true.shape[0]
    total += loss.item() * bs
    n += bs
    if batch_idx % 20 == 0:
      print(f"  batch {batch_idx}/{len(loader)}  loss={loss.item():.6f}")
  return total / max(n, 1)


@torch.no_grad()
def evaluate(model, model_type, loader, criterion, device, use_log):
  model.eval()
  total, n = 0.0, 0
  for inputs_a, inputs_b, a_true in loader:
    a_true = a_true.to(device)
    eps_a, eps_b = run_pairs(model, model_type, inputs_a, inputs_b, device, use_log)
    loss = criterion(eps_a, eps_b, a_true)
    bs = a_true.shape[0]
    total += loss.item() * bs
    n += bs
  return total / max(n, 1)


def plot_losses(train_losses, val_losses, path):
  plt.figure(figsize=(9, 5))
  epochs = range(1, len(train_losses) + 1)
  plt.plot(epochs, train_losses, label="Train")
  plt.plot(epochs, val_losses, label="Val")
  plt.xlabel("Epoch")
  plt.ylabel("Hamaker loss")
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(path, dpi=150)
  plt.close()


def main():
  parser = argparse.ArgumentParser(description="Meta-train MACE for pairwise Hamaker A_12")
  parser.add_argument("--backbone", type=str, required=True, help="Pretrained FrequencyMACE checkpoint")
  parser.add_argument("--pairs_dir", type=str, default="meta_hamaker_pairs")
  parser.add_argument("--lmdb_path", type=str, default=None,
                      help="LMDB path (default: read from pairs metadata.json)")
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--epochs", type=int, default=80)
  parser.add_argument("--lr", type=float, default=1e-5)
  parser.add_argument("--weight_decay", type=float, default=1e-5)
  parser.add_argument("--grad_clip", type=float, default=1.0)
  parser.add_argument("--freeze_encoder", action="store_true",
                      help="Only train fc_freq; freeze graph encoder")
  parser.add_argument("--n_matsubara", type=int, default=None)
  parser.add_argument("--loss", choices=("mae", "mse"), default="mae")
  parser.add_argument("--use_log_a", action="store_true",
                      help="Optimize log(A) instead of A (zJ)")
  parser.add_argument("--model_name", type=str, default=MODEL_NAME)
  args = parser.parse_args()

  meta_path = os.path.join(args.pairs_dir, "metadata.json")
  with open(meta_path) as f:
    meta = json.load(f)
  lmdb_path = args.lmdb_path or meta["lmdb_path"]
  n_matsubara = args.n_matsubara or meta["n_matsubara"]
  target_components = tuple(meta["target_components"])
  temperature_k = meta["temperature_k"]

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ckpt = torch.load(args.backbone, map_location=device)
  use_log = ckpt.get("use_log", False)
  use_mace = ckpt.get("model_type", "CGCNN") == "MACE"

  crystal_ds = LMDBCrystalDataset(
    lmdb_path,
    target_components=target_components,
    use_kkr=True,
    metal_gap_threshold=ckpt.get("metal_gap_threshold"),
    use_mace=use_mace,
    mace_r_max=ckpt.get("mace_r_max"),
  )
  splits = load_pair_splits(args.pairs_dir)
  pairs_npz = os.path.join(args.pairs_dir, "pairs.npz")
  hamaker_collate = get_hamaker_collate(use_mace=use_mace)

  train_ds = HamakerPairDataset(crystal_ds, pairs_npz, splits["train"])
  val_ds = HamakerPairDataset(crystal_ds, pairs_npz, splits["val"])
  test_ds = HamakerPairDataset(crystal_ds, pairs_npz, splits["test"])

  train_loader = DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True,
    num_workers=0, collate_fn=hamaker_collate,
  )
  val_loader = DataLoader(
    val_ds, batch_size=args.batch_size, shuffle=False,
    num_workers=0, collate_fn=hamaker_collate,
  )
  test_loader = DataLoader(
    test_ds, batch_size=args.batch_size, shuffle=False,
    num_workers=0, collate_fn=hamaker_collate,
  )

  model, model_type = load_model(ckpt, device)
  set_trainable(model, freeze_encoder=args.freeze_encoder)
  n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Backbone: {args.backbone}  type={model_type}  use_log={use_log}  trainable params: {n_trainable:,}")
  print(f"Pairs: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

  criterion = HamakerPairLoss(
    crystal_ds.kkr_xi_array,
    n_matsubara=n_matsubara,
    temperature_k=temperature_k,
    reduction=args.loss,
    use_log_a=args.use_log_a,
  ).to(device)

  optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.lr,
    weight_decay=args.weight_decay,
  )
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8)

  os.makedirs("models", exist_ok=True)
  model_path = f"models/{args.model_name}.pt"
  best_val = float("inf")
  train_losses, val_losses = [], []

  for epoch in range(args.epochs):
    print(f"\nEpoch {epoch + 1}/{args.epochs}")
    tr = train_epoch(model, model_type, train_loader, criterion, optimizer, device, use_log, args.grad_clip)
    va = evaluate(model, model_type, val_loader, criterion, device, use_log)
    scheduler.step(va)
    train_losses.append(tr)
    val_losses.append(va)
    print(f"  train={tr:.6f}  val={va:.6f}")

    if va < best_val:
      best_val = va
      torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": va,
        "model_type": model_type,
        "backbone_path": args.backbone,
        "pairs_dir": args.pairs_dir,
        "meta_objective": "hamaker_pair",
        "model_name": args.model_name,
        "use_log": use_log,
        "use_log_a": args.use_log_a,
        "freeze_encoder": args.freeze_encoder,
        "n_low_freq": ckpt.get("n_low_freq", 0),
        "metal_low_freq_weight": ckpt.get("metal_low_freq_weight", 0.1),
        "metal_gap_threshold": ckpt.get("metal_gap_threshold"),
        "n_matsubara": n_matsubara,
        "temperature_k": temperature_k,
        "mace_r_max": ckpt.get("mace_r_max"),
        "mace_num_interactions": ckpt.get("mace_num_interactions"),
        "mace_hidden_irreps": ckpt.get("mace_hidden_irreps"),
        "mace_max_ell": ckpt.get("mace_max_ell"),
        "mace_correlation": ckpt.get("mace_correlation"),
        "orig_atom_fea_len": ckpt.get("orig_atom_fea_len"),
        "nbr_fea_len": ckpt.get("nbr_fea_len"),
        "n_frequencies": ckpt["n_frequencies"],
        "n_output_channels": ckpt["n_output_channels"],
        "target_components": target_components,
        "atom_fea_len": ckpt.get("atom_fea_len", 128),
        "n_layers": ckpt.get("n_layers", 4),
        "h_fea_len": ckpt.get("h_fea_len", 256),
        "n_h": ckpt.get("n_h", 1),
        "dropout": ckpt.get("dropout", 0.1),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_idx": ckpt.get("train_idx"),
        "val_idx": ckpt.get("val_idx"),
        "test_idx": ckpt.get("test_idx"),
      }, model_path)
      print(f"  -> saved {model_path} (val={va:.6f})")

  plot_losses(train_losses, val_losses, f"models/{args.model_name}_loss.png")
  test_loss = evaluate(model, model_type, test_loader, criterion, device, use_log)
  print(f"\nTest Hamaker loss: {test_loss:.6f}")
  crystal_ds.close()


if __name__ == "__main__":
  main()
