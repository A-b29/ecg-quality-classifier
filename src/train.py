import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path

from dataset import ECGQualityDataset
from model import ECGQualityCNN

# -- Paths (resolved relative to project root, cwd-independent) --------------
ROOT       = Path(__file__).resolve().parent.parent
RECORDS    = ROOT / 'data' / 'records.csv'
MODEL_PATH = ROOT / 'outputs' / 'best_model.pt'

# -- Config -----------------------------------------------------------------
BATCH_SIZE   = 32
EPOCHS       = 20
LR           = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 2
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # -- Data ----------------------------------------------------------------
    if not RECORDS.exists():
        raise SystemExit(f"{RECORDS} not found. Run `python build_manifest.py` first.")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RECORDS)
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )

    # Weighted random sampler -- handles class imbalance
    counts = np.bincount(train_df['label'], minlength=3)
    weights = 1.0 / np.clip(counts, 1, None)
    sample_weights = torch.as_tensor(
        [weights[l] for l in train_df['label']], dtype=torch.float
    )
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_ds = ECGQualityDataset(train_df, augment=True)
    val_ds   = ECGQualityDataset(val_df,   augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=64, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    # -- Model ---------------------------------------------------------------
    model     = ECGQualityCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # -- Training loop -------------------------------------------------------
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(1, EPOCHS + 1):
        # - Train -
        model.train()
        running_loss = 0.0
        for x, y in tqdm(train_dl, desc=f'Epoch {epoch:02d}/{EPOCHS}', leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        avg_loss = running_loss / len(train_dl)

        # - Validate -
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += len(y)
        val_acc = correct / total
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch:2d} | loss: {avg_loss:.4f} | val acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"          + saved best model (val acc = {val_acc:.3f})")

    print(f"\nBest validation accuracy: {best_val_acc:.3f}")


if __name__ == '__main__':
    main()
