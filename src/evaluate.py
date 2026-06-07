import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import ECGQualityDataset, CLASS_NAMES
from model import ECGQualityCNN

ROOT        = Path(__file__).resolve().parent.parent
RECORDS     = ROOT / 'data' / 'records.csv'
MODEL_PATH  = ROOT / 'outputs' / 'best_model.pt'
CM_PATH     = ROOT / 'outputs' / 'confusion_matrix.png'
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # -- Load model & val data ----------------------------------------------
    model = ECGQualityCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)

    df = pd.read_csv(RECORDS)
    _, val_df = train_test_split(df, test_size=0.2, random_state=42)
    val_dl = DataLoader(ECGQualityDataset(val_df, train=False),
                        batch_size=64, shuffle=False, num_workers=2)

    # -- Collect predictions -------------------------------------------------
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for x, y in val_dl:
            logits = model(x.to(DEVICE))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            all_labels.extend(y.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    # -- Classification report ----------------------------------------------
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # -- Confusion matrix ----------------------------------------------------
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('ECG Quality Classifier - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    print(f"Saved {CM_PATH}")

    # -- ROC-AUC (one-vs-rest) ----------------------------------------------
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    print(f"Macro ROC-AUC: {auc:.4f}")


if __name__ == '__main__':
    main()
