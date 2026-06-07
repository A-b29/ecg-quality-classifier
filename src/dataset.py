import torch
from torch.utils.data import Dataset
import wfdb
import numpy as np
from scipy.signal import resample

TARGET_LEN = 5000   # 10 seconds at 500 Hz
TARGET_FS  = 500


class ECGQualityDataset(Dataset):
    def __init__(self, records_df, augment: bool = False):
        self.records = records_df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        signal, meta = wfdb.rdsamp(row['path'])
        # signal shape: [n_samples, 12]

        # --- Resample to 500 Hz if needed ---
        if meta['fs'] != TARGET_FS:
            n_target = int(signal.shape[0] * TARGET_FS / meta['fs'])
            signal = resample(signal, n_target, axis=0)

        # --- Crop or zero-pad to TARGET_LEN ---
        n = signal.shape[0]
        if n >= TARGET_LEN:
            signal = signal[:TARGET_LEN, :]
        else:
            pad = np.zeros((TARGET_LEN - n, 12))
            signal = np.vstack([signal, pad])
        # Transpose -> [12, TARGET_LEN]
        sig = signal.T.astype(np.float32)

        # --- Z-score per lead ---
        mu = sig.mean(axis=1, keepdims=True)
        sd = sig.std(axis=1, keepdims=True) + 1e-8
        sig = (sig - mu) / sd

        # --- Augmentation (training only) ---
        if self.augment:
            sig = self._augment(sig)

        return torch.from_numpy(sig), int(row['label'])

    def _augment(self, sig: np.ndarray) -> np.ndarray:
        # Gaussian noise injection
        if np.random.rand() < 0.4:
            sig = sig + np.random.normal(0, 0.05, sig.shape).astype(np.float32)
        # Amplitude scaling
        if np.random.rand() < 0.4:
            sig = sig * np.random.uniform(0.8, 1.2)
        # Random lead dropout (simulate electrode detachment)
        if np.random.rand() < 0.3:
            n_drop = np.random.randint(1, 3)
            leads = np.random.choice(12, n_drop, replace=False)
            sig[leads] = 0.0
        # Baseline wander (low-freq sinusoid)
        if np.random.rand() < 0.3:
            t = np.linspace(0, 10, TARGET_LEN, dtype=np.float32)
            freq = np.random.uniform(0.1, 0.5)
            wander = np.sin(2 * np.pi * freq * t) * np.random.uniform(0.1, 0.3)
            sig = sig + wander[np.newaxis, :]   # broadcast across leads
        return sig.astype(np.float32)
