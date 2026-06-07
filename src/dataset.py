"""Synthetic ECG signal-quality dataset.

PhysioNet 2021 has no signal-quality labels (only diagnoses), so we treat every
real recording as a CLEAN source and synthesise the Noisy / Artifact classes via
controlled corruption. This is a standard approach for ECG quality classification
when no quality labels exist, and it yields perfectly balanced classes.

    class 0 = Clean     (real ECG, unmodified)
    class 1 = Noisy     (Gaussian noise + baseline wander)
    class 2 = Artifact  (lead dropout + motion-artifact spikes)

Training corruption is random per sample (effectively unlimited augmentation).
Validation corruption is deterministic per record (reproducible, fair metrics).
"""
import torch
from torch.utils.data import Dataset
import wfdb
import numpy as np
from scipy.signal import resample

TARGET_LEN  = 5000   # 10 seconds at 500 Hz
TARGET_FS   = 500
N_CLASSES   = 3
CLASS_NAMES = ['Clean', 'Noisy', 'Artifact']


def _zscore(sig: np.ndarray) -> np.ndarray:
    mu = sig.mean(axis=1, keepdims=True)
    sd = sig.std(axis=1, keepdims=True) + 1e-8
    return ((sig - mu) / sd).astype(np.float32)


class ECGQualityDataset(Dataset):
    def __init__(self, records_df, train: bool = True, seed: int = 42):
        self.records = records_df.reset_index(drop=True)
        self.train = train
        self.seed = seed

    def __len__(self):
        return len(self.records)

    # -- Signal loading ------------------------------------------------------
    def _load(self, path: str) -> np.ndarray:
        signal, meta = wfdb.rdsamp(path)            # [n_samples, 12]
        if meta['fs'] != TARGET_FS:
            n_target = int(signal.shape[0] * TARGET_FS / meta['fs'])
            signal = resample(signal, n_target, axis=0)
        n = signal.shape[0]
        if n >= TARGET_LEN:
            signal = signal[:TARGET_LEN, :]
        else:
            signal = np.vstack([signal, np.zeros((TARGET_LEN - n, 12))])
        sig = signal.T.astype(np.float32)           # [12, TARGET_LEN]
        # NaNs occur in a few records -> zero them
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        return _zscore(sig)

    # -- Synthetic corruption ------------------------------------------------
    def _corrupt(self, sig: np.ndarray, cls: int, rng: np.random.Generator) -> np.ndarray:
        if cls == 0:                                # Clean
            return sig

        sig = sig.copy()
        if cls == 1:                                # Noisy
            # broadband (EMG-like) Gaussian noise
            sigma = rng.uniform(0.15, 0.35)
            sig = sig + rng.normal(0, sigma, sig.shape).astype(np.float32)
            # low-frequency baseline wander
            t = np.linspace(0, 10, TARGET_LEN, dtype=np.float32)
            freq = rng.uniform(0.15, 0.5)
            phase = rng.uniform(0, 2 * np.pi)
            wander = np.sin(2 * np.pi * freq * t + phase) * rng.uniform(0.3, 0.7)
            sig = sig + wander[np.newaxis, :]
            return sig.astype(np.float32)

        # cls == 2  -> Artifact
        # electrode detachment: a few leads go flat
        n_drop = int(rng.integers(2, 5))
        for lead in rng.choice(12, n_drop, replace=False):
            sig[lead] = 0.0
        # motion-artifact bursts: large transient spikes on some leads
        n_spike_leads = int(rng.integers(1, 4))
        for lead in rng.choice(12, n_spike_leads, replace=False):
            for _ in range(int(rng.integers(3, 8))):
                p = int(rng.integers(0, TARGET_LEN - 60))
                w = int(rng.integers(10, 60))
                amp = rng.uniform(3.0, 8.0) * (1 if rng.random() < 0.5 else -1)
                sig[lead, p:p + w] += amp
        return sig.astype(np.float32)

    # -- Item ----------------------------------------------------------------
    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        sig = self._load(row['path'])

        if self.train:
            rng = np.random.default_rng()           # fresh randomness each epoch
            cls = int(rng.integers(0, N_CLASSES))
        else:
            rng = np.random.default_rng(self.seed + idx)   # reproducible per record
            cls = idx % N_CLASSES                          # balanced, deterministic

        sig = self._corrupt(sig, cls, rng)
        sig = _zscore(sig)                          # re-normalise so amplitude isn't a giveaway
        return torch.from_numpy(sig), cls
