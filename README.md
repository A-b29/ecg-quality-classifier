# ECG Signal Quality Classifier

Automated quality assessment of 12-lead ECG recordings using a lightweight 1D CNN.
Classifies signals as **Clean**, **Noisy**, or **Artifact-contaminated**.

## Clinical motivation

Noise in ECG input directly degrades downstream classifiers for arrhythmias, MI,
and ST-changes. This tool acts as a **quality gate** before any diagnostic AI
pipeline — it is a pre-processing screen, not a diagnostic model.

## Architecture

3 × Conv1d blocks (channels: 12 → 32 → 64 → 128) + Global Average Pool + Linear(128 → 3).
~38k parameters. Trains in ~15 min on CPU.

```
Input [B, 12, 5000]
  → ConvBlock(12→32,  k=7, pool=2)
  → ConvBlock(32→64,  k=5, pool=2)
  → ConvBlock(64→128, k=3, pool=4)
  → AdaptiveAvgPool1d(1) → Flatten → Dropout(0.3) → Linear(128→3)
Output [B, 3]  (Clean / Noisy / Artifact)
```

## Dataset & labeling

Real ECGs come from the PhysioNet / Computing in Cardiology Challenge 2021 —
Georgia 12-lead subset. A 2,000-record sample is used by default (configurable in
`run_download.py`).

**Important:** PhysioNet 2021 contains *diagnostic* labels (SNOMED-CT codes for
arrhythmias, etc.), **not** signal-quality labels. Rather than mis-repurpose
diagnosis codes as quality labels (which would be meaningless), this project takes
the standard approach for quality classification without quality labels:

- every real recording is treated as a **Clean** source signal;
- **Noisy** (Gaussian noise + baseline wander) and **Artifact** (lead dropout +
  motion-artifact spikes) examples are **synthesised on the fly** from those clean
  signals.

This produces perfectly balanced, fully controlled classes. Training corruption is
random per sample (effectively unlimited augmentation); validation corruption is
deterministic per record, so metrics are reproducible. See `src/dataset.py`.

## Results

> Fill in after running `python src/evaluate.py`.

| Class    | Precision | Recall | F1     |
|----------|-----------|--------|--------|
| Clean    | 0.xx      | 0.xx   | 0.xx   |
| Noisy    | 0.xx      | 0.xx   | 0.xx   |
| Artifact | 0.xx      | 0.xx   | 0.xx   |
| **Macro**| **0.xx**  |**0.xx**|**0.xx**|

Expected ranges: overall val acc 82–87% · macro F1 0.78–0.84 · macro ROC-AUC 0.90–0.94.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data (~1.2 GB, Georgia subset)
python run_download.py

# 3. Catalog the records into data/records.csv
python build_manifest.py

# 4. Train (writes outputs/best_model.pt)
python src/train.py

# 5. Evaluate (writes outputs/confusion_matrix.png)
python src/evaluate.py

# 6. Launch the interactive demo
streamlit run app.py
```

## Project structure

```
.
├── data/
│   ├── raw/            # WFDB files from PhysioNet (git-ignored)
│   └── records.csv     # record manifest (generated)
├── src/
│   ├── dataset.py      # ECGQualityDataset
│   ├── model.py        # ECGQualityCNN
│   ├── train.py        # training loop
│   └── evaluate.py     # metrics + confusion matrix
├── notebooks/
│   └── eda.ipynb       # exploratory data analysis
├── outputs/            # model weights + figures (git-ignored)
├── run_download.py     # download PhysioNet data
├── build_manifest.py   # parse labels → records.csv
├── app.py              # Streamlit demo
└── requirements.txt
```

## Design decisions

1. **1D CNN over LSTM/Transformer** — CNNs capture local temporal patterns (QRS
   morphology, noise bursts) efficiently; LSTMs/Transformers are slower and need
   more data for a quality (not diagnostic) task.
2. **Global Average Pooling** — length-agnostic, fewer parameters than a flattened
   FC layer, and acts as regularization.
3. **Synthetic labels instead of repurposed diagnosis codes** — PhysioNet 2021 has
   no quality labels. Faking them from SNOMED-CT diagnosis codes (e.g. labeling
   "sinus rhythm" as "noisy") yields a model that secretly detects diagnoses, not
   quality. Controlled synthetic corruption gives honest, balanced, well-defined
   classes — and re-normalising after corruption forces the model to learn
   morphology rather than just overall amplitude.
4. **Per-lead z-score normalization** — removes per-patient/lead/electrode
   amplitude confounds before the CNN sees the signal.

## References

- Reyna et al., *"Will Two Do? Varying Dimensions in Electrocardiography"* (CinC 2021)
- PhysioNet Challenge 2021: https://physionetchallenges.org/2021/
