"""Parse WFDB headers into a label manifest (data/records.csv).

Remaps PhysioNet 2021 SNOMED-CT diagnostic codes into 3 signal-quality
categories: 0 = clean, 1 = noisy, 2 = artifact.
"""
import wfdb
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / 'data' / 'raw' / 'georgia'
OUT_CSV = ROOT / 'data' / 'records.csv'

# SNOMED-CT codes for signal quality issues
NOISY_CODES    = {'370247009', '426783006'}      # noise, baseline wander
ARTIFACT_CODES = {'251148006', '67741000119109'}  # electrode artifact, motion artifact


def get_quality_label(comments: dict) -> int:
    codes = set(comments.get('Dx', '').split(','))
    if codes & ARTIFACT_CODES:
        return 2   # artifact
    elif codes & NOISY_CODES:
        return 1   # noisy
    else:
        return 0   # clean


def main():
    if not RAW_DIR.exists():
        raise SystemExit(
            f"Raw data not found at {RAW_DIR}.\n"
            f"Run `python run_download.py` first."
        )

    records = []
    for hea_path in RAW_DIR.glob('*.hea'):
        stem = hea_path.with_suffix('')
        rec = wfdb.rdheader(str(stem))
        comments = dict(c.split(': ', 1) for c in rec.comments if ': ' in c)
        records.append({
            'path':    str(stem.resolve()),   # absolute → cwd-independent loading
            'label':   get_quality_label(comments),
            'fs':      rec.fs,
            'n_sig':   rec.n_sig,
            'sig_len': rec.sig_len,
        })

    df = pd.DataFrame(records)
    print(df['label'].value_counts().sort_index())
    # Expected roughly: 0 (clean) >> 1 (noisy) > 2 (artifact)
    # If clean dominates, the weighted sampler in training handles imbalance.

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} records to {OUT_CSV}")


if __name__ == '__main__':
    main()
