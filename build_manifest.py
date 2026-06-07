"""Catalog the downloaded Georgia records into data/records.csv.

Every record is a CLEAN source signal -- the Noisy / Artifact classes are
synthesised at training time (see src/dataset.py), so this manifest carries no
quality label, just paths and signal metadata.
"""
import wfdb
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).resolve().parent
RAW_DIR = ROOT / 'data' / 'raw' / 'georgia'
OUT_CSV = ROOT / 'data' / 'records.csv'


def main():
    if not RAW_DIR.exists():
        raise SystemExit(f"Raw data not found at {RAW_DIR}. Run `python run_download.py` first.")

    records = []
    for hea_path in sorted(RAW_DIR.glob('*.hea')):
        stem = hea_path.with_suffix('')
        try:
            rec = wfdb.rdheader(str(stem))
        except Exception as e:
            print(f"  ! skipping {stem.name}: {e}")
            continue
        records.append({
            'path':    str(stem.resolve()),   # absolute -> cwd-independent loading
            'fs':      rec.fs,
            'n_sig':   rec.n_sig,
            'sig_len': rec.sig_len,
        })

    if not records:
        raise SystemExit(f"No .hea files found in {RAW_DIR}. Did the download finish?")

    df = pd.DataFrame(records)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} clean source records to {OUT_CSV}")
    print(f"Sampling frequencies: {df['fs'].value_counts().to_dict()}")
    print(f"Signal lengths (samples): min={df['sig_len'].min()} "
          f"median={int(df['sig_len'].median())} max={df['sig_len'].max()}")


if __name__ == '__main__':
    main()
