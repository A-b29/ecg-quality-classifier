"""Download a subset of the PhysioNet 2021 Georgia 12-lead ECG records.

The Challenge-2021 archive is organised as nested subdirectories
(training/georgia/g1/, g2/, ...), each with its own RECORDS file, so the stock
`wfdb.dl_database` helper (which expects a single RECORDS file) does not apply.
This script reads the nested RECORDS files and downloads the .hea + .mat pairs
directly into a flat data/raw/georgia/ folder.

We only need real, clean ECGs -- the Noisy / Artifact classes are synthesised at
training time (see src/dataset.py) -- so a subset is plenty. Bump N_SUBDIRS for
more data (each subdir = ~1000 records, ~110 MB).
"""
import shutil
import urllib.request
import concurrent.futures as cf
from pathlib import Path

ROOT    = Path(__file__).resolve().parent
OUT_DIR = ROOT / 'data' / 'raw' / 'georgia'
BASE    = 'https://physionet.org/files/challenge-2021/1.0.3/'

N_SUBDIRS = 2    # g1..g2 = ~2000 records (~220 MB)
WORKERS   = 8


def get_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode('utf-8', 'replace')


def download(url: str, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return 'skip'
    tmp = dest.with_suffix(dest.suffix + '.part')
    with urllib.request.urlopen(url, timeout=180) as r, open(tmp, 'wb') as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)
    return 'ok'


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    root = get_text(BASE + 'RECORDS').split()
    subdirs = [l for l in root if l.startswith('training/georgia/')][:N_SUBDIRS]

    tasks = []
    for sub in subdirs:
        for rec in get_text(BASE + sub + 'RECORDS').split():
            for ext in ('.hea', '.mat'):
                tasks.append((BASE + sub + rec + ext, OUT_DIR / (rec + ext)))

    print(f"{len(subdirs)} subdir(s), {len(tasks)} files -> {OUT_DIR}")
    ok = skip = err = 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(download, u, d): u for u, d in tasks}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            try:
                r = fut.result()
                ok += (r == 'ok'); skip += (r == 'skip')
            except Exception as e:
                err += 1
                if err <= 5:
                    print(f"  ! {futures[fut]}: {e}")
            if i % 500 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  (downloaded={ok} skipped={skip} errors={err})")

    print(f"Done. downloaded={ok} skipped={skip} errors={err}. Files in {OUT_DIR}")


if __name__ == '__main__':
    main()
