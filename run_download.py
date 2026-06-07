"""Download the PhysioNet 2021 Challenge ECG data (Georgia 12-lead subset)."""
import wfdb

if __name__ == '__main__':
    # Georgia 12-lead ECG subset (~1.2 GB, ~10k recordings)
    print("Downloading Georgia 12-lead subset to ./data/raw/georgia ...")
    wfdb.dl_database(
        'challenge-2021/1.0.3/training/georgia',
        './data/raw/georgia',
    )
    print("Done.")

    # Optional: CPSC2018 subset (~500 MB) if bandwidth is limited
    # wfdb.dl_database('challenge-2021/1.0.3/training/cpsc_2018', './data/raw/cpsc')
