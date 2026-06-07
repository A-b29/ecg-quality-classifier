import streamlit as st
import torch
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import tempfile, os
from pathlib import Path

from src.dataset import TARGET_LEN, TARGET_FS
from src.model import ECGQualityCNN

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'outputs' / 'best_model.pt'

# -- Page config ------------------------------------------------------------
st.set_page_config(page_title='ECG Quality Classifier', page_icon='🫀', layout='wide')

LEAD_NAMES   = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
CLASS_NAMES  = ['Clean', 'Noisy', 'Artifact']
CLASS_ICONS  = {'Clean': '✅', 'Noisy': '⚠️', 'Artifact': '❌'}


@st.cache_resource
def load_model():
    m = ECGQualityCNN()
    m.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    m.eval()
    return m


def preprocess(signal, fs):
    from scipy.signal import resample
    if fs != TARGET_FS:
        n = int(signal.shape[0] * TARGET_FS / fs)
        signal = resample(signal, n, axis=0)
    n = signal.shape[0]
    if n >= TARGET_LEN:
        signal = signal[:TARGET_LEN, :]
    else:
        signal = np.vstack([signal, np.zeros((TARGET_LEN - n, 12))])
    sig = signal.T.astype(np.float32)
    mu = sig.mean(axis=1, keepdims=True)
    sd = sig.std(axis=1, keepdims=True) + 1e-8
    return (sig - mu) / sd


# -- UI ---------------------------------------------------------------------
st.title('🫀 ECG Signal Quality Classifier')
st.caption('Upload a 12-lead ECG (WFDB .hea + .dat pair) to get an automated quality assessment.')

with st.sidebar:
    st.header('About')
    st.markdown("""
    **Model**: Lightweight 1D CNN (~38k params)
    **Classes**: Clean · Noisy · Artifact
    **Dataset**: PhysioNet 2021 Challenge
    **Input**: 12-lead ECG, 10s @ 500 Hz
    """)
    st.divider()
    st.markdown('Built with PyTorch + Streamlit')

if not MODEL_PATH.exists():
    st.error("No trained model found at outputs/best_model.pt. "
             "Run `python src/train.py` first.")
    st.stop()

uploaded_hea = st.file_uploader('Upload .hea file', type=['hea'])
uploaded_dat = st.file_uploader('Upload .dat file', type=['dat'])

if uploaded_hea and uploaded_dat:
    model = load_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        hea_path = os.path.join(tmpdir, 'record.hea')
        dat_path = os.path.join(tmpdir, 'record.dat')
        with open(hea_path, 'wb') as f:
            f.write(uploaded_hea.read())
        with open(dat_path, 'wb') as f:
            f.write(uploaded_dat.read())

        try:
            sig, meta = wfdb.rdsamp(os.path.join(tmpdir, 'record'))
        except Exception as e:
            st.error(f'Failed to read file: {e}')
            st.stop()

        x = preprocess(sig, meta['fs'])
        x_tensor = torch.from_numpy(x).unsqueeze(0)

        with torch.no_grad():
            logits = model(x_tensor)
            probs  = torch.softmax(logits, dim=1).squeeze().numpy()

        pred_idx  = int(probs.argmax())
        pred_name = CLASS_NAMES[pred_idx]

        # -- Result display --------------------------------------------------
        col1, col2, col3 = st.columns(3)
        col1.metric('Prediction', f"{CLASS_ICONS[pred_name]} {pred_name}")
        col2.metric('Confidence', f"{probs[pred_idx] * 100:.1f}%")
        col3.metric('Sampling rate', f"{meta['fs']} Hz")

        st.divider()

        # Probability bar chart
        prob_df = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}
        st.bar_chart(prob_df)

        # 12-lead plot
        st.subheader('12-Lead Signal')
        n_display = min(2500, sig.shape[0])   # show first 5s
        fig, axes = plt.subplots(12, 1, figsize=(14, 18), sharex=True)
        fig.patch.set_facecolor('#0e1117')
        for i, ax in enumerate(axes):
            ax.plot(sig[:n_display, i], lw=0.7, color='#00d4aa')
            ax.set_ylabel(LEAD_NAMES[i], fontsize=9, rotation=0,
                          labelpad=24, color='white')
            ax.set_facecolor('#0e1117')
            ax.tick_params(colors='#666')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333')
        axes[-1].set_xlabel('Sample', color='white')
        plt.tight_layout()
        st.pyplot(fig)

elif uploaded_hea and not uploaded_dat:
    st.warning('Please also upload the corresponding .dat file.')
else:
    st.info('Upload a WFDB .hea and .dat file pair to classify signal quality.')
