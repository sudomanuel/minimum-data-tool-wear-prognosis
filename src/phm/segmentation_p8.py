"""
segmentation_p8.py — P8.2 full-contact segmentation + multi-domain features.

Operates on the existing per-contact recordings (data/raw/segments/{A,R}{exp}_p{n}.txt),
which already contain the COMPLETE contact event (~1-3.5 s at 50 kHz: pre-impact + main cut +
tail). "Full-contact segmentation" = detect the active-contact window [start,end] within each
file (envelope + adaptive threshold + margins + min/max bounds), instead of using only the peak.
Raw signals are NOT moved. No signals invented for 71-72; no imputation of exp77 p5/p6.

Pure-numpy/scipy (pywt unavailable -> manual Daubechies-2 DWT).
"""
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis

FS = 50000.0
EPS = 1e-12

# Daubechies-2 (db2) decomposition filters (pywt unavailable).
_s3 = np.sqrt(3.0)
_den = 4.0 * np.sqrt(2.0)
DB2_LO = np.array([(1 + _s3) / _den, (3 + _s3) / _den, (3 - _s3) / _den, (1 - _s3) / _den])
DB2_HI = DB2_LO[::-1] * np.array([1, -1, 1, -1])


# ----------------------------------------------------------------------------- IO
def load_signal(path):
    df = pd.read_csv(path)
    df = df.iloc[:, :2]
    df.columns = ["t", "v"]
    v = df["v"].to_numpy(float)
    v = v - np.nanmean(v)
    return df["t"].to_numpy(float), v


# ----------------------------------------------------- full-contact event detection
def _smooth_envelope(v, win):
    env = np.abs(v)
    k = np.ones(win) / win
    return np.convolve(env, k, mode="same")


def detect_contact(v, fs=FS, margin_s=0.03, min_s=0.05, max_s=None):
    """Detect the active-contact window via smoothed envelope + adaptive threshold.

    threshold = baseline_noise + 0.20*(peak_env - baseline_noise), baseline = 10th pct env.
    Returns dict(start, end, peak, threshold, active_frac, flag).
    """
    n = len(v)
    win = max(1, int(0.005 * fs))  # 5 ms smoothing
    env = _smooth_envelope(v, win)
    base = np.percentile(env, 10)
    peak = float(env.max())
    thr = base + 0.20 * (peak - base)
    active = env >= thr
    if not active.any():
        return dict(start=0, end=n - 1, peak_sample=int(np.argmax(env)),
                    threshold=float(thr), active_frac=0.0, flag="no_active_region")
    idx = np.flatnonzero(active)
    s, e = int(idx[0]), int(idx[-1])
    m = int(margin_s * fs)
    s = max(0, s - m)
    e = min(n - 1, e + m)
    dur = (e - s) / fs
    min_n = int(min_s * fs)
    flag = "valid"
    if (e - s) < min_n:
        flag = "short_contact"
    if max_s is not None and dur > max_s:
        e = min(n - 1, s + int(max_s * fs))
        flag = "clipped_max"
    # low peak-to-baseline contrast = sustained-plateau contact (NORMAL for these cuts),
    # an informational note only — NOT an exclusion criterion.
    if peak / (base + EPS) < 3.0 and flag == "valid":
        flag = "low_contrast"
    return dict(start=s, end=e, peak_sample=int(np.argmax(env)),
                threshold=float(thr), active_frac=float(active.mean()), flag=flag)


# ----------------------------------------------------------------- feature domains
def time_features(x):
    x = np.asarray(x, float)
    n = len(x)
    mean = float(x.mean())
    std = float(x.std())
    var = float(x.var())
    rms = float(np.sqrt(np.mean(x ** 2)))
    mav = float(np.mean(np.abs(x)))
    mx, mn = float(x.max()), float(x.min())
    p2p = mx - mn
    energy = float(np.sum(x ** 2))
    wl = float(np.sum(np.abs(np.diff(x))))               # waveform length
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))  # zero crossing rate
    sk = float(skew(x)) if std > 0 else 0.0
    ku = float(kurtosis(x)) if std > 0 else 0.0
    peak = float(np.max(np.abs(x)))
    crest = peak / (rms + EPS)
    impulse = peak / (mav + EPS)
    sqrt_mean_abs = (np.mean(np.sqrt(np.abs(x)))) ** 2
    clearance = peak / (sqrt_mean_abs + EPS)
    shape = rms / (mav + EPS)
    margin = peak / (sqrt_mean_abs + EPS)                # margin factor
    # spectral/Shannon-ish entropy on normalized |x|
    p = np.abs(x) / (np.sum(np.abs(x)) + EPS)
    entropy = float(-np.sum(p * np.log(p + EPS)))
    return dict(mean=mean, std=std, variance=var, rms=rms, mav=mav, max=mx, min=mn,
                peak_to_peak=p2p, energy=energy, waveform_length=wl, zero_crossing_rate=zcr,
                skewness=sk, kurtosis=ku, crest_factor=crest, impulse_factor=impulse,
                margin_factor=margin, shape_factor=shape, clearance_factor=clearance,
                entropy=entropy)


def freq_features(x, fs=FS, n_bands=5):
    x = np.asarray(x, float)
    n = len(x)
    X = np.abs(rfft(x))
    f = rfftfreq(n, d=1.0 / fs)
    P = X ** 2
    Psum = P.sum() + EPS
    dominant = float(f[np.argmax(X)])
    mean_freq = float(np.sum(f * P) / Psum)
    cum = np.cumsum(P)
    median_freq = float(f[np.searchsorted(cum, cum[-1] / 2)])
    centroid = float(np.sum(f * X) / (X.sum() + EPS))
    rms_freq = float(np.sqrt(np.sum((f ** 2) * P) / Psum))
    freq_var = float(np.sum(((f - mean_freq) ** 2) * P) / Psum)
    total_power = float(P.sum())
    mean_power = float(P.mean())
    spectral_power = float(np.sqrt(np.mean(P)))
    Pn = P / Psum
    spec_skew = float(np.sum(((f - mean_freq) ** 3) * Pn) / (freq_var ** 1.5 + EPS))
    spec_kurt = float(np.sum(((f - mean_freq) ** 4) * Pn) / (freq_var ** 2 + EPS))
    out = dict(dominant_freq=dominant, mean_freq=mean_freq, median_freq=median_freq,
               spectral_centroid=centroid, rms_freq=rms_freq, freq_variance=freq_var,
               spectral_power=spectral_power, total_power=total_power, mean_power=mean_power,
               spectral_skewness=spec_skew, spectral_kurtosis=spec_kurt)
    edges = np.linspace(0, fs / 2, n_bands + 1)
    for b in range(n_bands):
        m = (f >= edges[b]) & (f < edges[b + 1])
        be = float(P[m].sum())
        out[f"band{b+1}_energy"] = be
        out[f"band{b+1}_ratio"] = be / Psum
    return out


def _dwt1(x, lo, hi):
    a = np.convolve(x, lo[::-1], mode="full")[1::2]
    d = np.convolve(x, hi[::-1], mode="full")[1::2]
    return a, d


def wavelet_features(x, levels=3):
    x = np.asarray(x, float)
    a = x
    details = []
    for _ in range(levels):
        if len(a) < 4:
            break
        a, d = _dwt1(a, DB2_LO, DB2_HI)
        details.append(d)
    energies = [float(np.sum(d ** 2)) for d in details]
    approx_e = float(np.sum(a ** 2))
    total = sum(energies) + approx_e + EPS
    out = {}
    for i, e in enumerate(energies, 1):
        out[f"wavelet_d{i}_energy"] = e
        out[f"wavelet_d{i}_ratio"] = e / total
    out["wavelet_approx_energy"] = approx_e
    # low (approx) vs high (level-1 detail) ratio
    out["wavelet_low_high_ratio"] = approx_e / (energies[0] + EPS) if energies else np.nan
    fracs = np.array(energies + [approx_e]) / total
    out["wavelet_entropy"] = float(-np.sum(fracs * np.log(fracs + EPS)))
    return out


def segment_features(v, start, end, fs=FS):
    seg = v[start:end + 1]
    feats = {}
    feats.update(time_features(seg))
    feats.update(freq_features(seg, fs))
    feats.update(wavelet_features(seg))
    feats["segment_n_samples"] = int(len(seg))
    feats["segment_duration_s"] = float(len(seg) / fs)
    return feats
