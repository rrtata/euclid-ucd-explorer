#!/usr/bin/env python3
"""
Euclid Q1 UCD Explorer — Interactive Streamlit App
====================================================
Interactive tool for exploring the Euclid Q1 UCD candidate catalog.

Features:
  • Interactive color-color plot (click to select a source)
  • Side panel: object metadata, Euclid image cutout, NISP spectrum
  • Live spectral template fitting with best-fit SpT display
  • 2D grism image with source trace highlighted
  • Cross-match status with published catalogs

Usage:
  streamlit run euclid_ucd_explorer.py

Requirements:
  pip install streamlit plotly pandas numpy scipy astropy boto3 matplotlib
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import pickle, os, json, io, warnings
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Euclid Q1 UCD Explorer",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# CONFIGURATION — edit these paths for your setup
# =====================================================================
DEFAULT_CATALOG    = "full_q1_results/catalogs/classified_catalog.csv"
DEFAULT_STANDARDS  = "standards_cache/splat_standards.pkl"
DEFAULT_RF_MODEL   = "ml_model_v2/rf_classifier.pkl"
DEFAULT_SPT_MODEL  = "ml_model_v2/spt_regressor.pkl"
DEFAULT_SPECTRA    = "full_q1_results/spectra_fits"
DEFAULT_MODEL_CFG  = "ml_model_v2/model_config.json"

# S3 bucket for Euclid Q1
S3_BUCKET   = "nasa-irsa-euclid-q1"
S3_REGION   = "us-east-1"

APP_VERSION = "1.2.1"  # bump this on each push to verify deployment

# IRSA base URL
IRSA_BASE   = "https://irsa.ipac.caltech.edu"
IRSA_CUTOUT = f"{IRSA_BASE}/cgi-bin/2MASS/IM/nph-im_sia"

# Euclid Deep Field centers (for tile lookup)
EDF_CENTERS = {
    'EDF-N':  (269.4,  65.8),
    'EDF-F':  ( 53.1, -28.1),
    'EDF-S':  ( 61.2, -48.5),
}

# Template fitting config
FIT_WMIN = 12800   # Angstroms
FIT_WMAX = 18200
SYSTEMATIC_FRAC = 0.05
N_ALPHA_BINARY = 11

# Spectral type numeric mapping
SPT_MAP = {}
for i, sp in enumerate(['M6','M7','M8','M9','L0','L1','L2','L3','L4','L5',
                         'L6','L7','L8','L9','T0','T1','T2','T3','T4','T5',
                         'T6','T7','T8','T9']):
    SPT_MAP[sp] = i
SPT_MAP_INV = {v: k for k, v in SPT_MAP.items()}


# =====================================================================
# VETTING PERSISTENCE
# =====================================================================
DEFAULT_VETTING_FILE = "vetting_decisions.json"

class VettingManager:
    """Persist accept/reject decisions + notes to a JSON file.

    File format (vetting_decisions.json):
    {
      "<object_id>": {
        "decision": "accepted" | "rejected",
        "notes": "free-text",
        "best_fit_spt": "L3",
        "reviewer": "RT",
        "timestamp": "2026-02-10T12:34:56"
      },
      ...
    }
    """

    def __init__(self, path: str = DEFAULT_VETTING_FILE):
        self.path = path
        self._data: dict = {}
        self._load()

    # --- I/O --------------------------------------------------------
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = {}

    def _save(self):
        with open(self.path, 'w') as f:
            json.dump(self._data, f, indent=2, default=str)

    # --- Public API -------------------------------------------------
    def set_decision(self, object_id, decision, notes="",
                     best_fit_spt="", reviewer=""):
        """Record an accept/reject decision."""
        from datetime import datetime, timezone
        key = str(object_id)
        self._data[key] = {
            "decision": decision,
            "notes": notes,
            "best_fit_spt": best_fit_spt,
            "reviewer": reviewer,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def clear_decision(self, object_id):
        """Remove a vetting decision (reset to unvetted)."""
        key = str(object_id)
        if key in self._data:
            del self._data[key]
            self._save()

    def get_decision(self, object_id) -> dict | None:
        """Return the decision dict, or None if unvetted."""
        return self._data.get(str(object_id))

    def get_status(self, object_id) -> str:
        """Return 'accepted', 'rejected', or 'unvetted'."""
        d = self.get_decision(object_id)
        return d["decision"] if d else "unvetted"

    @property
    def all_decisions(self) -> dict:
        return dict(self._data)

    @property
    def n_accepted(self) -> int:
        return sum(1 for v in self._data.values() if v["decision"] == "accepted")

    @property
    def n_rejected(self) -> int:
        return sum(1 for v in self._data.values() if v["decision"] == "rejected")

    @property
    def n_total(self) -> int:
        return len(self._data)

    def accepted_ids(self) -> list:
        return [k for k, v in self._data.items() if v["decision"] == "accepted"]

    def rejected_ids(self) -> list:
        return [k for k, v in self._data.items() if v["decision"] == "rejected"]

    def to_dataframe(self) -> pd.DataFrame:
        """Export all decisions as a DataFrame."""
        if not self._data:
            return pd.DataFrame(columns=["object_id", "decision", "notes",
                                          "best_fit_spt", "reviewer", "timestamp"])
        rows = []
        for oid, info in self._data.items():
            rows.append({"object_id": oid, **info})
        return pd.DataFrame(rows)


# =====================================================================
# DATA LOADING
# =====================================================================
@st.cache_data
def load_catalog(path):
    """Load the classified UCD catalog."""
    for p in [path, DEFAULT_CATALOG, 'step5_classified.csv',
              'ml_results/ml_classified.csv']:
        if p and os.path.exists(p):
            df = pd.read_csv(p)
            st.sidebar.success(f"Loaded {len(df)} objects from {p}")
            return df
    return None


@st.cache_resource
def load_standards(path):
    """Load SpeX Prism Library templates."""
    for p in [path, DEFAULT_STANDARDS, 'splat_standards.pkl']:
        if p and os.path.exists(p):
            with open(p, 'rb') as f:
                stds = pickle.load(f)
            return stds
    return None


@st.cache_data
def load_extracted_spectra(spectra_dir):
    """Load pre-extracted spectra from FITS files."""
    from astropy.io import fits

    spectra = {}
    if not os.path.isdir(spectra_dir):
        return spectra

    for fname in os.listdir(spectra_dir):
        if not fname.endswith('.fits'):
            continue
        oid = fname.replace('spec_', '').replace('.fits', '')
        try:
            with fits.open(os.path.join(spectra_dir, fname)) as hdul:
                data = hdul[1].data
                spectra[oid] = {
                    'wave': np.array(data['WAVE'], dtype=float),
                    'flux': np.array(data['FLUX'], dtype=float),
                    'noise': np.array(data['NOISE'], dtype=float),
                }
        except Exception:
            continue

    return spectra


# =====================================================================
# S3 ACCESS
# =====================================================================
def get_s3_client():
    """Get anonymous S3 client for Euclid Q1 data."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        return boto3.client('s3', config=Config(signature_version=UNSIGNED),
                            region_name=S3_REGION)
    except ImportError:
        st.warning("boto3 not installed — S3 access disabled. `pip install boto3`")
        return None


def fetch_spectrum_from_s3(object_id, tileid, combspec_file, hdu_idx):
    """Download and extract a spectrum from a COMBSPEC file on S3."""
    from astropy.io import fits

    s3 = get_s3_client()
    if s3 is None:
        return None

    s3_key = f"q1/SIR/{tileid}/{combspec_file}"

    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        data = response['Body'].read()
        hdul = fits.open(io.BytesIO(data))

        if hdu_idx >= len(hdul):
            return None

        ext = hdul[hdu_idx]
        tbl = ext.data
        hdr = ext.header

        if tbl is None:
            return None

        colnames = [c.upper() for c in tbl.dtype.names]

        # Wavelength
        wave = None
        for wn in ['WAVELENGTH', 'WAVE', 'LAMBDA']:
            if wn in colnames:
                wave = np.array(tbl[wn], dtype=float); break
        if wave is None:
            return None

        # Signal
        flux_raw = None
        for fn in ['SIGNAL', 'FLUX', 'DATA']:
            if fn in colnames:
                flux_raw = np.array(tbl[fn], dtype=float); break
        if flux_raw is None:
            return None

        # FSCALE
        fscale = hdr.get('FSCALE', None)
        if fscale is None or fscale == 0:
            try:
                fscale = hdul[0].header.get('FSCALE', 1.0)
            except Exception:
                fscale = 1.0
        if fscale == 0:
            fscale = 1.0
        flux = flux_raw * fscale

        # Noise
        noise = None
        if 'VAR' in colnames:
            var = np.array(tbl['VAR'], dtype=float)
            noise = np.sqrt(np.abs(var)) * abs(fscale)
        else:
            noise = np.abs(flux) * 0.1

        # Mask
        bad_mask = np.zeros(len(wave), dtype=bool)
        if 'MASK' in colnames:
            mask_vals = np.array(tbl['MASK'], dtype=int)
            bad_mask = (mask_vals % 2 == 1) | (mask_vals >= 64)

        good = (~bad_mask & np.isfinite(wave) & np.isfinite(flux) &
                np.isfinite(noise) & (noise > 0) & (wave > 0))
        if good.sum() < 20:
            return None

        wave, flux, noise = wave[good], flux[good], noise[good]

        # Wavelength units
        med_w = np.nanmedian(wave)
        if med_w > 5000:
            wave_um = wave / 10000.0
        elif med_w > 500:
            wave_um = wave / 1000.0
        else:
            wave_um = wave

        # Convert to Angstroms for fitting
        wave_ang = wave_um * 10000.0

        hdul.close()
        return {'wave': wave_ang, 'flux': flux, 'noise': noise,
                'wave_um': wave_um}

    except Exception as e:
        st.error(f"S3 download failed: {e}")
        return None


def fetch_euclid_image_cutout(ra, dec, size_arcsec=30):
    """Fetch Euclid image cutout via IRSA cutout service or S3."""
    import urllib.request

    # Try IRSA SIA cutout
    url = (f"{IRSA_BASE}/cgi-bin/2MASS/IM/nph-im_sia?"
           f"POS={ra},{dec}&SIZE={size_arcsec/3600.0}")
    try:
        response = urllib.request.urlopen(url, timeout=10)
        return response.read()
    except Exception:
        pass

    return None


def fetch_2d_grism_from_s3(tileid, combspec_file):
    """
    Fetch the 2D grism spectrogram from S3.
    The COMBSPEC FITS file contains 2D spectrograms as image extensions.
    """
    from astropy.io import fits

    s3 = get_s3_client()
    if s3 is None:
        return None

    # The 2D grism data is in the SIR science frames (SPEC2D)
    # Pattern: EUC_SIR_W-SPEC2D_{tileid}_{obs}.fits
    # Try the COMBSPEC file first — some HDUs have 2D data
    s3_key = f"q1/SIR/{tileid}/{combspec_file}"

    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        data = response['Body'].read()
        hdul = fits.open(io.BytesIO(data))
        return hdul
    except Exception as e:
        st.warning(f"2D grism fetch failed: {e}")
        return None


def auto_fetch_spectra(object_id):
    """
    Auto-fetch spectral data for an object: queries IRSA, downloads COMBSPEC,
    extracts 1D spectrum + returns FITS HDU list for 2D display.

    Returns dict cached in session state:
    {
        'spectrum':  {'wave':..., 'flux':..., 'noise':..., 'wave_um':...} or None,
        'hdul_bytes': bytes of the COMBSPEC FITS file (for 2D),
        'hdu_idx':   int HDU index for this object,
        'tileid':    str,
        'combspec':  str filename,
        'status':    'ok' | 'no_association' | 'download_failed' | 'no_spectrum',
        'error':     str or None,
    }
    """
    import re
    from astropy.io import fits

    cache_key = f"spectra_cache_{object_id}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    result = {'spectrum': None, 'hdul_bytes': None, 'hdu_idx': 1,
              'tileid': '', 'combspec': '', 'status': 'pending', 'error': None}

    # Step 1: Query IRSA for spectral association
    assoc = query_spectral_association(object_id)
    if assoc is None:
        result['status'] = 'no_association'
        st.session_state[cache_key] = result
        return result

    uri = assoc.get('uri', '')
    result['tileid'] = str(assoc.get('tileid', ''))
    result['hdu_idx'] = int(assoc.get('hdu', 1))

    match = re.search(r'(EUC_SIR_W-COMBSPEC_\d+_[^?]+\.fits)', uri)
    if not match or not result['tileid']:
        result['status'] = 'no_association'
        result['error'] = f"Could not parse COMBSPEC from URI: {uri[:100]}"
        st.session_state[cache_key] = result
        return result

    result['combspec'] = match.group(1)

    # Step 2: Download COMBSPEC from S3
    s3 = get_s3_client()
    if s3 is None:
        result['status'] = 'download_failed'
        result['error'] = 'boto3 not available'
        st.session_state[cache_key] = result
        return result

    s3_key = f"q1/SIR/{result['tileid']}/{result['combspec']}"
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        raw_bytes = response['Body'].read()
        result['hdul_bytes'] = raw_bytes
    except Exception as e:
        result['status'] = 'download_failed'
        result['error'] = str(e)
        st.session_state[cache_key] = result
        return result

    # Step 3: Extract 1D spectrum from the correct HDU
    try:
        hdul = fits.open(io.BytesIO(raw_bytes))
        hdu_idx = result['hdu_idx']

        if hdu_idx < len(hdul):
            ext = hdul[hdu_idx]
            tbl = ext.data
            hdr = ext.header

            if tbl is not None and hasattr(tbl, 'dtype') and tbl.dtype.names:
                colnames = [c.upper() for c in tbl.dtype.names]

                # Wavelength
                wave = None
                for wn in ['WAVELENGTH', 'WAVE', 'LAMBDA']:
                    if wn in colnames:
                        wave = np.array(tbl[wn], dtype=float)
                        break

                # Signal
                flux_raw = None
                for fn in ['SIGNAL', 'FLUX', 'DATA']:
                    if fn in colnames:
                        flux_raw = np.array(tbl[fn], dtype=float)
                        break

                if wave is not None and flux_raw is not None:
                    # FSCALE
                    fscale = hdr.get('FSCALE', None)
                    if fscale is None or fscale == 0:
                        try:
                            fscale = hdul[0].header.get('FSCALE', 1.0)
                        except Exception:
                            fscale = 1.0
                    if fscale == 0:
                        fscale = 1.0
                    flux = flux_raw * fscale

                    # Noise
                    if 'VAR' in colnames:
                        var = np.array(tbl['VAR'], dtype=float)
                        noise = np.sqrt(np.abs(var)) * abs(fscale)
                    else:
                        noise = np.abs(flux) * 0.1

                    # Mask
                    bad_mask = np.zeros(len(wave), dtype=bool)
                    if 'MASK' in colnames:
                        mask_vals = np.array(tbl['MASK'], dtype=int)
                        bad_mask = (mask_vals % 2 == 1) | (mask_vals >= 64)

                    good = (~bad_mask & np.isfinite(wave) & np.isfinite(flux) &
                            np.isfinite(noise) & (noise > 0) & (wave > 0))

                    if good.sum() >= 20:
                        w, f, n = wave[good], flux[good], noise[good]
                        # Wavelength unit conversion
                        med_w = np.nanmedian(w)
                        if med_w > 5000:
                            wave_um = w / 10000.0
                        elif med_w > 500:
                            wave_um = w / 1000.0
                        else:
                            wave_um = w
                        wave_ang = wave_um * 10000.0

                        result['spectrum'] = {
                            'wave': wave_ang, 'flux': f, 'noise': n,
                            'wave_um': wave_um
                        }

        hdul.close()
    except Exception as e:
        result['error'] = f"Spectrum extraction: {e}"

    result['status'] = 'ok' if result['spectrum'] else 'no_spectrum'
    st.session_state[cache_key] = result
    return result


# =====================================================================
# SPECTRAL FITTING ENGINE
# =====================================================================
def smooth_spectrum(wave, flux, noise, R_target=120):
    """Smooth observed spectrum to match SpeX Prism resolution."""
    dw = np.nanmedian(np.diff(wave))
    w_cen = np.nanmedian(wave)
    dw_nisp = w_cen / 450.0
    dw_target = w_cen / R_target
    sigma_ang = np.sqrt(max(dw_target**2 - dw_nisp**2, 0.01)) / 2.355
    sigma_pix = max(sigma_ang / dw, 0.5)

    flux_smooth = gaussian_filter1d(flux, sigma_pix)
    noise_smooth = noise / np.sqrt(max(2 * sigma_pix, 1.0))
    return flux_smooth, noise_smooth


def fit_single_template(wave_obs, flux_obs, noise_obs, wave_tmpl, flux_tmpl):
    """Fit a single template. Returns chi2_red, scale, wave_common, model."""
    mask = ((wave_obs >= FIT_WMIN) & (wave_obs <= FIT_WMAX) &
            np.isfinite(flux_obs) & np.isfinite(noise_obs) & (noise_obs > 0))
    if mask.sum() < 20:
        return np.inf, 1.0, None, None

    w, f, n = wave_obs[mask], flux_obs[mask], noise_obs[mask]

    overlap = (w >= wave_tmpl.min()) & (w <= wave_tmpl.max())
    if overlap.sum() < 20:
        return np.inf, 1.0, None, None
    w, f, n = w[overlap], f[overlap], n[overlap]

    interp_func = interp1d(wave_tmpl, flux_tmpl, kind='cubic',
                           bounds_error=False, fill_value=np.nan)
    t = interp_func(w)
    good = np.isfinite(t) & np.isfinite(f) & (n > 0)
    if good.sum() < 20:
        return np.inf, 1.0, None, None

    w, f, n, t = w[good], f[good], n[good], t[good]

    # Optimal scaling
    w2 = 1.0 / n**2
    denom = np.sum(t**2 * w2)
    if denom <= 0:
        return np.inf, 1.0, None, None
    scale = np.sum(f * t * w2) / denom
    if scale <= 0:
        t_med = np.nanmedian(np.abs(t))
        f_med = np.nanmedian(np.abs(f))
        scale = f_med / t_med if t_med > 0 else 1.0

    # Sigma-clipping
    resid = f - scale * t
    rms = np.nanstd(resid)
    keep = np.abs(resid) < 3.0 * rms
    if keep.sum() >= 20:
        w, f, n, t = w[keep], f[keep], n[keep], t[keep]
        w2 = 1.0 / n**2
        denom = np.sum(t**2 * w2)
        if denom > 0:
            scale = np.sum(f * t * w2) / denom
        resid = f - scale * t

    # Effective noise
    n_frac = SYSTEMATIC_FRAC * np.abs(f)
    mad_f = np.nanmedian(np.abs(f - np.nanmedian(f))) * 1.4826
    n_abs = mad_f * 0.5
    n_eff = np.sqrt(n**2 + n_frac**2 + n_abs**2)

    chi2 = np.sum((resid / n_eff)**2)
    dof = max(len(f) - 1, 1)
    chi2_red = chi2 / dof

    return chi2_red, scale, w, scale * t


def fit_binary_composite(wave_obs, flux_obs, noise_obs,
                         wave_a, flux_a, wave_b, flux_b):
    """Fit binary composite. Returns chi2_red, scale, alpha, wave, model."""
    mask = ((wave_obs >= FIT_WMIN) & (wave_obs <= FIT_WMAX) &
            np.isfinite(flux_obs) & np.isfinite(noise_obs) & (noise_obs > 0))
    if mask.sum() < 20:
        return np.inf, 1.0, 0.5, None, None

    w, f, n = wave_obs[mask], flux_obs[mask], noise_obs[mask]

    w_min_t = max(wave_a.min(), wave_b.min())
    w_max_t = min(wave_a.max(), wave_b.max())
    overlap = (w >= w_min_t) & (w <= w_max_t)
    if overlap.sum() < 20:
        return np.inf, 1.0, 0.5, None, None
    w, f, n = w[overlap], f[overlap], n[overlap]

    ta = interp1d(wave_a, flux_a, kind='cubic', bounds_error=False, fill_value=np.nan)(w)
    tb = interp1d(wave_b, flux_b, kind='cubic', bounds_error=False, fill_value=np.nan)(w)
    good = np.isfinite(ta) & np.isfinite(tb) & np.isfinite(f) & (n > 0)
    if good.sum() < 20:
        return np.inf, 1.0, 0.5, None, None
    w, f, n, ta, tb = w[good], f[good], n[good], ta[good], tb[good]

    best = (np.inf, 1.0, 0.5, None, None)
    for alpha in np.linspace(0.1, 0.9, N_ALPHA_BINARY):
        composite = alpha * ta + (1 - alpha) * tb
        w2 = 1.0 / n**2
        denom = np.sum(composite**2 * w2)
        if denom <= 0:
            continue
        scale = np.sum(f * composite * w2) / denom
        if scale <= 0:
            continue

        resid = f - scale * composite
        n_frac = SYSTEMATIC_FRAC * np.abs(f)
        mad_f = np.nanmedian(np.abs(f - np.nanmedian(f))) * 1.4826
        n_abs = mad_f * 0.5
        n_eff = np.sqrt(n**2 + n_frac**2 + n_abs**2)
        chi2 = np.sum((resid / n_eff)**2)
        dof = max(len(f) - 2, 1)
        chi2_r = chi2 / dof

        if chi2_r < best[0]:
            best = (chi2_r, scale, alpha, w.copy(), (scale * composite).copy())

    return best


def run_full_fit(spectrum, standards, try_binaries=True,
                 progress_callback=None):
    """Run full template fitting against all standards."""
    wave = spectrum['wave']
    flux = spectrum['flux']
    noise = spectrum['noise']

    # Quality check
    finite = np.isfinite(flux) & np.isfinite(noise)
    if finite.sum() < 30:
        return None

    f_fin = flux[finite]
    frac_positive = np.mean(f_fin > 0)
    med_f = np.nanmedian(f_fin)
    mad = np.nanmedian(np.abs(f_fin - med_f))
    if frac_positive < 0.6 or (mad > 0 and med_f / mad < 0.5):
        return None

    # Smooth
    flux_sm, noise_sm = smooth_spectrum(wave, flux, noise)

    results = []
    n_stds = len(standards)

    # Single template fits
    for i, (spt_name, std) in enumerate(standards.items()):
        if progress_callback:
            progress_callback((i + 1) / (n_stds + 1))
        try:
            chi2_r, scale, w_com, model = fit_single_template(
                wave, flux_sm, noise_sm, std['wave'], std['flux'])
            if np.isfinite(chi2_r) and chi2_r < 1e6:
                results.append({
                    'spt': spt_name,
                    'chi2_red': chi2_r,
                    'scale': scale,
                    'wave_common': w_com,
                    'model': model,
                    'is_binary': False,
                    'std_class': std.get('class', 'dwarf'),
                })
        except Exception:
            continue

    if not results:
        return None

    results.sort(key=lambda x: x['chi2_red'])
    best_chi2 = results[0]['chi2_red']

    # Binary composites (top 8 pairs)
    if try_binaries and len(results) >= 2:
        top = [r['spt'] for r in results[:8]]
        for i, spt_a in enumerate(top):
            for spt_b in top[i+1:]:
                try:
                    std_a = standards[spt_a]
                    std_b = standards[spt_b]
                    chi2, sc, alpha, w_bin, model_bin = fit_binary_composite(
                        wave, flux_sm, noise_sm,
                        std_a['wave'], std_a['flux'],
                        std_b['wave'], std_b['flux'])
                    if np.isfinite(chi2) and chi2 < best_chi2 * 0.85:
                        results.append({
                            'spt': f'{spt_a}+{spt_b}',
                            'chi2_red': chi2,
                            'scale': sc,
                            'wave_common': w_bin,
                            'model': model_bin,
                            'is_binary': True,
                            'alpha': alpha,
                            'std_class': 'binary',
                        })
                except Exception:
                    continue

    if progress_callback:
        progress_callback(1.0)

    results.sort(key=lambda x: x['chi2_red'])
    return results


# =====================================================================
# PLOTTING
# =====================================================================
CLASS_COLORS = {
    'late_M': '#e74c3c',
    'L_dwarf': '#e67e22',
    'T_dwarf': '#3498db',
    'subdwarf': '#2ecc71',
    'contaminant': '#95a5a6',
    'T_dwarf_candidate': '#1abc9c',
}

CLASS_SYMBOLS = {
    'late_M': 'circle',
    'L_dwarf': 'diamond',
    'T_dwarf': 'star',
    'subdwarf': 'square',
    'contaminant': 'x',
    'T_dwarf_candidate': 'triangle-up',
}


def make_color_color_plot(df, x_col, y_col, class_col='predicted_class',
                          title=None, selected_idx=None, vetting=None,
                          oid_col=None):
    """Create interactive Plotly color-color diagram."""
    fig = go.Figure()

    # Build vetting lookup for fast access
    vet_status = {}
    if vetting is not None and oid_col is not None and oid_col in df.columns:
        for idx, row in df.iterrows():
            vet_status[idx] = vetting.get_status(row[oid_col])

    for cls in df[class_col].unique():
        mask = df[class_col] == cls
        subset = df[mask]
        color = CLASS_COLORS.get(cls, '#777777')
        symbol = CLASS_SYMBOLS.get(cls, 'circle')

        fig.add_trace(go.Scattergl(
            x=subset[x_col],
            y=subset[y_col],
            mode='markers',
            name=cls.replace('_', ' '),
            marker=dict(
                color=color,
                size=4,
                symbol=symbol,
                opacity=0.6,
                line=dict(width=0),
            ),
            customdata=[[idx] for idx in subset.index.values],
            text=[f"ID: {row.get('object_id', idx)}<br>"
                  f"Class: {cls}<br>"
                  f"SpT: {row.get('predicted_spt', 'N/A')}<br>"
                  f"J={row.get('mag_j', row.get('j_mag', 'N/A')):.1f}"
                  if isinstance(row.get('mag_j', row.get('j_mag', None)), (int, float))
                  else f"ID: {row.get('object_id', idx)}<br>Class: {cls}"
                  for idx, row in subset.iterrows()],
            hovertemplate='%{text}<extra></extra>',
        ))

    # --- Vetting overlays: rings around accepted (green) / rejected (red) ---
    if vet_status:
        for decision, ring_color, label in [
            ('accepted',  '#00ff88', '✓ Accepted'),
            ('rejected',  '#ff4444', '✗ Rejected'),
        ]:
            idxs = [i for i, s in vet_status.items() if s == decision]
            if not idxs:
                continue
            sub = df.loc[idxs]
            fig.add_trace(go.Scattergl(
                x=sub[x_col],
                y=sub[y_col],
                mode='markers',
                name=label,
                marker=dict(
                    color='rgba(0,0,0,0)',
                    size=10,
                    symbol='circle',
                    line=dict(color=ring_color, width=2),
                ),
                customdata=sub.index.values,
                hoverinfo='skip',
            ))

    # Highlight selected point
    if selected_idx is not None and selected_idx in df.index:
        sel = df.loc[selected_idx]
        fig.add_trace(go.Scatter(
            x=[sel[x_col]],
            y=[sel[y_col]],
            mode='markers',
            name='SELECTED',
            marker=dict(color='yellow', size=18, symbol='circle-open',
                        line=dict(color='black', width=3)),
            showlegend=True,
        ))

    fig.update_layout(
        title=title or f'{y_col} vs {x_col}',
        xaxis_title=x_col.replace('_', ' '),
        yaxis_title=y_col.replace('_', ' '),
        template='plotly_dark',
        height=600,
        legend=dict(orientation='h', y=-0.15),
        margin=dict(l=60, r=20, t=50, b=80),
    )

    return fig


def plot_spectral_fit(spectrum, fit_results, top_n=3):
    """Create matplotlib figure of spectral fit."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 1],
                             gridspec_kw={'hspace': 0.05})

    wave = spectrum['wave'] / 10000.0  # Angstrom -> micron
    flux = spectrum['flux']
    noise = spectrum['noise']

    # Top panel: data + models
    ax = axes[0]
    ax.fill_between(wave, flux - noise, flux + noise,
                    alpha=0.2, color='gray', label='±1σ')
    ax.plot(wave, flux, 'k-', lw=0.8, alpha=0.8, label='Observed')

    colors_fit = ['#e74c3c', '#3498db', '#2ecc71']
    for i, res in enumerate(fit_results[:top_n]):
        if res['wave_common'] is not None and res['model'] is not None:
            w_model = res['wave_common'] / 10000.0
            label = (f"{res['spt']} (χ²ᵥ={res['chi2_red']:.2f})"
                     if not res['is_binary']
                     else f"{res['spt']} binary (χ²ᵥ={res['chi2_red']:.2f})")
            ax.plot(w_model, res['model'], '-', color=colors_fit[i % 3],
                    lw=2, alpha=0.8, label=label)

    ax.set_ylabel('Flux [erg/s/cm²/Å]', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title(f"Best fit: {fit_results[0]['spt']}  "
                 f"(χ²ᵥ = {fit_results[0]['chi2_red']:.3f})",
                 fontsize=13, fontweight='bold')
    ax.set_xlim(1.2, 1.9)

    # Bottom panel: chi2 vs SpT
    ax2 = axes[1]
    spt_vals, chi2_vals, classes = [], [], []
    for r in fit_results:
        if not r['is_binary']:
            name = r['spt']
            base = name.replace('sd', '').replace('esd', '')
            if base in SPT_MAP:
                spt_vals.append(SPT_MAP[base])
                chi2_vals.append(r['chi2_red'])
                classes.append(r['std_class'])

    if spt_vals:
        for cls, color, label in [('dwarf', '#e67e22', 'Field'),
                                   ('subdwarf', '#9b59b6', 'sd/esd')]:
            mask = [c == cls for c in classes]
            x = [spt_vals[i] for i in range(len(mask)) if mask[i]]
            y = [chi2_vals[i] for i in range(len(mask)) if mask[i]]
            ax2.scatter(x, y, c=color, s=30, alpha=0.7, label=label, zorder=5)

        tick_pos = list(range(0, 24, 2))
        tick_labels = [SPT_MAP_INV.get(i, '') for i in tick_pos]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels(tick_labels, fontsize=9)
        ax2.set_ylabel('χ²ᵥ', fontsize=11)
        ax2.set_xlabel('Spectral Type', fontsize=11)
        ax2.legend(fontsize=8)
        ax2.set_ylim(0, min(max(chi2_vals) * 1.3, 50))

    plt.tight_layout()
    return fig


def plot_2d_grism(hdul, hdu_idx, object_id):
    """Plot 2D grism image with source trace highlighted."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    # Try to find a 2D image extension
    found_2d = False
    for ext_idx in range(len(hdul)):
        ext = hdul[ext_idx]
        if ext.data is not None and ext.data.ndim == 2:
            img = ext.data
            found_2d = True

            # Display with log stretch
            vmin = np.nanpercentile(img[img > 0], 5) if np.any(img > 0) else 0
            vmax = np.nanpercentile(img, 99)
            ax.imshow(img, origin='lower', aspect='auto',
                      cmap='viridis', vmin=vmin, vmax=vmax,
                      interpolation='nearest')
            ax.set_xlabel('Dispersion (pixels)', fontsize=11)
            ax.set_ylabel('Spatial (pixels)', fontsize=11)
            ax.set_title(f'2D Grism — HDU {ext_idx}  |  Object {object_id}',
                         fontsize=12, fontweight='bold')

            # If we know the trace position (from HDU header)
            crpix2 = ext.header.get('CRPIX2', img.shape[0] // 2)
            # Draw box around expected trace
            y_cen = crpix2
            box_h = 5
            from matplotlib.patches import Rectangle
            rect = Rectangle((0, y_cen - box_h), img.shape[1], 2 * box_h,
                              linewidth=2, edgecolor='red', facecolor='none',
                              linestyle='--', label='Source trace')
            ax.add_patch(rect)
            ax.legend(fontsize=9, loc='upper left')
            break

    if not found_2d:
        # Fallback: show 1D as if it were 2D
        if hdu_idx < len(hdul) and hdul[hdu_idx].data is not None:
            tbl = hdul[hdu_idx].data
            colnames = [c.upper() for c in tbl.dtype.names]
            if 'SIGNAL' in colnames:
                sig = np.array(tbl['SIGNAL'], dtype=float)
                # Reshape into a fake 2D strip
                strip = np.tile(sig, (20, 1))
                ax.imshow(strip, origin='lower', aspect='auto',
                          cmap='inferno', interpolation='nearest')
                ax.set_xlabel('Dispersion (pixels)', fontsize=11)
                ax.set_ylabel('Spatial (pixels)', fontsize=11)
                ax.set_title(f'1D Spectrum (as strip) — Object {object_id}',
                             fontsize=12, fontweight='bold')
                from matplotlib.patches import Rectangle
                rect = Rectangle((0, 7), len(sig), 6,
                                  linewidth=2, edgecolor='red', facecolor='none',
                                  linestyle='--', label='Source trace')
                ax.add_patch(rect)
                ax.legend(fontsize=9)
                found_2d = True

    if not found_2d:
        ax.text(0.5, 0.5, 'No 2D data available in this file',
                transform=ax.transAxes, ha='center', va='center', fontsize=14)

    plt.tight_layout()
    return fig


# =====================================================================
# IRSA SPECTRAL ASSOCIATION QUERY
# =====================================================================
def query_spectral_association(object_id):
    """Query IRSA TAP for spectral file association."""
    # Handle float64 IDs (pandas stores big ints as float)
    oid_int = int(float(object_id))
    q = f"""SELECT objectid, uri, tileid, hdu
            FROM euclid.objectid_spectrafile_association_q1
            WHERE objectid = {oid_int}"""

    # Try method 1: astroquery Irsa
    try:
        from astroquery.ipac.irsa import Irsa
        result = Irsa.query_tap(q).to_table().to_pandas()
        result.columns = [c.lower() for c in result.columns]
        if len(result) > 0:
            return result.iloc[0]
    except TypeError:
        pass  # API mismatch, try fallback
    except Exception as e:
        st.caption(f"Irsa.query_tap: {e}")

    # Try method 2: pyvo TAP service directly
    try:
        import pyvo
        tap = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
        result = tap.run_sync(q).to_table().to_pandas()
        result.columns = [c.lower() for c in result.columns]
        if len(result) > 0:
            return result.iloc[0]
    except ImportError:
        pass  # pyvo not installed
    except Exception as e:
        st.caption(f"pyvo TAP: {e}")

    # Try method 3: raw HTTP TAP query
    try:
        import urllib.request, urllib.parse
        tap_url = "https://irsa.ipac.caltech.edu/TAP/sync"
        params = urllib.parse.urlencode({
            'REQUEST': 'doQuery',
            'LANG': 'ADQL',
            'QUERY': q,
            'FORMAT': 'json'
        })
        req = urllib.request.Request(f"{tap_url}?{params}")
        with urllib.request.urlopen(req, timeout=30) as resp:
            import json as jmod
            data = jmod.loads(resp.read().decode())
            cols = [c['name'].lower() for c in data['metadata']]
            if data.get('data') and len(data['data']) > 0:
                row = dict(zip(cols, data['data'][0]))
                return pd.Series(row)
    except Exception as e:
        st.warning(f"IRSA TAP query failed: {e}")

    return None


# =====================================================================
# MAIN APP
# =====================================================================
def main():
    # --- Header ---
    st.title("🔭 Euclid Q1 UCD Explorer")
    st.caption(f"v{APP_VERSION}  •  Interactive exploration of the ML UCD catalog  •  "
               "Tata, Domínguez-Tagle, Martín, Žerjal & Mohandasan (2026)")

    # --- Sidebar: Configuration ---
    st.sidebar.header("⚙️ Configuration")

    catalog_path = st.sidebar.text_input("Catalog CSV path",
                                          value=DEFAULT_CATALOG)
    standards_path = st.sidebar.text_input("Standards PKL path",
                                            value=DEFAULT_STANDARDS)
    spectra_dir = st.sidebar.text_input("Pre-extracted spectra dir",
                                         value=DEFAULT_SPECTRA)

    # --- Vetting persistence ---
    vetting_path = st.sidebar.text_input("Vetting file",
                                          value=DEFAULT_VETTING_FILE)
    vetting = VettingManager(vetting_path)

    # --- Sidebar: Vetting Dashboard ---
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Vetting Progress")
    n_acc, n_rej, n_tot = vetting.n_accepted, vetting.n_rejected, vetting.n_total
    vc1, vc2, vc3 = st.sidebar.columns(3)
    vc1.metric("✓ Accepted", n_acc)
    vc2.metric("✗ Rejected", n_rej)
    vc3.metric("Total", n_tot)

    if n_tot > 0:
        st.sidebar.progress(n_acc / max(n_tot, 1),
                            text=f"{n_acc}/{n_tot} accepted "
                                 f"({100*n_acc/n_tot:.0f}%)")

    # Reviewer name (persists in session)
    if 'reviewer' not in st.session_state:
        st.session_state.reviewer = ""
    st.session_state.reviewer = st.sidebar.text_input(
        "👤 Reviewer initials", value=st.session_state.reviewer,
        max_chars=10, help="Your initials — saved with each decision")

    # Export / Import
    st.sidebar.markdown("---")
    exp_col1, exp_col2 = st.sidebar.columns(2)
    with exp_col1:
        if st.sidebar.button("📥 Export CSV"):
            vet_df = vetting.to_dataframe()
            if len(vet_df) > 0:
                csv_path = vetting_path.replace('.json', '_export.csv')
                vet_df.to_csv(csv_path, index=False)
                st.sidebar.success(f"Saved {csv_path}")
            else:
                st.sidebar.info("No decisions to export")
    with exp_col2:
        uploaded = st.sidebar.file_uploader("📤 Import JSON", type=['json'],
                                             key="import_vetting")
        if uploaded is not None:
            try:
                imported = json.load(uploaded)
                # Merge: imported decisions overwrite existing
                for oid, info in imported.items():
                    vetting._data[oid] = info
                vetting._save()
                st.sidebar.success(f"Imported {len(imported)} decisions")
            except Exception as e:
                st.sidebar.error(f"Import failed: {e}")

    # Filter by vetting status
    st.sidebar.markdown("---")
    vet_filter = st.sidebar.radio(
        "🔍 Filter by vetting status",
        options=["All", "Unvetted only", "Accepted only", "Rejected only"],
        index=0, horizontal=True)

    # Load data
    catalog = load_catalog(catalog_path)
    standards = load_standards(standards_path)

    if catalog is None:
        st.error("❌ No catalog found. Please provide the correct path to "
                 "your classified catalog CSV.")
        st.info("Run the pipeline first: `python run_full_q1_pipeline.py`")

        # Demo mode with synthetic data
        st.warning("🎮 Loading demo mode with synthetic data...")
        catalog = generate_demo_catalog()

    if standards is None:
        st.sidebar.warning("⚠️ No standards loaded — live fitting disabled")

    # Try to load pre-extracted spectra
    local_spectra = {}
    if os.path.isdir(spectra_dir):
        local_spectra = load_extracted_spectra(spectra_dir)
        st.sidebar.info(f"📊 {len(local_spectra)} pre-extracted spectra loaded")

    # --- Identify available columns ---
    color_cols = [c for c in catalog.columns if 'color' in c.lower() or
                  c in ['y_j', 'j_h', 'y_h', 'ie_ye', 'Y-J', 'J-H']]

    # Normalize column names
    col_map = {}
    for c in catalog.columns:
        cl = c.lower().replace('-', '_').replace(' ', '_')
        if 'y_j' in cl or cl == 'yj' or 'ye_je' in cl:
            col_map['y_j'] = c
        elif 'j_h' in cl or cl == 'jh' or 'je_he' in cl:
            col_map['j_h'] = c
        elif 'ie_ye' in cl or 'vis_y' in cl:
            col_map['ie_ye'] = c
        elif 'predicted_class' in cl or cl == 'class' or cl == 'v2_class':
            col_map['class'] = c
        elif 'predicted_spt' in cl or 'spectral_type' in cl:
            col_map['spt'] = c
        elif 'object_id' in cl or cl == 'objectid':
            col_map['object_id'] = c
        elif 'mag_j' in cl or cl == 'j_mag':
            col_map['mag_j'] = c

    class_col = col_map.get('class', catalog.columns[0])

    # --- Main Panel: Color-Color Plot ---
    st.subheader("📊 Color-Color Diagram")

    col1, col2, col3 = st.columns([1, 1, 1])
    available_cols = [c for c in catalog.columns
                      if catalog[c].dtype in ['float64', 'float32', 'int64']]

    x_default = col_map.get('j_h', available_cols[0] if available_cols else None)
    y_default = col_map.get('y_j', available_cols[1] if len(available_cols) > 1 else None)

    with col1:
        x_col = st.selectbox("X axis", available_cols,
                              index=available_cols.index(x_default)
                              if x_default in available_cols else 0)
    with col2:
        y_col = st.selectbox("Y axis", available_cols,
                              index=available_cols.index(y_default)
                              if y_default in available_cols else min(1, len(available_cols)-1))
    with col3:
        all_classes = sorted(catalog[class_col].unique())
        default_classes = [c for c in all_classes if 'contam' not in c.lower()]
        class_filter = st.multiselect(
            "Show classes",
            options=all_classes,
            default=default_classes)

    # Filter
    mask = catalog[class_col].isin(class_filter)
    filtered = catalog[mask].copy()

    # Apply vetting status filter
    oid_col_name = col_map.get('object_id', None)
    if vet_filter != "All" and oid_col_name and oid_col_name in filtered.columns:
        vet_statuses = filtered[oid_col_name].apply(
            lambda x: vetting.get_status(x))
        if vet_filter == "Unvetted only":
            filtered = filtered[vet_statuses == "unvetted"]
        elif vet_filter == "Accepted only":
            filtered = filtered[vet_statuses == "accepted"]
        elif vet_filter == "Rejected only":
            filtered = filtered[vet_statuses == "rejected"]

    # Session state for selected object
    if 'selected_idx' not in st.session_state:
        st.session_state.selected_idx = None

    # Object selector
    st.markdown("---")
    sel_col1, sel_col2 = st.columns([3, 1])
    with sel_col1:
        oid_col = col_map.get('object_id', None)
        if oid_col:
            object_ids = filtered[oid_col].dropna().astype(str).tolist()
            selected_oid = st.selectbox(
                "🔍 Select object (or click on plot)",
                options=[''] + object_ids[:2000],
                help="Type an object ID or select from dropdown")
            if selected_oid:
                matches = filtered[filtered[oid_col].astype(str) == selected_oid]
                if len(matches) > 0:
                    st.session_state.selected_idx = matches.index[0]
    with sel_col2:
        if st.button("🎲 Random source"):
            st.session_state.selected_idx = np.random.choice(filtered.index)

    # Draw plot
    fig = make_color_color_plot(filtered, x_col, y_col, class_col,
                                selected_idx=st.session_state.selected_idx,
                                vetting=vetting,
                                oid_col=col_map.get('object_id'))

    # Plotly click handler
    event = st.plotly_chart(fig, width="stretch",
                            on_select="rerun", key="cc_plot")

    # Handle click/select on plot
    try:
        if event and event.selection:
            pts = event.selection.get("points", event.selection.points
                                      if hasattr(event.selection, 'points') else [])
            if pts:
                pt = pts[0]
                # customdata can be: int, [int], or nested
                cd = None
                if isinstance(pt, dict):
                    cd = pt.get('customdata', None)
                else:
                    cd = getattr(pt, 'customdata', None)

                if cd is not None:
                    idx_val = cd[0] if isinstance(cd, (list, tuple, np.ndarray)) else cd
                    if idx_val != st.session_state.get('selected_idx'):
                        st.session_state.selected_idx = idx_val
                        st.rerun()
    except Exception as e:
        st.caption(f"⚠️ Click handler: {e}")

    # --- Side Panel: Selected Object Details ---
    st.markdown("---")

    if st.session_state.selected_idx is not None:
        idx = st.session_state.selected_idx
        if idx in catalog.index:
            obj = catalog.loc[idx]
            render_object_panel(obj, col_map, standards, local_spectra, vetting)
        else:
            st.info("Selected object not found in filtered catalog")
    else:
        st.info("👆 Click on a point in the plot or select an object ID to inspect it")

    # --- Footer: Published catalog summary ---
    with st.expander("📚 Published Spectroscopically Confirmed UCDs"):
        st.markdown("""
        | Catalog | Total | Spec-confirmed | L | T | Fields |
        |---------|-------|----------------|---|---|--------|
        | **Žerjal+2025** | 5,306 | 546 | 329 | 26 | EDF-N/F/S |
        | **Domínguez-Tagle+2025** | — | 178 classified | ~130 | ~48 | EDF-N/F/S |
        | **Mohandasan+2025** | 142 | 142 (33 new) | ~80 | ~5 | EDF-N |
        | **This work** | 27,002 | 141 fitted | 31 | 7 | EDF-N/F/S |

        *Estimated ~570–600 unique spectroscopically confirmed UCDs across all three papers (significant overlap).*
        """)


def render_object_panel(obj, col_map, standards, local_spectra, vetting):
    """Render the detailed panel for a selected object."""
    oid = obj.get(col_map.get('object_id', 'object_id'), 'unknown')
    oid_str = str(oid)

    st.subheader(f"🔎 Object: {oid}")

    # =================================================================
    # VETTING STATUS + ACCEPT / REJECT BUTTONS
    # =================================================================
    current = vetting.get_decision(oid)
    status = current["decision"] if current else "unvetted"

    # Status banner
    status_styles = {
        "accepted": ("✅ ACCEPTED", "#00cc66", "white"),
        "rejected": ("❌ REJECTED", "#cc3333", "white"),
        "unvetted": ("⏳ UNVETTED", "#555555", "#cccccc"),
    }
    label, bg, fg = status_styles[status]
    extra = ""
    if current:
        extra = f" — by {current.get('reviewer','?')} at {current.get('timestamp','')[:16]}"
    st.markdown(
        f'<div style="background:{bg};color:{fg};padding:8px 16px;'
        f'border-radius:8px;font-size:1.1em;font-weight:bold;'
        f'margin-bottom:12px">{label}{extra}</div>',
        unsafe_allow_html=True)

    # Buttons row
    btn_c1, btn_c2, btn_c3 = st.columns([1, 1, 1])
    with btn_c1:
        if st.button("✅ Accept", key=f"accept_{oid}",
                      type="primary" if status != "accepted" else "secondary",
                      width="stretch"):
            notes = st.session_state.get(f"notes_{oid}", "")
            spt = st.session_state.get(f"spt_override_{oid}", "")
            vetting.set_decision(oid, "accepted", notes=notes,
                                 best_fit_spt=spt,
                                 reviewer=st.session_state.get('reviewer', ''))
            st.rerun()
    with btn_c2:
        if st.button("❌ Reject", key=f"reject_{oid}",
                      type="primary" if status != "rejected" else "secondary",
                      width="stretch"):
            notes = st.session_state.get(f"notes_{oid}", "")
            vetting.set_decision(oid, "rejected", notes=notes,
                                 reviewer=st.session_state.get('reviewer', ''))
            st.rerun()
    with btn_c3:
        if status != "unvetted":
            if st.button("🔄 Reset", key=f"reset_{oid}",
                          width="stretch"):
                vetting.clear_decision(oid)
                st.rerun()

    # Notes + optional SpT override
    note_c1, note_c2 = st.columns([3, 1])
    with note_c1:
        existing_notes = current.get("notes", "") if current else ""
        st.text_input("📝 Notes", value=existing_notes,
                      key=f"notes_{oid}",
                      placeholder="e.g. noisy spectrum, possible binary, contaminant…")
    with note_c2:
        existing_spt = current.get("best_fit_spt", "") if current else ""
        st.text_input("SpT override", value=existing_spt,
                      key=f"spt_override_{oid}",
                      placeholder="e.g. L3, T2+L8")

    # =================================================================
    # PROPERTIES (compact)
    # =================================================================
    st.markdown("---")
    st.markdown("#### 📋 Properties")

    # Build property table as columns for compact display
    prop_cols = st.columns(5)
    prop_items = [
        (col_map.get('class', 'predicted_class'), 'RF Class'),
        (col_map.get('spt', 'predicted_spt'), 'Phot SpT'),
        (col_map.get('mag_j', 'mag_j'), 'J mag'),
        ('y_j_color', 'Y−J'),
        ('j_h_color', 'J−H'),
        ('ie_ye_color', 'IE−YE'),
        ('has_vis', 'VIS?'),
        ('prob_ucd', 'P(UCD)'),
        ('ra', 'RA'),
        ('dec', 'Dec'),
    ]
    col_idx = 0
    for key, label in prop_items:
        if key in obj.index:
            val = obj[key]
            if isinstance(val, float) and not np.isnan(val):
                disp = f"{val:.4f}" if label in ['RA', 'Dec'] else f"{val:.3f}"
            else:
                disp = str(val)
            prop_cols[col_idx % 5].markdown(f"**{label}:** `{disp}`")
            col_idx += 1

    # VIS dropout warning
    if obj.get('has_vis', 1) == 0:
        st.warning("⚠️ VIS dropout — potential T dwarf or artifact")

    # Class badge
    cls = obj.get(col_map.get('class', 'predicted_class'), 'unknown')
    color = CLASS_COLORS.get(cls, '#777')
    st.markdown(f'<span style="background:{color};color:white;'
                f'padding:4px 12px;border-radius:12px;font-weight:bold">'
                f'{cls.replace("_", " ")}</span>',
                unsafe_allow_html=True)

    # =================================================================
    # 1D SPECTRUM (auto-fetched from S3)
    # =================================================================
    st.markdown("---")
    st.markdown("#### 📈 NISP 1D Spectrum")

    # Check for pre-extracted local spectrum first
    spectrum = None
    for key in [oid_str, str(int(float(oid_str))) if oid_str.replace('-','').replace('.','').isdigit() else '']:
        if key in local_spectra:
            spectrum = local_spectra[key]
            st.info("📁 Using locally pre-extracted spectrum")
            break

    # If no local spectrum, auto-fetch from S3
    s3_data = None
    if spectrum is None:
        status_box = st.empty()
        status_box.info("🌐 Querying IRSA TAP for spectral association...")

        s3_data = auto_fetch_spectra(oid)

        if s3_data['status'] == 'ok' and s3_data['spectrum']:
            spectrum = s3_data['spectrum']
            status_box.success(
                f"✅ Spectrum retrieved! Tile `{s3_data['tileid']}`, "
                f"HDU {s3_data['hdu_idx']}, "
                f"`{s3_data['combspec'][:60]}...`")
        elif s3_data['status'] == 'no_association':
            status_box.info(
                "ℹ️ No NISP spectrum for this object (not in spectral footprint)")
        elif s3_data['status'] == 'download_failed':
            status_box.error(
                f"❌ S3 download failed: {s3_data.get('error', 'unknown')}")
        elif s3_data['status'] == 'no_spectrum':
            status_box.warning(
                "⚠️ COMBSPEC file downloaded but no valid 1D data extracted from HDU")

        # Re-fetch button
        cache_key = f"spectra_cache_{oid}"
        if cache_key in st.session_state:
            if st.button("🔄 Re-fetch spectrum from S3", key=f"refetch_{oid}"):
                del st.session_state[cache_key]
                st.rerun()

    # --- Display 1D spectrum ---
    if spectrum is not None:
        fig_raw, ax = plt.subplots(1, 1, figsize=(12, 4))
        w = spectrum.get('wave_um', spectrum['wave'] / 10000.0)
        ax.fill_between(w, spectrum['flux'] - spectrum['noise'],
                        spectrum['flux'] + spectrum['noise'],
                        alpha=0.3, color='steelblue', label='±1σ noise')
        ax.plot(w, spectrum['flux'], 'k-', lw=0.8, label='Calibrated flux')
        ax.set_xlabel('Wavelength (μm)', fontsize=12)
        ax.set_ylabel('Flux', fontsize=12)
        ax.set_title(f'1D NISP Spectrum — {oid}', fontsize=13, fontweight='bold')
        ax.set_xlim(1.1, 2.0)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig_raw)
        plt.close(fig_raw)

        # --- TEMPLATE FITTING ---
        if standards is not None:
            st.markdown("---")
            st.markdown("#### 🔬 Template Fitting")
            fit_c1, fit_c2 = st.columns([1, 3])
            with fit_c1:
                try_binary = st.checkbox("Try binary composites", value=True,
                                         key=f"bin_{oid}")
                fit_button = st.button("Run Template Fit", key=f"fit_{oid}")
            with fit_c2:
                if fit_button:
                    progress_bar = st.progress(0, text="Fitting against 37 templates...")
                    results = run_full_fit(
                        spectrum, standards,
                        try_binaries=try_binary,
                        progress_callback=lambda p: progress_bar.progress(
                            p, text=f"Fitting... {int(p*100)}%"))

                    if results:
                        best = results[0]
                        if best['chi2_red'] < 5:
                            st.success(f"✅ Best fit: **{best['spt']}** "
                                       f"(χ²ᵥ = {best['chi2_red']:.3f})")
                        elif best['chi2_red'] < 20:
                            st.warning(f"⚠️ Marginal fit: **{best['spt']}** "
                                       f"(χ²ᵥ = {best['chi2_red']:.2f})")
                        else:
                            st.error(f"❌ Poor fit: {best['spt']} "
                                     f"(χ²ᵥ = {best['chi2_red']:.1f})")

                        top_df = pd.DataFrame([{
                            'Rank': i+1,
                            'SpT': r['spt'],
                            'χ²ᵥ': f"{r['chi2_red']:.3f}",
                            'Type': 'binary' if r['is_binary'] else r['std_class'],
                        } for i, r in enumerate(results[:5])])
                        st.dataframe(top_df, hide_index=True)

                        fig_spec = plot_spectral_fit(spectrum, results)
                        st.pyplot(fig_spec)
                        plt.close(fig_spec)
                    else:
                        st.error("Spectrum failed quality check (noise-dominated)")
                    progress_bar.empty()
    else:
        st.markdown("*No spectrum available for display.*")

    # =================================================================
    # 2D GRISM SPECTROGRAM
    # =================================================================
    st.markdown("---")
    st.markdown("#### 📸 2D Grism Spectrogram")

    if s3_data and s3_data.get('hdul_bytes'):
        from astropy.io import fits as afits
        st.info("🖼️ Rendering 2D grism from downloaded COMBSPEC...")
        try:
            hdul = afits.open(io.BytesIO(s3_data['hdul_bytes']))
            hdu_idx = s3_data['hdu_idx']
            fig_2d = plot_2d_grism(hdul, hdu_idx, oid)
            st.pyplot(fig_2d)
            plt.close(fig_2d)
            hdul.close()
        except Exception as e:
            st.warning(f"Could not render 2D grism: {e}")
    elif spectrum is not None and oid_str in local_spectra:
        st.info("2D grism not available for locally pre-extracted spectra")
    elif s3_data and s3_data['status'] == 'no_association':
        st.info("No grism data — object has no NISP spectral coverage")
    else:
        st.info("No 2D grism data available")


def generate_demo_catalog():
    """Generate synthetic demo catalog for testing."""
    np.random.seed(42)
    n = 5000

    # Simulate UCDs
    classes = np.random.choice(
        ['late_M', 'L_dwarf', 'T_dwarf', 'subdwarf', 'contaminant'],
        size=n, p=[0.55, 0.20, 0.03, 0.02, 0.20])

    y_j = np.zeros(n)
    j_h = np.zeros(n)

    for i, cls in enumerate(classes):
        if cls == 'late_M':
            y_j[i] = np.random.normal(0.5, 0.15)
            j_h[i] = np.random.normal(0.4, 0.12)
        elif cls == 'L_dwarf':
            y_j[i] = np.random.normal(0.9, 0.2)
            j_h[i] = np.random.normal(0.5, 0.15)
        elif cls == 'T_dwarf':
            y_j[i] = np.random.normal(1.2, 0.3)
            j_h[i] = np.random.normal(-0.2, 0.2)
        elif cls == 'subdwarf':
            y_j[i] = np.random.normal(0.7, 0.15)
            j_h[i] = np.random.normal(0.15, 0.1)
        else:
            y_j[i] = np.random.normal(0.4, 0.3)
            j_h[i] = np.random.normal(0.3, 0.3)

    ie_ye = np.where(classes == 'T_dwarf', -99,
                     np.random.normal(3.5, 1.0, n))
    mag_j = np.random.normal(20.5, 1.5, n)

    df = pd.DataFrame({
        'object_id': np.arange(n) - 580000000000000000,
        'y_j_color': y_j,
        'j_h_color': j_h,
        'ie_ye_color': ie_ye,
        'mag_j': mag_j,
        'predicted_class': classes,
        'predicted_spt': np.random.uniform(6, 20, n).astype(int),
        'has_vis': (classes != 'T_dwarf').astype(int),
        'prob_ucd': np.random.uniform(0.3, 1.0, n),
        'ra': np.random.uniform(260, 280, n),
        'dec': np.random.uniform(60, 70, n),
    })

    return df


if __name__ == '__main__':
    main()
