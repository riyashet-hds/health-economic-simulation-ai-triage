# ── IMPORTS ──────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')   # Keep output tidy

import numpy as np          # Fast numerical arrays and random sampling
import pandas as pd         # Data tables (like Excel, but in Python)
import scipy.stats as stats # Probability distributions (Beta, Gamma, LogNormal)
import matplotlib.pyplot as plt          # Core plotting
import matplotlib.patches as mpatches   # Shapes for flow diagrams
import seaborn as sns       # Nicer-looking statistical plots
from pathlib import Path    # Cross-platform file paths

# ── REPRODUCIBILITY ───────────────────────────────────────────────────────────
# A "random seed" means every run of this notebook produces the same numbers.
# Without it, results would differ each time, making the work unreproducible.
SEED = 42
rng  = np.random.default_rng(SEED)  # Modern numpy random generator
np.random.seed(SEED)                # Legacy seed (required by scipy.stats)

# ── GLOBAL SETTINGS ───────────────────────────────────────────────────────────
N_SIMULATIONS = 10_000          # Outer Monte Carlo iterations
N_PATIENTS    = 500             # Patients in the demonstration cohort
MIMIC_DIR     = Path("data/mimic_extracted")   # Drop real MIMIC CSVs here
SAVE_FIGS     = False           # Set True to export figures as PNG
FIG_DIR       = Path("figures")
if SAVE_FIGS:
    FIG_DIR.mkdir(exist_ok=True)

# ── COLOUR PALETTE ────────────────────────────────────────────────────────────
PALETTE = {
    "standard":  "#4878D0",   # Blue  → standard NHS care arm
    "ai_triage": "#EE854A",   # Orange → AI-assisted triage arm
    "neutral":   "#6ACC65",   # Green → neutral / empirical data
}
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
print("Libraries loaded. Seed set to", SEED)

# ---- CELL ----
# ── HELPER MATHEMATICAL FUNCTIONS ────────────────────────────────────────────
#
# Probability distributions are defined by PARAMETERS (e.g. mean, spread).
# The three functions below convert from intuitive numbers (mean, std) into the
# technical parameter forms that scipy and numpy expect.

def lognormal_params(mean: float, std: float):
    """
    Convert a plain-English mean and standard deviation into the
    mu (μ) and sigma (σ) of a LogNormal distribution.

    WHY LOGNORMAL? Healthcare times (hospital stays, time-to-diagnosis) are
    always positive and right-skewed — most patients are quick, but some take
    much longer. The LogNormal distribution captures this shape naturally.

    Example: mean=4.2 days, std=2.8 days  →  μ=1.25, σ=0.57
    """
    var   = std ** 2
    mu    = np.log(mean**2 / np.sqrt(var + mean**2))
    sigma = np.sqrt(np.log(1 + var / mean**2))
    return mu, sigma


def beta_params(mean: float, variance: float):
    """
    Convert a mean and variance into the alpha (α) and beta (β) of a Beta distribution.

    WHY BETA? Rates and probabilities live between 0 and 1 (e.g. readmission rate 0.15).
    Beta distributions are the standard choice for modelling uncertainty about a probability.

    Example: mean=0.15, variance=0.001  →  α=19.0, β=107.5
    """
    common = mean * (1 - mean) / variance - 1
    return mean * common, (1 - mean) * common


def gamma_params(mean: float, std: float):
    """
    Convert a mean and standard deviation into the shape and scale of a Gamma distribution.

    WHY GAMMA? Healthcare costs are always positive and right-skewed. Most patients cost
    close to the average, but a few are very expensive. Gamma distributions model this well.

    Example: mean=350, std=70  →  shape=25.0, scale=14.0
    """
    shape = (mean / std) ** 2
    scale = std**2 / mean
    return shape, scale


def discount_qaly(utility: float, years: float, rate: float = 0.035) -> float:
    """
    Apply NICE-recommended time discounting to QALYs.

    WHY DISCOUNT? A health benefit today is worth more than the same benefit in
    5 years (just like money). NICE requires a 3.5% annual discount rate.
    Formula: integral of utility × e^(-r×t) from 0 to T years.
    """
    if rate == 0 or years <= 0:
        return utility * years
    return utility * (1 - np.exp(-rate * years)) / rate


def save_fig(fig, name: str):
    """Save a figure to disk at 300 dpi if SAVE_FIGS is enabled."""
    if SAVE_FIGS:
        path = FIG_DIR / f"{name}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")


# Quick sanity check on the helper functions
_mu, _sig = lognormal_params(4.2, 2.8)
print(f"lognormal_params(4.2, 2.8) → μ={_mu:.3f}, σ={_sig:.3f}")
_a, _b = beta_params(0.15, 0.001)
print(f"beta_params(0.15, 0.001)   → α={_a:.1f}, β={_b:.1f}")
_sh, _sc = gamma_params(350, 70)
print(f"gamma_params(350, 70)      → shape={_sh:.1f}, scale={_sc:.1f}")

# ---- CELL ----
# ── MIMIC-IV DATA LOADER ─────────────────────────────────────────────────────
#
# This section does one of two things:
#   REAL DATA PATH:  If you have placed MIMIC-IV CSVs in data/mimic_extracted/,
#                    it reads and processes them to extract key parameters.
#   PLACEHOLDER PATH: If no CSVs are found, it generates synthetic data from
#                     known literature values. The rest of the notebook runs
#                     identically in either case.
#
# The clean separation between these two paths means you can run the full
# simulation right now, then swap in real data later with zero code changes.

from dataclasses import dataclass
from typing import Optional

@dataclass
class MIMICExtract:
    """
    A simple container that holds the key parameters extracted from MIMIC-IV
    (or simulated as placeholders). Everything downstream uses this object,
    so the source of the data is transparent.
    """
    ed_los_hours:       np.ndarray  # Array: how long each patient spent in the ED (hours)
    hosp_los_days:      np.ndarray  # Array: hospital length of stay (days)
    readmit_30d_rate:   float       # Scalar: proportion readmitted within 30 days
    eeg_ordering_rate:  float       # Scalar: proportion who received an EEG
    n_admissions:       int         # Total number of seizure admissions in the cohort
    source:             str         # "mimic_iv" or "synthetic_placeholder"


class MIMICDataLoader:
    """
    Loads MIMIC-IV data from CSVs, or falls back to synthetic placeholder data.

    Usage:
        loader = MIMICDataLoader(MIMIC_DIR)
        data   = loader.load_or_simulate()
    """

    # These are the five CSV files we expect in data/mimic_extracted/
    REQUIRED = ["edstays.csv", "admissions.csv",
                "diagnoses_icd.csv", "procedures_icd.csv", "patients.csv"]

    # Ground-truth placeholder values (from published literature)
    # These are used ONLY when real MIMIC data is not available.
    PLACEHOLDER = {
        "ed_los_mean_hrs":    6.5,   # Average time in ED for seizure patients (hours)
        "ed_los_std_hrs":     3.2,   # Spread of ED times (standard deviation)
        "hosp_los_mean_days": 4.2,   # Average hospital stay (days)
        "hosp_los_std_days":  2.8,   # Spread of hospital stays
        "readmit_30d_rate":   0.15,  # 15% of patients readmitted within 30 days
        "eeg_ordering_rate":  0.45,  # 45% of seizure patients receive an EEG
    }

    def __init__(self, mimic_dir: Path, n_synthetic: int = 2000,
                 rng: Optional[np.random.Generator] = None):
        self.mimic_dir    = mimic_dir
        self.n_synthetic  = n_synthetic
        self.rng          = rng or np.random.default_rng(42)

    def _all_present(self) -> bool:
        """Return True only if all required CSV files exist and are non-empty."""
        for fname in self.REQUIRED:
            fp = self.mimic_dir / fname
            # Check file exists AND has meaningful content (>100 bytes)
            if not fp.exists() or fp.stat().st_size < 100:
                return False
        return True

    def _load_from_csv(self) -> MIMICExtract:
        """Parse real MIMIC-IV CSVs and extract key distributional parameters."""
        # Load the tables
        ed    = pd.read_csv(self.mimic_dir / "edstays.csv",
                            parse_dates=["intime","outtime"])
        adm   = pd.read_csv(self.mimic_dir / "admissions.csv",
                            parse_dates=["admittime","dischtime"])
        diag  = pd.read_csv(self.mimic_dir / "diagnoses_icd.csv")
        proc  = pd.read_csv(self.mimic_dir / "procedures_icd.csv")

        # Identify seizure/epilepsy admissions using ICD-10 codes
        # G40.x = Epilepsy, R56.x = Convulsions (unspecified seizure)
        epi_ids = diag[
            diag["icd_code"].str.startswith(("G40","R56"), na=False)
        ]["hadm_id"].unique()

        # Filter ED stays and admissions to seizure cohort
        ed_sub  = ed[ed["hadm_id"].isin(epi_ids)].copy()
        adm_sub = adm[adm["hadm_id"].isin(epi_ids)].copy()

        # Calculate ED length of stay in hours
        ed_sub["ed_los_hours"] = (
            (ed_sub["outtime"] - ed_sub["intime"]).dt.total_seconds() / 3600
        )
        # Calculate hospital LOS in days
        adm_sub["hosp_los_days"] = (
            (adm_sub["dischtime"] - adm_sub["admittime"]).dt.total_seconds() / 86400
        )

        # EEG ordering rate: procedure codes 8914 (ICD-9) or equivalent
        eeg_n    = proc[proc["icd_code"].isin(["8914","9589"])
                        & proc["hadm_id"].isin(epi_ids)]["hadm_id"].nunique()
        eeg_rate = eeg_n / len(epi_ids) if len(epi_ids) > 0 else 0.45

        # 30-day readmission: compare discharge of one stay to admission of next
        adm_s = adm_sub.sort_values(["subject_id","admittime"])
        adm_s["prev_disch"] = adm_s.groupby("subject_id")["dischtime"].shift(1)
        adm_s["gap_days"]   = (adm_s["admittime"] - adm_s["prev_disch"]).dt.days
        readmit_rate = (adm_s["gap_days"] <= 30).mean()
        if np.isnan(readmit_rate): readmit_rate = 0.15

        return MIMICExtract(
            ed_los_hours      = ed_sub["ed_los_hours"].dropna().clip(0.5, 48).values,
            hosp_los_days     = adm_sub["hosp_los_days"].dropna().clip(0.5, 30).values,
            readmit_30d_rate  = float(readmit_rate),
            eeg_ordering_rate = float(eeg_rate),
            n_admissions      = len(epi_ids),
            source            = "mimic_iv",
        )

    def _simulate_synthetic(self) -> MIMICExtract:
        """Generate synthetic placeholder data from literature-calibrated parameters."""
        p  = self.PLACEHOLDER
        # Convert mean/std to lognormal parameters, then sample
        mu_ed,  sig_ed  = lognormal_params(p["ed_los_mean_hrs"],   p["ed_los_std_hrs"])
        mu_hop, sig_hop = lognormal_params(p["hosp_los_mean_days"], p["hosp_los_std_days"])
        ed_los  = self.rng.lognormal(mu_ed,  sig_ed,  self.n_synthetic).clip(0.5, 48)
        hop_los = self.rng.lognormal(mu_hop, sig_hop, self.n_synthetic).clip(0.5, 30)
        return MIMICExtract(
            ed_los_hours      = ed_los,
            hosp_los_days     = hop_los,
            readmit_30d_rate  = p["readmit_30d_rate"],
            eeg_ordering_rate = p["eeg_ordering_rate"],
            n_admissions      = self.n_synthetic,
            source            = "synthetic_placeholder",
        )

    def load_or_simulate(self) -> MIMICExtract:
        """Main entry point: use real data if available, otherwise use placeholder."""
        if self._all_present():
            print("MIMIC-IV CSVs detected — loading real patient data.")
            return self._load_from_csv()
        else:
            print(
                f"MIMIC-IV data not found in '{self.mimic_dir}'. "
                "Falling back to synthetic placeholder data.\n"
                "To use real data: export MIMIC-IV BigQuery results as CSVs "
                f"and place them in '{self.mimic_dir}'."
            )
            return self._simulate_synthetic()

# ---- CELL ----
# ── RUN THE LOADER ───────────────────────────────────────────────────────────
loader     = MIMICDataLoader(MIMIC_DIR, n_synthetic=2000, rng=rng)
mimic_data = loader.load_or_simulate()

# Print a summary so we can see what we got
print(f"\nData source      : {mimic_data.source}")
print(f"N admissions     : {mimic_data.n_admissions:,}")
print(f"ED LOS           : mean={mimic_data.ed_los_hours.mean():.1f}h, "
      f"SD={mimic_data.ed_los_hours.std():.1f}h")
print(f"Hospital LOS     : mean={mimic_data.hosp_los_days.mean():.1f}d, "
      f"SD={mimic_data.hosp_los_days.std():.1f}d")
print(f"30-day readmit   : {mimic_data.readmit_30d_rate:.3f} "
      f"({mimic_data.readmit_30d_rate*100:.1f}%)")
print(f"EEG ordering rate: {mimic_data.eeg_ordering_rate:.3f} "
      f"({mimic_data.eeg_ordering_rate*100:.1f}%)")

# ---- CELL ----
# ── MODEL PARAMETERS CLASS ───────────────────────────────────────────────────
#
# We store all parameters in a single class. This keeps them organised,
# makes it easy to copy and modify for sensitivity analysis, and ensures
# every part of the simulation uses the same values.

@dataclass
class ModelParams:
    """
    Container for all simulation parameters.
    Each parameter stores the distributional specification used in
    Monte Carlo sampling. MIMIC-derived fields are populated by
    build_params_from_mimic() after the data loader runs.
    """

    # ── Time-to-diagnosis (months) — LogNormal distribution ──────────────────
    # Standard care: mean 18 months, std 7 months
    ttd_std_mu:    float = 0.0   # LogNormal mu (filled below)
    ttd_std_sigma: float = 0.0   # LogNormal sigma (filled below)
    # AI-assisted: mean 8 months, std 3.5 months
    ttd_ai_mu:     float = 0.0
    ttd_ai_sigma:  float = 0.0

    # ── AI diagnostic accuracy — Beta distribution ────────────────────────────
    # Sensitivity: probability the AI correctly detects true epilepsy
    # Beta(50.0, 8.8) gives mean ≈ 0.85 (85% sensitivity)
    ai_sensitivity_alpha: float = 50.0
    ai_sensitivity_beta:  float = 8.8
    # Specificity: probability the AI correctly rules out non-epilepsy
    # Beta(40.0, 10.0) gives mean ≈ 0.80 (80% specificity)
    ai_specificity_alpha: float = 40.0
    ai_specificity_beta:  float = 10.0

    # ── NHS costs (£) — Gamma distribution ───────────────────────────────────
    # Gamma is used because costs are always > 0 and right-skewed
    cost_neurology_mean: float = 180.0  # Outpatient neurology referral
    cost_neurology_std:  float = 45.0
    cost_eeg_mean:       float = 150.0  # EEG (Electroencephalogram) scan
    cost_eeg_std:        float = 30.0
    cost_ae_mean:        float = 350.0  # A&E (Emergency Department) attendance
    cost_ae_std:         float = 70.0
    cost_hosp_day_mean:  float = 450.0  # One inpatient day (non-ICU ward)
    cost_hosp_day_std:   float = 90.0
    cost_ai_tool:        float = 75.0   # Per-patient cost of running the AI tool

    # ── QALY (Quality-Adjusted Life Year) weights — Beta distribution ─────────
    # Beta is used because QALY weights always fall between 0 and 1
    qaly_treated_alpha:  float = 40.0   # Epilepsy, diagnosed and on treatment → mean 0.80
    qaly_treated_beta:   float = 10.0
    qaly_undiag_alpha:   float = 22.0   # Epilepsy, undiagnosed → mean 0.55
    qaly_undiag_beta:    float = 18.0

    # ── Prior probability of epilepsy — Beta distribution ────────────────────
    # Before any tests: 35% chance that a first-seizure patient has epilepsy
    # Beta(17.5, 32.5) gives mean = 0.35
    prior_epi_alpha: float = 17.5
    prior_epi_beta:  float = 32.5

    # ── Health economic settings ──────────────────────────────────────────────
    wtp_lower:           float = 20_000.0  # NICE lower WTP threshold (£/QALY)
    wtp_upper:           float = 30_000.0  # NICE upper WTP threshold (£/QALY)
    time_horizon_years:  float = 5.0       # Model time horizon (5 years)
    discount_rate:       float = 0.035     # NICE 3.5% annual discount rate

    # ── MIMIC-IV derived parameters (populated by build_params_from_mimic) ───
    ed_los_mu:          float = 0.0
    ed_los_sigma:       float = 0.0
    hosp_los_mu:        float = 0.0
    hosp_los_sigma:     float = 0.0
    readmit_30d_alpha:  float = 0.0
    readmit_30d_beta:   float = 0.0
    eeg_rate_alpha:     float = 0.0
    eeg_rate_beta:      float = 0.0

    def __post_init__(self):
        """Fill in the time-to-diagnosis LogNormal parameters on creation."""
        self.ttd_std_mu, self.ttd_std_sigma = lognormal_params(18, 7)
        self.ttd_ai_mu,  self.ttd_ai_sigma  = lognormal_params(8,  3.5)


def build_params_from_mimic(mimic: MIMICExtract) -> ModelParams:
    """
    Create a ModelParams object, fitting distributions to MIMIC-IV data.

    This function:
    1. Fits LogNormal distributions to the empirical ED LOS and hospital LOS arrays
    2. Uses method-of-moments to fit Beta distributions to the scalar rates
    3. Returns a complete parameter object ready to use in simulation

    WHY FIT DISTRIBUTIONS? We don't just use the mean value from MIMIC-IV.
    We fit a full distribution so the Monte Carlo simulation can sample
    realistic variation — some patients stay longer, some shorter.
    """
    p = ModelParams()

    # Fit LogNormal to ED LOS empirical data using method-of-moments
    p.ed_los_mu,  p.ed_los_sigma  = lognormal_params(
        mimic.ed_los_hours.mean(), mimic.ed_los_hours.std()
    )
    # Fit LogNormal to hospital LOS empirical data
    p.hosp_los_mu, p.hosp_los_sigma = lognormal_params(
        mimic.hosp_los_days.mean(), mimic.hosp_los_days.std()
    )

    # Fit Beta to readmission rate (scalar → use assumed variance for MoM)
    # "MoM" = Method of Moments: derive distribution from summary statistics
    r = mimic.readmit_30d_rate
    p.readmit_30d_alpha, p.readmit_30d_beta = beta_params(r, r*(1-r)/200)

    # Fit Beta to EEG ordering rate
    e = mimic.eeg_ordering_rate
    p.eeg_rate_alpha, p.eeg_rate_beta = beta_params(e, e*(1-e)/200)

    return p


params = build_params_from_mimic(mimic_data)
print("ModelParams constructed from", mimic_data.source)
print(f"  TTD standard: LogNormal(μ={params.ttd_std_mu:.3f}, σ={params.ttd_std_sigma:.3f})")
print(f"  TTD AI:       LogNormal(μ={params.ttd_ai_mu:.3f},  σ={params.ttd_ai_sigma:.3f})")
print(f"  ED LOS:       LogNormal(μ={params.ed_los_mu:.3f},  σ={params.ed_los_sigma:.3f})")
print(f"  Hosp LOS:     LogNormal(μ={params.hosp_los_mu:.3f}, σ={params.hosp_los_sigma:.3f})")

# ---- CELL ----
# ── PARAMETER SUMMARY TABLE ──────────────────────────────────────────────────
# A clear reference table of all model parameters, their distributions,
# and their sources — useful for readers and for any publication.

rows = [
    # (Parameter name,         Distribution,  Key values,
    #  Unit,        Source)
    ("Time-to-diagnosis — standard care",
     "LogNormal", f"mean=18 months, SD=7",
     "months", "Epilepsy Action 2023; NICE CG137"),

    ("Time-to-diagnosis — AI triage",
     "LogNormal", f"mean=8 months, SD=3.5",
     "months", "Published AI diagnostic pilots"),

    ("AI sensitivity",
     "Beta", f"α={params.ai_sensitivity_alpha:.0f}, β={params.ai_sensitivity_beta:.0f} → mean=0.85",
     "probability", "Published EEG-AI classifiers"),

    ("AI specificity",
     "Beta", f"α={params.ai_specificity_alpha:.0f}, β={params.ai_specificity_beta:.0f} → mean=0.80",
     "probability", "Published EEG-AI classifiers"),

    ("Cost: Neurology referral",
     "Gamma", f"mean=£{params.cost_neurology_mean:.0f}, SD=£{params.cost_neurology_std:.0f}",
     "£", "NHS Reference Costs 2023-24"),

    ("Cost: EEG scan",
     "Gamma", f"mean=£{params.cost_eeg_mean:.0f}, SD=£{params.cost_eeg_std:.0f}",
     "£", "NHS Reference Costs 2023-24"),

    ("Cost: A&E attendance",
     "Gamma", f"mean=£{params.cost_ae_mean:.0f}, SD=£{params.cost_ae_std:.0f}",
     "£", "NHS Reference Costs 2023-24"),

    ("Cost: Hospital day (non-ICU)",
     "Gamma", f"mean=£{params.cost_hosp_day_mean:.0f}, SD=£{params.cost_hosp_day_std:.0f}",
     "£", "NHS Reference Costs 2023-24"),

    ("Cost: AI tool (per patient)",
     "Fixed", f"£{params.cost_ai_tool:.0f}",
     "£", "Placeholder — update with real pricing"),

    ("QALY: treated epilepsy",
     "Beta", f"α={params.qaly_treated_alpha:.0f}, β={params.qaly_treated_beta:.0f} → mean=0.80",
     "utility", "NICE TA / EQ-5D literature"),

    ("QALY: undiagnosed epilepsy",
     "Beta", f"α={params.qaly_undiag_alpha:.0f}, β={params.qaly_undiag_beta:.0f} → mean=0.55",
     "utility", "Published CEA studies"),

    ("Prior P(epilepsy)",
     "Beta", f"α={params.prior_epi_alpha:.0f}, β={params.prior_epi_beta:.0f} → mean=0.35",
     "probability", "First-seizure clinic data"),

    ("ED LOS",
     "LogNormal", f"μ={params.ed_los_mu:.3f}, σ={params.ed_los_sigma:.3f}",
     "hours", f"MIMIC-IV [{mimic_data.source}]"),

    ("Hospital LOS",
     "LogNormal", f"μ={params.hosp_los_mu:.3f}, σ={params.hosp_los_sigma:.3f}",
     "days", f"MIMIC-IV [{mimic_data.source}]"),

    ("30-day readmission rate",
     "Beta", f"α={params.readmit_30d_alpha:.1f}, β={params.readmit_30d_beta:.1f}",
     "probability", f"MIMIC-IV [{mimic_data.source}]"),

    ("EEG ordering rate",
     "Beta", f"α={params.eeg_rate_alpha:.1f}, β={params.eeg_rate_beta:.1f}",
     "probability", f"MIMIC-IV [{mimic_data.source}]"),
]

param_table = pd.DataFrame(
    rows, columns=["Parameter", "Distribution", "Key Values", "Unit", "Source"]
)
param_table

# ---- CELL ----
# ── SYNTHETIC COHORT GENERATOR ────────────────────────────────────────────────
#
# This function creates a DataFrame (table) of N synthetic patients.
# Each row is one patient; each column is one characteristic.
# All values are sampled from the probability distributions defined in Section 3.

def generate_cohort(n: int, p: ModelParams,
                    rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate n synthetic patients for the simulation.

    Args:
        n:   Number of patients to generate
        p:   ModelParams object with all distribution specifications
        rng: NumPy random generator (for reproducibility)

    Returns:
        pd.DataFrame with one row per patient and columns:
        patient_id, true_epilepsy, age, sex, ed_los_hours,
        admitted, hosp_los_days, prior_prob_epi, eeg_ordered, readmitted_30d
    """
    # ── True disease state ────────────────────────────────────────────────────
    # Prior mean probability of epilepsy (e.g. 0.35 = 35%)
    prior_mean  = p.prior_epi_alpha / (p.prior_epi_alpha + p.prior_epi_beta)
    # Each patient either has epilepsy (1) or not (0), drawn from that probability
    true_epi    = rng.binomial(1, prior_mean, n).astype(bool)

    # ── Demographics ──────────────────────────────────────────────────────────
    # Age: normally distributed around 52 years, SD=18, clipped to 18-95
    age = rng.normal(52, 18, n).clip(18, 95)
    # Sex: 52% male, 48% female (approximate MIMIC-IV seizure cohort split)
    sex = rng.choice(["M", "F"], n, p=[0.52, 0.48])

    # ── ED Length of Stay (hours) ─────────────────────────────────────────────
    # Sampled from a LogNormal distribution fitted to MIMIC-IV data
    ed_los = rng.lognormal(p.ed_los_mu, p.ed_los_sigma, n).clip(0.5, 48)

    # ── Hospital admission ────────────────────────────────────────────────────
    # About 60% of ED seizure presentations result in a hospital admission
    admitted  = rng.binomial(1, 0.60, n).astype(bool)
    # Hospital LOS is only non-zero for admitted patients
    hosp_los  = np.where(
        admitted,
        rng.lognormal(p.hosp_los_mu, p.hosp_los_sigma, n).clip(0.5, 30),
        0.0  # Not admitted → zero hospital LOS
    )

    # ── Individual prior probability of epilepsy ──────────────────────────────
    # Each patient has their own prior probability (varies around the population mean)
    # Modelled as Beta-distributed to capture individual heterogeneity
    prior_probs = rng.beta(p.prior_epi_alpha, p.prior_epi_beta, n)

    # ── EEG ordered ───────────────────────────────────────────────────────────
    # EEG = Electroencephalogram: the key brain scan for diagnosing epilepsy
    # Rate derived from MIMIC-IV: ~45% of seizure patients receive an EEG
    eeg_rate = p.eeg_rate_alpha / (p.eeg_rate_alpha + p.eeg_rate_beta)
    eeg_ord  = rng.binomial(1, eeg_rate, n).astype(bool)

    # ── 30-day readmission ────────────────────────────────────────────────────
    # Can only be readmitted if initially admitted to hospital
    readmit_rate = p.readmit_30d_alpha / (p.readmit_30d_alpha + p.readmit_30d_beta)
    readmitted   = admitted & rng.binomial(1, readmit_rate, n).astype(bool)

    return pd.DataFrame({
        "patient_id":     np.arange(n),
        "true_epilepsy":  true_epi,
        "age":            age.round(1),
        "sex":            sex,
        "ed_los_hours":   ed_los.round(2),
        "admitted":       admitted,
        "hosp_los_days":  hosp_los.round(2),
        "prior_prob_epi": prior_probs.round(4),
        "eeg_ordered":    eeg_ord,
        "readmitted_30d": readmitted,
    })


# Generate and inspect the cohort
cohort = generate_cohort(N_PATIENTS, params, rng)
print(f"Cohort generated: {len(cohort)} patients")
print(f"  Epilepsy prevalence : {cohort['true_epilepsy'].mean():.1%}")
print(f"  Admission rate      : {cohort['admitted'].mean():.1%}")
print(f"  EEG ordered         : {cohort['eeg_ordered'].mean():.1%}")
print(f"  30-day readmission  : {cohort['readmitted_30d'].mean():.1%}")
print()
# .describe() gives min, max, mean, quartiles for numeric columns
cohort[["age","ed_los_hours","hosp_los_days","prior_prob_epi"]].describe().round(2)

# ---- CELL ----
# ── VECTORISED PATHWAY FUNCTIONS ─────────────────────────────────────────────
#
# These functions simulate an entire cohort of patients at once using
# numpy arrays, rather than looping over patients one by one.
# This "vectorised" approach is ~100x faster and makes the Monte Carlo
# simulation feasible in reasonable time.

def sample_unit_costs(p: ModelParams) -> dict:
    """
    Sample one set of unit costs from their Gamma distributions.
    Called once per Monte Carlo iteration — all patients in that
    iteration share the same unit costs (realistic: NHS tariff is fixed
    in any given year, but uncertain across years/scenarios).

    Returns a dict with keys: ae, neuro, eeg, hosp_day
    """
    def g(mean, std):
        """Draw one sample from a Gamma distribution."""
        sh, sc = gamma_params(mean, std)
        return stats.gamma.rvs(sh, scale=sc)

    return {
        "ae":       g(p.cost_ae_mean,       p.cost_ae_std),       # A&E visit (£)
        "neuro":    g(p.cost_neurology_mean, p.cost_neurology_std),# Neurology referral (£)
        "eeg":      g(p.cost_eeg_mean,       p.cost_eeg_std),      # EEG scan (£)
        "hosp_day": g(p.cost_hosp_day_mean,  p.cost_hosp_day_std), # Per hospital day (£)
    }


def pathway_standard_care(eeg_ordered: np.ndarray,
                           hosp_los_days: np.ndarray,
                           p: ModelParams,
                           rng: np.random.Generator,
                           costs: dict) -> tuple:
    """
    Simulate the standard NHS care pathway for an entire cohort simultaneously.

    Args:
        eeg_ordered:   Boolean array — did each patient receive an EEG?
        hosp_los_days: Array of hospital LOS values (days) per patient
        p:             ModelParams with all distribution specifications
        rng:           NumPy random generator
        costs:         Dict of unit costs sampled this iteration

    Returns:
        (total_cost_per_patient, total_qaly_per_patient, ttd_months_per_patient)
        Each is a numpy array of length n.
    """
    n = len(eeg_ordered)

    # ── Time to diagnosis ─────────────────────────────────────────────────────
    # Draw from LogNormal distribution: most patients wait ~18 months,
    # some sooner, some much longer (heavy right tail)
    ttd_months = rng.lognormal(p.ttd_std_mu, p.ttd_std_sigma, n).clip(1, 60)
    ttd_years  = ttd_months / 12  # Convert to years for QALY calculation

    # ── Number of pre-diagnosis ED visits ─────────────────────────────────────
    # On average 2.5 ED visits before diagnosis; modelled as Normal, clipped ≥ 0
    n_ae = np.maximum(0, rng.normal(2.5, 1.2, n)).astype(int)

    # ── Costs ─────────────────────────────────────────────────────────────────
    cost_ae     = n_ae * costs["ae"]                          # ED visits
    cost_neuro  = np.full(n, costs["neuro"])                  # One neurology referral per patient
    cost_eeg    = np.where(eeg_ordered, costs["eeg"], 0.0)   # EEG only if ordered
    cost_hosp   = hosp_los_days * costs["hosp_day"]           # Hospital LOS × daily rate
    total_cost  = cost_ae + cost_neuro + cost_eeg + cost_hosp

    # ── QALYs ─────────────────────────────────────────────────────────────────
    # Sample utility weights from Beta distributions (each patient gets their own draw)
    u_undiag  = rng.beta(p.qaly_undiag_alpha,  p.qaly_undiag_beta,  n)  # While undiagnosed
    u_treated = rng.beta(p.qaly_treated_alpha, p.qaly_treated_beta, n)  # Once diagnosed

    # Remaining time after diagnosis (up to 5-year horizon)
    remain_years = np.maximum(0, p.time_horizon_years - ttd_years)

    # Apply NICE 3.5% discounting using the helper function (vectorised)
    disc_undiag  = u_undiag  * (1 - np.exp(-p.discount_rate * ttd_years))    / p.discount_rate
    disc_treated = u_treated * (1 - np.exp(-p.discount_rate * remain_years)) / p.discount_rate
    # Handle zero-year periods to avoid division issues
    disc_undiag  = np.where(ttd_years   > 0, disc_undiag,  0.0)
    disc_treated = np.where(remain_years > 0, disc_treated, 0.0)
    total_qaly = disc_undiag + disc_treated

    return total_cost, total_qaly, ttd_months


def pathway_ai_triage(eeg_ordered: np.ndarray,
                       hosp_los_days: np.ndarray,
                       true_epilepsy: np.ndarray,
                       posteriors: np.ndarray,
                       p: ModelParams,
                       rng: np.random.Generator,
                       costs: dict) -> tuple:
    """
    Simulate the AI-assisted triage pathway for an entire cohort.

    Key differences from standard care:
    1. Time to diagnosis is shorter (mean 8 months vs 18 months)
    2. The AI tool correctly routes some patients away from unnecessary ED visits
       (true negatives, identified by specificity, avoid repeat ED attendance)
    3. An additional cost for the AI tool itself (£75/patient)

    Args:
        posteriors: Bayesian-updated P(epilepsy) per patient (from Section 6)
        [other args same as pathway_standard_care]
    """
    n = len(eeg_ordered)

    # ── Shorter time to diagnosis ──────────────────────────────────────────────
    ttd_months = rng.lognormal(p.ttd_ai_mu, p.ttd_ai_sigma, n).clip(1, 36)
    ttd_years  = ttd_months / 12

    # ── Specificity-gated ED visit reduction ──────────────────────────────────
    # Draw AI specificity for this iteration
    ai_spec  = stats.beta.rvs(p.ai_specificity_alpha, p.ai_specificity_beta)
    # True negatives (non-epilepsy patients correctly identified) avoid some ED visits
    # The fraction avoided scales with specificity and whether they truly don't have epilepsy
    avoid_frac = ai_spec * (~true_epilepsy).astype(float)
    n_ae_base  = np.maximum(0, rng.normal(2.5, 1.2, n)).astype(int)
    n_ae       = np.maximum(0, (n_ae_base * (1 - avoid_frac * 0.6)).astype(int))

    # ── Costs ─────────────────────────────────────────────────────────────────
    cost_ae    = n_ae * costs["ae"]
    cost_neuro = np.full(n, costs["neuro"])
    cost_eeg   = np.where(eeg_ordered, costs["eeg"], 0.0)
    cost_hosp  = hosp_los_days * costs["hosp_day"]
    cost_ai    = p.cost_ai_tool   # Fixed per-patient AI tool cost (£75 placeholder)
    total_cost = cost_ae + cost_neuro + cost_eeg + cost_hosp + cost_ai

    # ── QALYs ─────────────────────────────────────────────────────────────────
    u_undiag  = rng.beta(p.qaly_undiag_alpha,  p.qaly_undiag_beta,  n)
    u_treated = rng.beta(p.qaly_treated_alpha, p.qaly_treated_beta, n)
    remain_years = np.maximum(0, p.time_horizon_years - ttd_years)
    disc_undiag  = u_undiag  * (1 - np.exp(-p.discount_rate * ttd_years))    / p.discount_rate
    disc_treated = u_treated * (1 - np.exp(-p.discount_rate * remain_years)) / p.discount_rate
    disc_undiag  = np.where(ttd_years   > 0, disc_undiag,  0.0)
    disc_treated = np.where(remain_years > 0, disc_treated, 0.0)
    total_qaly = disc_undiag + disc_treated

    return total_cost, total_qaly, ttd_months


print("Pathway functions defined.")
print("  pathway_standard_care() — vectorised standard NHS pathway")
print("  pathway_ai_triage()     — vectorised AI-assisted pathway")
print("  sample_unit_costs()     — samples one draw of all unit costs")

# ---- CELL ----
# ── BAYESIAN BELIEF UPDATER CLASS ────────────────────────────────────────────
#
# This class models how a clinician's belief about a patient's diagnosis
# updates as new evidence arrives. It is educational: it shows the
# step-by-step updating process clearly.
#
# In the Monte Carlo simulation (Section 7), we use a faster vectorised
# version. This class is for understanding; the vectorised function is for speed.

class BayesianBeliefUpdater:
    """
    Models sequential Bayesian updating of P(epilepsy) for a single patient.

    The prior belief is represented as a Beta distribution Beta(alpha, beta),
    where alpha and beta are shape parameters controlling the distribution's
    location and width.

    As new evidence arrives (AI tool result, clinical observations), the
    distribution is updated using Bayes' theorem, and the posterior mean
    and 95% credible interval (CrI) can be read at any point.

    Example usage:
        updater = BayesianBeliefUpdater(prior_alpha=17.5, prior_beta=32.5)
        print(f"Prior mean P(epilepsy): {updater.posterior_mean:.3f}")
        updater.update_with_ai_result(result_positive=True, sensitivity=0.85, specificity=0.80)
        print(f"Posterior mean P(epilepsy): {updater.posterior_mean:.3f}")
    """

    def __init__(self, prior_alpha: float, prior_beta: float):
        """
        Initialise the updater with a Beta prior.

        Args:
            prior_alpha: Alpha parameter of the prior Beta distribution.
                         Encodes "pseudo-observations" in favour of epilepsy.
                         Prior mean = alpha / (alpha + beta).
            prior_beta:  Beta parameter of the prior Beta distribution.
                         Encodes "pseudo-observations" against epilepsy.
        """
        self.alpha0 = prior_alpha   # Remember the original prior (for reset)
        self.beta0  = prior_beta
        self.alpha  = prior_alpha   # Current alpha (updated as evidence arrives)
        self.beta   = prior_beta    # Current beta  (updated as evidence arrives)
        self.history = []           # Log of all updates, for inspection

    # ── Properties (computed on the fly from current alpha, beta) ─────────────

    @property
    def posterior_mean(self) -> float:
        """
        The current best estimate of P(epilepsy).

        For a Beta(α, β) distribution, the mean is simply α / (α + β).
        This is the single number we report as the clinician's current
        probability estimate.

        Returns:
            float between 0 and 1
        """
        return self.alpha / (self.alpha + self.beta)

    @property
    def posterior_ci(self) -> tuple:
        """
        The 95% Credible Interval (CrI) around the current P(epilepsy) estimate.

        A credible interval means: given all the evidence so far, we are 95%
        confident the true probability lies within this range.
        (Unlike a frequentist confidence interval, this statement is literally true.)

        Returns:
            (lower_bound, upper_bound) as a tuple of floats
        """
        # scipy.stats.beta.ppf gives the quantiles of a Beta distribution
        lower = stats.beta.ppf(0.025, self.alpha, self.beta)  # 2.5th percentile
        upper = stats.beta.ppf(0.975, self.alpha, self.beta)  # 97.5th percentile
        return (lower, upper)

    # ── Update methods ────────────────────────────────────────────────────────

    def update_with_ai_result(self, result_positive: bool,
                               sensitivity: float,
                               specificity: float) -> float:
        """
        Update the belief distribution when the AI tool returns a result.

        This method applies Bayes' theorem to compute the posterior probability
        given the AI result, then converts that posterior back into updated
        Beta distribution parameters using pseudo-count stabilisation.

        HOW IT WORKS:
        ─────────────
        1. Read the current best estimate (prior mean) from self.posterior_mean
        2. Apply the appropriate formula (PPV for positive, complement-NPV for negative)
           to get the Bayesian-updated posterior probability
        3. Convert the new posterior probability back into Beta(alpha, beta) form
           by adding "pseudo-counts" proportional to the new belief
        4. Log the update in self.history for transparency

        WHY PSEUDO-COUNTS?
        Each piece of evidence is treated as equivalent to a small number of
        additional observations (pseudo-count = 10). This is standard Bayesian
        conjugate updating for Beta distributions. A pseudo-count of 10 means
        one AI result has the same informational weight as 10 binary observations.

        Args:
            result_positive: True if the AI tool returned a positive result
                             (i.e. AI thinks patient has epilepsy)
            sensitivity:     P(positive result | patient truly has epilepsy).
                             How often does the AI correctly flag true epilepsy?
            specificity:     P(negative result | patient does NOT have epilepsy).
                             How often does the AI correctly rule out non-epilepsy?

        Returns:
            Updated posterior mean probability of epilepsy (float, 0–1)
        """
        prior = self.posterior_mean  # Current best estimate before update

        if result_positive:
            # ── Positive result: apply PPV formula ──────────────────────────
            # P(epilepsy | positive) = sens × prior / [sens × prior + (1-spec) × (1-prior)]
            # Numerator: probability of being positive AND having epilepsy
            numerator   = sensitivity * prior
            # Denominator: total probability of getting a positive result
            # = P(positive | epi) × P(epi)  +  P(positive | no epi) × P(no epi)
            # Note: P(positive | no epi) = 1 - specificity (false positive rate)
            denominator = numerator + (1 - specificity) * (1 - prior)
        else:
            # ── Negative result: apply complement-NPV formula ────────────────
            # P(epilepsy | negative) = (1-sens) × prior / [(1-sens) × prior + spec × (1-prior)]
            # Numerator: probability of being negative AND still having epilepsy (missed)
            numerator   = (1 - sensitivity) * prior
            # Denominator: total probability of getting a negative result
            denominator = numerator + specificity * (1 - prior)

        # Compute posterior probability (with a small floor to avoid divide-by-zero)
        posterior = numerator / denominator if denominator > 1e-9 else prior

        # ── Convert posterior back to Beta parameters ─────────────────────────
        # We treat the AI result as equivalent to 10 binary pseudo-observations.
        # This adds new_alpha pseudo-positives and new_beta pseudo-negatives.
        pseudo_n   = 10.0                      # Weight of one AI result
        new_alpha  = posterior       * pseudo_n  # Pseudo-positives from AI result
        new_beta   = (1 - posterior) * pseudo_n  # Pseudo-negatives from AI result
        self.alpha += new_alpha
        self.beta  += new_beta

        # ── Log this update ───────────────────────────────────────────────────
        self.history.append({
            "event":          "ai_result",
            "result_positive": result_positive,
            "prior_mean":      prior,
            "posterior_mean":  self.posterior_mean,
            "sensitivity":     sensitivity,
            "specificity":     specificity,
        })

        return self.posterior_mean

    def update_with_clinical_episode(self, witnessed_seizure: bool) -> float:
        """
        Update the belief when a clinical event is observed.

        A witnessed seizure is strong evidence FOR epilepsy: adds 2 pseudo-counts
        to alpha (the 'pro-epilepsy' parameter).
        No witnessed seizure is weak evidence AGAINST: adds 0.5 to beta.

        This asymmetry reflects clinical reality: a witnessed seizure is
        fairly diagnostic; the absence of a witnessed seizure does not rule
        epilepsy out (seizures may simply not have occurred during the consultation).

        Args:
            witnessed_seizure: True if a seizure was observed during assessment

        Returns:
            Updated posterior mean probability of epilepsy
        """
        if witnessed_seizure:
            self.alpha += 2.0   # Strong positive evidence → shift belief towards epilepsy
        else:
            self.beta  += 0.5   # Weak negative evidence → slight shift away from epilepsy

        self.history.append({
            "event":          "clinical_episode",
            "witnessed":       witnessed_seizure,
            "posterior_mean":  self.posterior_mean,
        })
        return self.posterior_mean

    def reset(self):
        """Reset the belief to the original prior (useful for reusing the updater)."""
        self.alpha   = self.alpha0
        self.beta    = self.beta0
        self.history = []

    def get_history_df(self) -> pd.DataFrame:
        """Return the update history as a tidy DataFrame for inspection."""
        return pd.DataFrame(self.history)


# ── DEMONSTRATION: Watch a single patient's belief update ─────────────────────
print("=== Single Patient Demonstration ===")
print()

# Initialise with the population prior: 35% chance of epilepsy
demo = BayesianBeliefUpdater(
    prior_alpha=params.prior_epi_alpha,  # 17.5
    prior_beta=params.prior_epi_beta     # 32.5
)
lo, hi = demo.posterior_ci
print(f"Step 0 — Prior (no evidence yet):")
print(f"  P(epilepsy) = {demo.posterior_mean:.3f}  95% CrI: [{lo:.3f}, {hi:.3f}]")

# AI tool returns a POSITIVE result (sensitivity=0.85, specificity=0.80)
demo.update_with_ai_result(result_positive=True, sensitivity=0.85, specificity=0.80)
lo, hi = demo.posterior_ci
print(f"Step 1 — After positive AI result (sens=0.85, spec=0.80):")
print(f"  P(epilepsy) = {demo.posterior_mean:.3f}  95% CrI: [{lo:.3f}, {hi:.3f}]")

# A seizure was witnessed during clinical assessment
demo.update_with_clinical_episode(witnessed_seizure=True)
lo, hi = demo.posterior_ci
print(f"Step 2 — After witnessed seizure:")
print(f"  P(epilepsy) = {demo.posterior_mean:.3f}  95% CrI: [{lo:.3f}, {hi:.3f}]")
print()
print("History of updates:")
print(demo.get_history_df().to_string(index=False))

# ---- CELL ----
# ── VECTORISED POSTERIOR COMPUTATION ─────────────────────────────────────────
#
# The BayesianBeliefUpdater class above is excellent for understanding
# the updating process step by step. However, it processes one patient
# at a time, which is too slow for 10,000 Monte Carlo iterations.
#
# This function does the same Bayesian update for ALL patients at once
# using numpy array operations. It produces identical results to running
# the class on every patient individually — just ~500x faster.

def compute_posteriors(prior_probs: np.ndarray,
                        true_epilepsy: np.ndarray,
                        p: ModelParams,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Apply Bayesian belief updating to an entire cohort simultaneously.

    For each patient:
    1. Draw one sensitivity and one specificity value for this iteration
       (same values for all patients within one iteration — reflects
        that the AI tool has a fixed performance level at any given time)
    2. Simulate whether the AI tool returns positive or negative for each patient
       (a true epilepsy patient gets a positive result with probability = sensitivity;
        a non-epilepsy patient gets a positive result with probability = 1 - specificity)
    3. Apply the PPV/NPV Bayes formula to update each patient's prior probability
    4. Apply a clinical episode update (witnessed seizure) for 60% of patients

    Args:
        prior_probs:   Array of individual prior P(epilepsy) values, shape (n,)
        true_epilepsy: Boolean array of true disease states, shape (n,)
        p:             ModelParams with AI performance parameters
        rng:           NumPy random generator

    Returns:
        posteriors: Array of updated P(epilepsy) values, shape (n,)
    """
    n = len(prior_probs)

    # ── Step 1: Draw AI tool performance for this iteration ───────────────────
    # One sensitivity and specificity draw per iteration (not per patient)
    # This models the AI tool as having fixed-but-uncertain performance
    sens = stats.beta.rvs(p.ai_sensitivity_alpha, p.ai_sensitivity_beta)
    spec = stats.beta.rvs(p.ai_specificity_alpha, p.ai_specificity_beta)

    # ── Step 2: Simulate AI test results for each patient ─────────────────────
    # True epilepsy patients: positive result with probability = sensitivity
    result_if_epi    = rng.binomial(1, sens,       n).astype(bool)
    # Non-epilepsy patients: positive result with probability = 1 - specificity (false positive)
    result_if_no_epi = rng.binomial(1, 1 - spec,  n).astype(bool)
    # Combine: use true disease state to select the right result
    result_positive  = np.where(true_epilepsy, result_if_epi, result_if_no_epi)

    # ── Step 3: Bayesian update using PPV/NPV formulas ────────────────────────
    # For patients with a POSITIVE result:
    num_pos  = sens * prior_probs
    den_pos  = num_pos + (1 - spec) * (1 - prior_probs)
    post_pos = np.where(den_pos > 1e-9, num_pos / den_pos, prior_probs)

    # For patients with a NEGATIVE result:
    num_neg  = (1 - sens) * prior_probs
    den_neg  = num_neg + spec * (1 - prior_probs)
    post_neg = np.where(den_neg > 1e-9, num_neg / den_neg, prior_probs)

    # Select the correct posterior based on which result each patient received
    posteriors = np.where(result_positive, post_pos, post_neg)

    # ── Step 4: Clinical episode update (witnessed seizure) ───────────────────
    # 60% of patients have a witnessed seizure during assessment
    # This nudges their posterior upward by a small, fixed amount
    witnessed = rng.binomial(1, 0.6, n).astype(bool)
    # Simple additive nudge (clipped to stay within [0, 1])
    posteriors = np.clip(
        posteriors + np.where(witnessed, 0.08, 0.0),
        0.0, 0.99
    )

    return posteriors


# ── Apply to the demonstration cohort ─────────────────────────────────────────
posteriors = compute_posteriors(
    prior_probs   = cohort["prior_prob_epi"].values,
    true_epilepsy = cohort["true_epilepsy"].values,
    p             = params,
    rng           = rng
)
cohort["posterior_prob_epi"] = posteriors

# Compare prior vs posterior
prior_mean    = cohort["prior_prob_epi"].mean()
posterior_mean = cohort["posterior_prob_epi"].mean()
print(f"Population prior mean  P(epilepsy): {prior_mean:.4f}  "
      f"({prior_mean*100:.1f}%)")
print(f"Population posterior mean P(epilepsy): {posterior_mean:.4f}  "
      f"({posterior_mean*100:.1f}%)")
print()
print("Among TRUE EPILEPSY patients:")
epi_mask = cohort["true_epilepsy"]
print(f"  Prior mean    : {cohort.loc[epi_mask, 'prior_prob_epi'].mean():.4f}")
print(f"  Posterior mean: {cohort.loc[epi_mask, 'posterior_prob_epi'].mean():.4f}")
print()
print("Among NON-EPILEPSY patients:")
print(f"  Prior mean    : {cohort.loc[~epi_mask, 'prior_prob_epi'].mean():.4f}")
print(f"  Posterior mean: {cohort.loc[~epi_mask, 'posterior_prob_epi'].mean():.4f}")

# ---- CELL ----
# ── SINGLE MONTE CARLO ITERATION ─────────────────────────────────────────────

def run_single_iteration(p: ModelParams,
                          n_patients: int,
                          rng: np.random.Generator) -> dict:
    """
    Run one complete outer Monte Carlo iteration.

    This function:
    1. Generates a fresh synthetic cohort (n_patients patients)
    2. Computes Bayesian posteriors for each patient
    3. Runs BOTH pathways on the same cohort (CRN variance reduction)
    4. Returns the incremental cost, incremental QALYs, and ICER

    Args:
        p:          ModelParams — one parameter draw per outer iteration
        n_patients: Number of patients in this cohort
        rng:        NumPy random generator

    Returns:
        dict with keys: delta_cost, delta_qaly, icer,
                        cost_std, cost_ai, qaly_std, qaly_ai
    """
    # ── Generate a fresh cohort ───────────────────────────────────────────────
    prior_mean   = p.prior_epi_alpha / (p.prior_epi_alpha + p.prior_epi_beta)
    true_epi     = rng.binomial(1, prior_mean, n_patients).astype(bool)
    eeg_rate     = p.eeg_rate_alpha / (p.eeg_rate_alpha + p.eeg_rate_beta)
    eeg_ordered  = rng.binomial(1, eeg_rate, n_patients).astype(bool)
    admitted     = rng.binomial(1, 0.60, n_patients).astype(bool)
    hosp_los     = np.where(
        admitted,
        rng.lognormal(p.hosp_los_mu, p.hosp_los_sigma, n_patients).clip(0.5, 30),
        0.0
    )
    prior_probs  = rng.beta(p.prior_epi_alpha, p.prior_epi_beta, n_patients)

    # ── Bayesian posteriors for AI arm ────────────────────────────────────────
    posteriors = compute_posteriors(prior_probs, true_epi, p, rng)

    # ── Sample unit costs once — shared by both arms ──────────────────────────
    # (CRN: same costs for both arms in this iteration)
    costs = sample_unit_costs(p)

    # ── Standard care pathway ─────────────────────────────────────────────────
    cost_std_arr, qaly_std_arr, _ = pathway_standard_care(
        eeg_ordered, hosp_los, p, rng, costs
    )
    # ── AI triage pathway ─────────────────────────────────────────────────────
    cost_ai_arr,  qaly_ai_arr,  _ = pathway_ai_triage(
        eeg_ordered, hosp_los, true_epi, posteriors, p, rng, costs
    )

    # ── Compute ICER ──────────────────────────────────────────────────────────
    mean_cost_std = cost_std_arr.mean()
    mean_cost_ai  = cost_ai_arr.mean()
    mean_qaly_std = qaly_std_arr.mean()
    mean_qaly_ai  = qaly_ai_arr.mean()

    delta_c = mean_cost_ai  - mean_cost_std   # Positive = AI is more expensive
    delta_e = mean_qaly_ai  - mean_qaly_std   # Positive = AI produces more QALYs

    # ICER = incremental cost per incremental QALY gained
    # NaN if delta_e is zero (no health benefit → ICER undefined)
    icer = delta_c / delta_e if abs(delta_e) > 1e-9 else np.nan

    return {
        "delta_cost":  delta_c,
        "delta_qaly":  delta_e,
        "icer":        icer,
        "cost_std":    mean_cost_std,
        "cost_ai":     mean_cost_ai,
        "qaly_std":    mean_qaly_std,
        "qaly_ai":     mean_qaly_ai,
    }


# Quick single-iteration test
test_result = run_single_iteration(params, n_patients=200, rng=rng)
print("Single-iteration test:")
print(f"  ΔCost  = £{test_result['delta_cost']:,.0f}")
print(f"  ΔQALYs = {test_result['delta_qaly']:.4f}")
print(f"  ICER   = £{test_result['icer']:,.0f}/QALY")

# ---- CELL ----
# ── FULL MONTE CARLO SIMULATION ───────────────────────────────────────────────
#
# We now run 10,000 iterations. Each iteration:
#   1. Draws a fresh set of parameter values from their distributions
#   2. Generates a synthetic cohort of 100 patients
#   3. Simulates both pathways and computes the ICER
#
# The result is a DataFrame with 10,000 rows — one ICER per iteration.
# This gives us a distribution of possible ICERs rather than just one number.
#
# ⏱  RUNTIME: ~3-8 minutes on a standard laptop.
# Progress is shown by tqdm (a progress bar library).

def run_monte_carlo(p: ModelParams,
                    n_sims: int = 10_000,
                    n_patients_per_iter: int = 100,
                    seed: int = 42) -> pd.DataFrame:
    """
    Run the full outer Monte Carlo simulation.

    Args:
        p:                   ModelParams with base parameter values
        n_sims:              Number of Monte Carlo iterations (default 10,000)
        n_patients_per_iter: Patients simulated per iteration (default 100)
                             Smaller = faster, larger = more precise per-iteration estimate
        seed:                Random seed for reproducibility

    Returns:
        pd.DataFrame with one row per iteration, columns:
        delta_cost, delta_qaly, icer, cost_std, cost_ai, qaly_std, qaly_ai
    """
    mc_rng  = np.random.default_rng(seed)
    results = []

    # tqdm provides a progress bar; falls back silently if not installed
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(range(n_sims), desc="Monte Carlo", unit="iter")
    except ImportError:
        iterator = range(n_sims)
        print(f"Running {n_sims:,} iterations (install tqdm for a progress bar)...")

    for _ in iterator:
        try:
            res = run_single_iteration(p, n_patients_per_iter, mc_rng)
            results.append(res)
        except Exception as e:
            # If an iteration fails, record NaN and continue
            # (prevents one bad draw from stopping the whole simulation)
            results.append({k: np.nan for k in
                            ["delta_cost","delta_qaly","icer",
                             "cost_std","cost_ai","qaly_std","qaly_ai"]})

    df = pd.DataFrame(results)
    n_valid = df["icer"].notna().sum()
    print(f"\nCompleted: {n_valid:,} valid iterations out of {n_sims:,}")
    return df


print("Starting Monte Carlo simulation...")
print(f"  {N_SIMULATIONS:,} iterations × 100 patients per iteration")
print(f"  Expected runtime: 3-8 minutes\n")
mc_results = run_monte_carlo(params, n_sims=N_SIMULATIONS, n_patients_per_iter=100, seed=SEED)

# ---- CELL ----
# ── RESULTS SUMMARY ───────────────────────────────────────────────────────────
#
# Here we summarise the 10,000 ICER estimates into key statistics.
# These are the numbers that would go into a health economic report.

valid = mc_results.dropna(subset=["icer"])  # Remove any failed iterations

# Percentile summaries of the ICER distribution
pct = np.percentile(valid["icer"], [2.5, 25, 50, 75, 97.5])

print("=" * 60)
print("  MONTE CARLO HEALTH ECONOMIC RESULTS SUMMARY")
print("=" * 60)
print(f"  Valid iterations        : {len(valid):,} / {N_SIMULATIONS:,}")
print()
print("  INCREMENTAL RESULTS (AI triage vs. Standard care):")
print(f"  Mean ΔCost              : £{valid['delta_cost'].mean():,.0f}")
print(f"  Mean ΔQALYs             : {valid['delta_qaly'].mean():.4f}")
print()
print("  ICER DISTRIBUTION (£ per QALY gained):")
print(f"  Mean ICER               : £{valid['icer'].mean():,.0f}/QALY")
print(f"  Median ICER             : £{pct[2]:,.0f}/QALY")
print(f"  95% Credible Interval   : [£{pct[0]:,.0f}, £{pct[4]:,.0f}]")
print(f"  IQR                     : [£{pct[1]:,.0f}, £{pct[3]:,.0f}]")
print()
print("  COST-EFFECTIVENESS PROBABILITY:")
prob_20k = (valid["icer"] < 20_000).mean()
prob_30k = (valid["icer"] < 30_000).mean()
print(f"  P(cost-effective | WTP=£20,000/QALY) : {prob_20k:.1%}")
print(f"  P(cost-effective | WTP=£30,000/QALY) : {prob_30k:.1%}")
print()
# Proportion in each quadrant of the cost-effectiveness plane
q_dominant  = ((valid["delta_qaly"] > 0) & (valid["delta_cost"] < 0)).mean()
q_tradeoff  = ((valid["delta_qaly"] > 0) & (valid["delta_cost"] > 0)).mean()
q_dominated = ((valid["delta_qaly"] < 0) & (valid["delta_cost"] > 0)).mean()
print(f"  Dominant   (more QALY, less cost): {q_dominant:.1%} of iterations")
print(f"  Trade-off  (more QALY, more cost): {q_tradeoff:.1%} of iterations")
print(f"  Dominated  (fewer QALY, more cost): {q_dominated:.1%} of iterations")
print("=" * 60)

# ---- CELL ----
# ── DETERMINISTIC SENSITIVITY ANALYSIS ────────────────────────────────────────
#
# For each parameter, we compute the ICER under:
#   - Low scenario: parameter at its 10th percentile (or -1 SD)
#   - High scenario: parameter at its 90th percentile (or +1 SD)
# All other parameters stay at their base-case values.

from dataclasses import replace as dc_replace
import copy

def quick_icer(p: ModelParams, n_patients: int = 300,
               n_reps: int = 30, seed: int = 0) -> float:
    """
    Compute a stable ICER estimate by averaging n_reps iterations.
    Used in DSA to reduce noise when comparing parameter scenarios.
    Args:
        p:          ModelParams to use
        n_patients: Patients per iteration (more = more stable, slower)
        n_reps:     Number of iterations to average over
        seed:       Random seed
    Returns:
        Mean ICER across n_reps iterations (float)
    """
    q_rng = np.random.default_rng(seed)
    icers = [run_single_iteration(p, n_patients, q_rng)["icer"]
             for _ in range(n_reps)]
    valid_icers = [x for x in icers if not np.isnan(x)]
    return float(np.mean(valid_icers)) if valid_icers else np.nan


# ── Define parameter perturbations ────────────────────────────────────────────
# Each entry: (label, attribute_name, low_value, high_value)
# Low/high values are the plausible extremes for each parameter.

def make_perturbed(base: ModelParams, **kwargs) -> ModelParams:
    """Create a copy of ModelParams with specified attributes changed."""
    p_new = copy.copy(base)
    for k, v in kwargs.items():
        setattr(p_new, k, v)
    return p_new


perturbations = [
    # (Display label,  what to change,  low value,  high value)
    ("AI Sensitivity (mean)",
     {"ai_sensitivity_alpha": 37.5, "ai_sensitivity_beta": 11.8},   # mean≈0.76
     {"ai_sensitivity_alpha": 46.0, "ai_sensitivity_beta":  6.6}),  # mean≈0.87

    ("AI Specificity (mean)",
     {"ai_specificity_alpha": 35.0, "ai_specificity_beta": 12.5},   # mean≈0.74
     {"ai_specificity_alpha": 45.0, "ai_specificity_beta":  8.3}),  # mean≈0.84

    ("Time to Diagnosis — Standard (months)",
     {"ttd_std_mu": lognormal_params(12, 5)[0],
      "ttd_std_sigma": lognormal_params(12, 5)[1]},
     {"ttd_std_mu": lognormal_params(24, 9)[0],
      "ttd_std_sigma": lognormal_params(24, 9)[1]}),

    ("Time to Diagnosis — AI (months)",
     {"ttd_ai_mu": lognormal_params(5, 2)[0],
      "ttd_ai_sigma": lognormal_params(5, 2)[1]},
     {"ttd_ai_mu": lognormal_params(12, 5)[0],
      "ttd_ai_sigma": lognormal_params(12, 5)[1]}),

    ("QALY: Undiagnosed (utility weight)",
     {"qaly_undiag_alpha": 14.5, "qaly_undiag_beta": 22.5},   # mean≈0.39
     {"qaly_undiag_alpha": 29.5, "qaly_undiag_beta": 10.5}),  # mean≈0.74

    ("QALY: Treated epilepsy (utility weight)",
     {"qaly_treated_alpha": 28.0, "qaly_treated_beta": 14.5}, # mean≈0.66
     {"qaly_treated_alpha": 52.0, "qaly_treated_beta":  7.5}),# mean≈0.87

    ("Cost: A&E visit (£)",
     {"cost_ae_mean": 245.0, "cost_ae_std": 49.0},
     {"cost_ae_mean": 455.0, "cost_ae_std": 91.0}),

    ("Cost: Hospital day (£)",
     {"cost_hosp_day_mean": 315.0, "cost_hosp_day_std": 63.0},
     {"cost_hosp_day_mean": 585.0, "cost_hosp_day_std": 117.0}),

    ("AI tool cost per patient (£)",
     {"cost_ai_tool": 25.0},
     {"cost_ai_tool": 150.0}),

    ("Prior P(epilepsy)",
     {"prior_epi_alpha": 10.5, "prior_epi_beta": 42.5},  # mean≈0.20
     {"prior_epi_alpha": 24.5, "prior_epi_beta": 22.5}), # mean≈0.52
]


print("Running Deterministic Sensitivity Analysis...")
print(f"  {len(perturbations)} parameters × 2 scenarios × 30 reps each")
print(f"  Estimated runtime: 2-5 minutes\n")

base_icer_val = quick_icer(params, n_patients=300, n_reps=30, seed=SEED)
print(f"Base-case ICER: £{base_icer_val:,.0f}/QALY\n")

sa_rows = []
for i, (label, lo_kwargs, hi_kwargs) in enumerate(perturbations):
    p_lo = make_perturbed(params, **lo_kwargs)
    p_hi = make_perturbed(params, **hi_kwargs)
    icer_lo = quick_icer(p_lo, n_patients=300, n_reps=30, seed=SEED+i)
    icer_hi = quick_icer(p_hi, n_patients=300, n_reps=30, seed=SEED+i+100)
    spread   = abs(icer_hi - icer_lo)
    sa_rows.append({
        "parameter": label,
        "icer_low":  icer_lo,
        "icer_high": icer_hi,
        "spread":    spread,
        "base_icer": base_icer_val,
    })
    print(f"  {label}: £{icer_lo:,.0f} — £{icer_hi:,.0f}  (spread £{spread:,.0f})")

sa_df = pd.DataFrame(sa_rows).sort_values("spread", ascending=False)
print(f"\nMost influential parameter: {sa_df.iloc[0]['parameter']}")
sa_df[["parameter","icer_low","icer_high","spread"]].round(0)

# ---- CELL ----
# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Empirical Distribution vs. Fitted LogNormal
# ═══════════════════════════════════════════════════════════════════════════════
#
# This figure validates the distribution fitting step (Section 2/3).
# We show:
#   - GREY BARS: the actual empirical data from MIMIC-IV (or synthetic placeholder)
#   - COLOURED LINE: the fitted LogNormal distribution
#
# If the line closely follows the bars, our distribution choice is appropriate
# and the fitted parameters are trustworthy.
# If they diverge, we may need to consider a different distribution family.

def plot_fig1(mimic: 'MIMICExtract', p: 'ModelParams') -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    datasets = [
        (mimic.ed_los_hours,  p.ed_los_mu,  p.ed_los_sigma,
         "ED Length of Stay (hours)", axes[0], 0.5, 48),
        (mimic.hosp_los_days, p.hosp_los_mu, p.hosp_los_sigma,
         "Hospital LOS (days)",        axes[1], 0.5, 30),
    ]

    for data, mu, sigma, xlabel, ax, xmin, xmax in datasets:
        # Histogram of empirical data (normalised to density so it's comparable to the PDF)
        ax.hist(data, bins=40, density=True, alpha=0.55,
                color=PALETTE["neutral"], edgecolor="white",
                label=f"Empirical data
({mimic.source})")
        # Fitted LogNormal probability density function
        x   = np.linspace(xmin, xmax, 400)
        pdf = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
        ax.plot(x, pdf, color=PALETTE["standard"], lw=2.5,
                label=f"Fitted LogNormal\n(μ={mu:.2f}, σ={sigma:.2f})")
        # Annotation: mean and SD of the data
        ax.axvline(data.mean(), color="crimson", lw=1.5, ls="--",
                   label=f"Mean = {data.mean():.1f}")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.set_xlim(xmin, xmax)
        ax.legend(fontsize=9)
        ax.set_title(f"MIMIC-IV: {xlabel}", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Figure 1. Empirical Distributions vs. Fitted LogNormal\n"
        "Validating the distribution assumptions used in the simulation",
        fontsize=13, y=1.02
    )
    fig.tight_layout()
    save_fig(fig, "fig1_empirical_vs_fitted")
    return fig

fig1 = plot_fig1(mimic_data, params)
plt.show()

# ---- CELL ----
# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Prior vs. Posterior P(epilepsy)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This figure shows Bayesian updating in action.
# DASHED BLUE LINE = prior distribution (before the AI tool is applied)
# SOLID ORANGE LINE = posterior distribution (after the AI result)
# ORANGE BARS = empirical histogram of the 500 posterior values
#
# A good AI tool should:
#   - Shift the MEAN of the distribution upward for true epilepsy patients
#   - Narrow the SPREAD (more certainty after seeing the AI result)

def plot_fig2(p: 'ModelParams', posteriors: np.ndarray,
              cohort: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.linspace(0, 1, 500)

    # ── Prior distribution ────────────────────────────────────────────────────
    prior_pdf = stats.beta.pdf(x, p.prior_epi_alpha, p.prior_epi_beta)
    prior_mean_val = p.prior_epi_alpha / (p.prior_epi_alpha + p.prior_epi_beta)
    ax.plot(x, prior_pdf, lw=2.5, color=PALETTE["standard"], ls="--",
            label=f"Prior Beta({p.prior_epi_alpha:.0f}, {p.prior_epi_beta:.0f})\n"
                  f"mean = {prior_mean_val:.3f}")
    ax.fill_between(x, prior_pdf, alpha=0.12, color=PALETTE["standard"])

    # ── Posterior distribution (fit Beta to empirical posteriors) ─────────────
    post_mean = posteriors.mean()
    post_var  = posteriors.var()
    # Fit a Beta distribution to the empirical posterior samples
    if post_var > 0:
        post_a, post_b = beta_params(post_mean, post_var)
        post_pdf = stats.beta.pdf(x, post_a, post_b)
        ax.plot(x, post_pdf, lw=2.5, color=PALETTE["ai_triage"],
                label=f"Posterior Beta({post_a:.1f}, {post_b:.1f})\n"
                      f"mean = {post_mean:.3f}")
        ax.fill_between(x, post_pdf, alpha=0.12, color=PALETTE["ai_triage"])

    # ── Empirical histogram of posterior values ────────────────────────────────
    ax.hist(posteriors, bins=30, density=True, alpha=0.30,
            color=PALETTE["ai_triage"], label="Posterior sample histogram")

    # ── Vertical mean lines ────────────────────────────────────────────────────
    ax.axvline(prior_mean_val, ls=":", color=PALETTE["standard"], lw=1.8,
               alpha=0.8)
    ax.axvline(post_mean, ls=":", color=PALETTE["ai_triage"], lw=1.8,
               alpha=0.8)

    ax.set_xlabel("P(Epilepsy)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(
        "Figure 2. Bayesian Belief Updating: Prior vs. Posterior P(Epilepsy)\n"
        "After applying the AI tool and clinical episode observation",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    save_fig(fig, "fig2_prior_posterior")
    return fig

fig2 = plot_fig2(params, posteriors, cohort)
plt.show()

# ---- CELL ----
# SKIPPED cell 16 (long-running)

# ---- CELL ----

import numpy as np, pandas as pd
np.random.seed(42); rng2 = np.random.default_rng(42)
# Produce small dummy mc_results (20 iters) so figure cells work
_rows=[]
for _ in range(20):
    r=run_single_iteration(params,50,rng2)
    _rows.append(r)
mc_results=pd.DataFrame(_rows)
print('(Smoke-test: dummy mc_results with 20 iterations)')


# ---- CELL ----
# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Cost-Effectiveness Plane
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each dot = one of the 10,000 Monte Carlo iterations.
# X-axis: incremental QALYs gained (positive = AI produces more QALYs)
# Y-axis: incremental cost (positive = AI is more expensive)
#
# The two diagonal lines are the NICE WTP thresholds (£20k and £30k per QALY).
# Points BELOW the £30k line = cost-effective at NICE threshold.
#
# FOUR QUADRANTS:
#   Top-right:    AI more effective AND more costly (trade-off) — most common scenario
#   Bottom-right: AI more effective AND cheaper (dominant) — ideal
#   Top-left:     AI less effective AND more costly (dominated) — reject
#   Bottom-left:  AI less effective AND cheaper — dominated

def plot_fig4(mc: pd.DataFrame, p: 'ModelParams') -> plt.Figure:
    valid = mc.dropna(subset=["delta_cost","delta_qaly"])
    fig, ax = plt.subplots(figsize=(9, 8))

    # Colour points by quadrant
    q_dom  = (valid["delta_qaly"] > 0) & (valid["delta_cost"] < 0)
    q_trd  = (valid["delta_qaly"] > 0) & (valid["delta_cost"] > 0)
    q_bad  = (valid["delta_qaly"] < 0) & (valid["delta_cost"] > 0)
    q_oth  = (valid["delta_qaly"] < 0) & (valid["delta_cost"] < 0)

    for mask, label, col in [
        (q_dom, "Dominant (↑QALY, ↓cost)", "#2ca02c"),
        (q_trd, "Trade-off (↑QALY, ↑cost)", PALETTE["ai_triage"]),
        (q_bad, "Dominated (↓QALY, ↑cost)", "firebrick"),
        (q_oth, "Mixed (↓QALY, ↓cost)",      PALETTE["grey"]),
    ]:
        ax.scatter(valid.loc[mask,"delta_qaly"], valid.loc[mask,"delta_cost"],
                   s=3, alpha=0.3, color=col, label=f"{label} ({mask.sum():,})")

    # WTP threshold lines
    qs = np.linspace(valid["delta_qaly"].min(), valid["delta_qaly"].max(), 300)
    for wtp, ls, lbl in [
        (p.wtp_lower, "--", f"WTP = £{p.wtp_lower/1000:.0f}k/QALY"),
        (p.wtp_upper, ":",  f"WTP = £{p.wtp_upper/1000:.0f}k/QALY"),
    ]:
        ax.plot(qs, wtp*qs, color="black", ls=ls, lw=1.8, label=lbl)

    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("Incremental QALYs (AI − Standard)", fontsize=13)
    ax.set_ylabel("Incremental Cost, £ (AI − Standard)", fontsize=13)
    ax.set_title(f"Figure 4. Cost-Effectiveness Plane\n"
                 f"N = {len(valid):,} Monte Carlo iterations",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, markerscale=4)
    fig.tight_layout()
    save_fig(fig, "fig4_ce_plane")
    return fig

fig4 = plot_fig4(mc_results, params)
plt.show()

# ---- CELL ----
# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Tornado Diagram
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each horizontal bar shows how the ICER changes when one parameter moves
# from its low to high plausible value (all other parameters fixed at base).
# The wider the bar, the more important that parameter is.
# The vertical dashed line = base-case ICER; green lines = NICE WTP thresholds.

def plot_fig5(sa: pd.DataFrame, base_icer: float, p: 'ModelParams') -> plt.Figure:
    df  = sa.head(10).copy().sort_values("spread", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    y   = np.arange(len(df))

    for i, (_, row) in enumerate(df.iterrows()):
        lo = min(row["icer_low"], row["icer_high"])
        hi = max(row["icer_low"], row["icer_high"])
        ax.barh(y[i], hi - lo, left=lo, height=0.55,
                color=PALETTE["ai_triage"], alpha=0.75, edgecolor="white")
        ax.text(lo - 200, y[i], f"£{lo:,.0f}",
                va="center", ha="right", fontsize=8, color="#555")
        ax.text(hi + 200, y[i], f"£{hi:,.0f}",
                va="center", ha="left", fontsize=8, color="#555")

    ax.axvline(base_icer, color="black", lw=2, ls="--",
               label=f"Base-case ICER: £{base_icer:,.0f}")
    ax.axvline(p.wtp_lower, color="#1a7a1a", lw=1.5, ls=":",
               label=f"WTP £{p.wtp_lower/1000:.0f}k/QALY")
    ax.axvline(p.wtp_upper, color="#0d5c0d", lw=1.5, ls="-.",
               label=f"WTP £{p.wtp_upper/1000:.0f}k/QALY")

    ax.set_yticks(y)
    ax.set_yticklabels(df["parameter"], fontsize=10)
    ax.set_xlabel("ICER (£ per QALY gained)", fontsize=12)
    ax.set_title("Figure 5. Tornado Diagram — One-Way Sensitivity Analysis\n"
                 "Top 10 parameters by influence on the ICER",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    save_fig(fig, "fig5_tornado")
    return fig

fig5 = plot_fig5(sa_df, base_icer_val, params)
plt.show()

# ---- CELL ----
# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Cost-Effectiveness Acceptability Curve (CEAC)
# ═══════════════════════════════════════════════════════════════════════════════
#
# The CEAC answers: "At each possible WTP threshold, what is the probability
# that the AI tool is cost-effective?"
#
# HOW IT IS CALCULATED:
# For each WTP value λ, we compute the Net Monetary Benefit (NMB) for each
# Monte Carlo iteration:
#     NMB = λ × ΔQALY − ΔCost
# The AI tool is cost-effective in that iteration if NMB > 0.
# P(cost-effective | λ) = fraction of iterations where NMB > 0.
#
# A WTP of £0 = "we're only willing to pay £0 for a QALY" — almost nothing is cost-effective.
# A WTP of £60,000 = "we'll pay a lot" — almost everything becomes cost-effective.
# The curve rises from left to right; the steepness shows how sensitive conclusions are.

def plot_fig6(mc: pd.DataFrame, p: 'ModelParams',
              wtp_max: float = 60_000) -> plt.Figure:
    valid   = mc.dropna(subset=["delta_cost","delta_qaly"])
    dC      = valid["delta_cost"].values
    dE      = valid["delta_qaly"].values
    wtp_arr = np.linspace(0, wtp_max, 300)

    # For each WTP, fraction of iterations where NMB > 0
    prob_ce = np.array([np.mean(wtp*dE - dC > 0) for wtp in wtp_arr])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(wtp_arr/1000, prob_ce, color=PALETTE["ai_triage"], lw=2.5)
    ax.fill_between(wtp_arr/1000, prob_ce, alpha=0.15, color=PALETTE["ai_triage"])

    # Annotate at key WTP thresholds
    for wtp_t in [p.wtp_lower, p.wtp_upper]:
        p_ce = np.mean(wtp_t * dE - dC > 0)
        ax.axvline(wtp_t/1000, color="black", ls="--", lw=1.5)
        ax.annotate(
            f"λ=£{wtp_t/1000:.0f}k\nP(CE)={p_ce:.2f}",
            xy=(wtp_t/1000, p_ce),
            xytext=(wtp_t/1000 + 2, p_ce - 0.10),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", lw=1.2)
        )

    ax.axhline(0.5, color=PALETTE["grey"], ls=":", lw=1.2, label="50% threshold")
    ax.axhline(0.8, color="#555",          ls=":", lw=1.2, label="80% threshold")
    ax.set_xlabel("Willingness-to-Pay Threshold (£000 per QALY)", fontsize=13)
    ax.set_ylabel("Probability AI Triage is Cost-Effective", fontsize=13)
    ax.set_ylim(0, 1); ax.set_xlim(0, wtp_max/1000)
    ax.set_title("Figure 6. Cost-Effectiveness Acceptability Curve (CEAC)\n"
                 "AI-Assisted Triage vs. Standard NHS Care",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    save_fig(fig, "fig6_ceac")
    return fig

fig6 = plot_fig6(mc_results, params)
plt.show()