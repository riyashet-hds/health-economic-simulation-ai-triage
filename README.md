# Health Economic Impact Estimation for AI-Assisted Diagnostic Triage: A Simulation Framework Using MIMIC-IV-Derived Parameters

A Monte Carlo health economic simulation evaluating the cost-effectiveness of an AI-assisted diagnostic triage tool for epilepsy, compared to standard NHS care pathways.

## Overview

This project implements a probabilistic cost-effectiveness analysis (CEA) using a decision-tree microsimulation framework. It models two clinical pathways:

- **Standard care:** Current NHS diagnostic pathway for patients presenting with seizures
- **AI-assisted triage:** An AI tool that risk-stratifies patients at first presentation, fast-tracking high-risk patients to neurology

The simulation integrates three data sources:
1. **MIMIC-IV** (PhysioNet) -- empirical distributions for ED length-of-stay, hospital length-of-stay, readmission rates, and EEG ordering
2. **Clinical literature** -- diagnostic accuracy parameters, time-to-diagnosis, QALY utility weights
3. **NHS Reference Costs (2023/24)** -- unit costs for A&E visits, neurology consultations, EEG, and inpatient days

## Key Results

Based on 10,000 Monte Carlo iterations using real MIMIC-IV data (N=10,050 seizure admissions):

| Metric | Value |
|--------|-------|
| Mean incremental cost | -£237 per patient (AI saves money) |
| Mean incremental QALYs | +0.1455 per patient |
| Mean ICER | -£1,743/QALY (**dominant**) |
| P(cost-effective \| WTP=£20k) | 100.0% |
| P(cost-effective \| WTP=£30k) | 100.0% |

The AI intervention **dominates** standard care in 100% of iterations -- it is simultaneously cheaper and more effective.

## Repository Structure

```
health-economic-simulation-ai-triage/
├── health_economic_simulation_ai_triage.ipynb  # Main simulation notebook
├── data/
│   ├── mimic_extracted/              # MIMIC-IV CSVs (not included -- see instructions)
│   │   └── DATA_README.md
│   ├── literature/                   # Reference PDFs for parameter calibration
│   └── nhs_reference/               # NHS Reference Cost data (not included -- too large)
│       └── DATA_README.md
├── _compile_check.py                 # Syntax checker for notebook code cells
├── _smoke_test.py                    # Standalone smoke test of core simulation logic
├── _run_test.py                      # Functional test runner (executes notebook cells)
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook or Google Colab

### Installation

```bash
git clone https://github.com/riyashet-hds/health-economic-simulation-ai-triage.git
cd health-economic-simulation-ai-triage
pip install -r requirements.txt
```

### Running the Simulation

**Option 1: Google Colab (recommended)**
- Upload `health_economic_simulation_ai_triage.ipynb` to Google Colab
- Place MIMIC-IV CSVs in `data/mimic_extracted/` (or let the notebook use synthetic fallback data)
- Run all cells

**Option 2: Local Jupyter**
```bash
jupyter notebook health_economic_simulation_ai_triage.ipynb
```

### Data Setup

The notebook auto-detects data availability:

- **With MIMIC-IV data:** Place exported CSVs (`admissions.csv`, `diagnoses_icd.csv`, `edstays.csv`, `patients.csv`, `procedures_icd.csv`) in `data/mimic_extracted/`. See [data/mimic_extracted/DATA_README.md](data/mimic_extracted/DATA_README.md) for extraction instructions.
- **Without MIMIC-IV data:** The notebook automatically generates synthetic data calibrated from published literature values. The simulation runs identically in either case.

## Notebook Sections

1. **MIMIC-IV Data Loading** -- Load real or synthetic patient data
2. **Distribution Fitting** -- Fit lognormal distributions to ED/hospital LOS
3. **Clinical Parameters** -- Define costs, utilities, diagnostic accuracy
4. **Standard Care Pathway** -- Microsimulation of current NHS pathway
5. **AI-Assisted Pathway** -- Microsimulation with Bayesian belief-updating
6. **Bayesian Triage Module** -- AI tool sensitivity/specificity and belief updating
7. **Monte Carlo CEA** -- 10,000-iteration probabilistic cost-effectiveness analysis
8. **Sensitivity Analysis** -- Deterministic one-way sensitivity analysis with tornado diagram
9. **Discussion** -- Interpretation of results, limitations, and next steps

## Figures Generated

- **Figure 1:** Distribution fits (ED LOS, Hospital LOS) -- empirical vs. fitted lognormal
- **Figure 2:** Cost-effectiveness plane (incremental cost vs. incremental QALYs)
- **Figure 3:** Cost-effectiveness acceptability curve (CEAC)
- **Figure 4:** ICER distribution histogram
- **Figure 5:** Tornado diagram (one-way sensitivity analysis)

## Testing

```bash
# Check notebook syntax (no execution)
python _compile_check.py

# Run functional smoke test
python _run_test.py
```

## Key References

- NICE NG217 (2022). *Epilepsies in children, young people and adults*
- Mulhern et al. *Development of a QALY measure for epilepsy: NEWQOL-6D*
- Mulhern et al. *Valuations of epilepsy-specific health states*
- Tittensor et al. *Updating beliefs using a diagnostic decision-support aid in a nurse-led first-seizure clinic*
- Lopes et al. (2019). *Revealing epilepsy type using computational analysis of interictal EEG*
- Schmidt et al. (2016). *A computational biomarker of idiopathic generalised epilepsy from resting state EEG*
- Bonnon et al. *Comprehensive cost-effectiveness analysis of resective epilepsy surgery in an NHS setting*
- *Incidence and prevalence of epilepsy in the United Kingdom 2013-2018*
- *Impact of diagnostic delay on seizure outcome in newly diagnosed focal epilepsy* (2020)

See [data/literature/](data/literature/) for full PDFs.

## License

This project is licensed under the MIT License -- see [LICENSE](LICENSE) for details.

## Acknowledgements

- **MIMIC-IV** database via [PhysioNet](https://physionet.org/content/mimiciv/)
- **NHS National Cost Collection** 2023/24
