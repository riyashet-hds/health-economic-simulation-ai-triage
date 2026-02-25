# MIMIC-IV Data Extraction

This directory should contain CSV files extracted from [MIMIC-IV](https://physionet.org/content/mimiciv/) via PhysioNet BigQuery.

## Required Files

| File | Source Table | Description |
|------|-------------|-------------|
| `admissions.csv` | `mimiciv_hosp.admissions` | Hospital admission records |
| `diagnoses_icd.csv` | `mimiciv_hosp.diagnoses_icd` | ICD diagnosis codes per admission |
| `edstays.csv` | `mimiciv_ed.edstays` | Emergency department stay records |
| `patients.csv` | `mimiciv_hosp.patients` | Patient demographics |
| `procedures_icd.csv` | `mimiciv_hosp.procedures_icd` | ICD procedure codes per admission |

## How to Obtain the Data

1. **Get PhysioNet access:** Complete the required CITI training and sign the data use agreement at [physionet.org](https://physionet.org/content/mimiciv/)
2. **Access BigQuery:** Link your PhysioNet account to Google Cloud and access MIMIC-IV via BigQuery
3. **Run extraction queries:** Filter for seizure/epilepsy-related admissions using ICD codes G40.x (epilepsy) and R56.x (convulsions)
4. **Export as CSV:** Download query results as CSV files and place them in this directory

## ICD Code Filter

The notebook filters for the following ICD-10 code prefixes:
- `G40` -- Epilepsy and recurrent seizures
- `R56` -- Convulsions, not elsewhere classified

## Without MIMIC-IV Data

If these files are not present, the notebook automatically generates synthetic data calibrated from published literature values. The simulation runs identically in either case.
