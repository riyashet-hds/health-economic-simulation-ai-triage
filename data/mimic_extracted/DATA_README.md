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
3. **Run extraction queries:** Filter for seizure/epilepsy-related admissions using ICD codes G40.x (epilepsy), G41.x (status epilepticus), and R56.x (convulsions)
4. **Export as CSV:** Download query results as CSV files and place them in this directory

## Extraction Queries

The exact BigQuery SQL used to build the cohort and produce these CSVs is committed
alongside this file in [`extraction_queries.sql`](extraction_queries.sql) (dataset:
`physionet-data.mimiciv_3_1_hosp` and `physionet-data.mimiciv_ed`). It contains:

- **Part A -- Cohort export queries (5):** the actual queries saved in the BigQuery
  console, one per CSV above. All five share a cohort seed -- subjects with at least
  one `G40%`/`G41%`/`R56%` diagnosis -- then export the full record set for those
  subjects (ED stays are restricted to the seizure admissions).
- **Part B -- Analysis queries (3):** the ED/hospital length-of-stay, EEG-ordering-rate,
  and 30-day-readmission queries that define the parameters the notebook extracts.

Query-to-file mapping:

| Query | Output file | Grain |
|-------|-------------|-------|
| A1 | `patients.csv` | one row per cohort subject |
| A2 | `admissions.csv` | all admissions of cohort subjects |
| A3 | `diagnoses_icd.csv` | all diagnoses (ICD-9 and ICD-10) for cohort subjects |
| A4 | `procedures_icd.csv` | all procedures for cohort subjects |
| A5 | `edstays.csv` | ED stays for the seizure admissions |

The seizure-diagnosis filter identifies **10,050 distinct seizure admissions**
(the `N` reported in the top-level README). The subject-level export deliberately keeps
each cohort subject's complete admission history so the 30-day readmission rate can be
computed across all of their encounters, not only the seizure ones.

## ICD Code Filter

The notebook filters for the following ICD-10 code prefixes:
- `G40` -- Epilepsy and recurrent seizures
- `G41` -- Status epilepticus
- `R56` -- Convulsions, not elsewhere classified

## Without MIMIC-IV Data

If these files are not present, the notebook automatically generates synthetic data calibrated from published literature values. The simulation runs identically in either case.
