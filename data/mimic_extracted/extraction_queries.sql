-- =============================================================================
-- MIMIC-IV Cohort Extraction Queries  (Google BigQuery Standard SQL)
-- Project: Health Economic Simulation for AI-Assisted Diagnostic Triage (Epilepsy)
-- =============================================================================
--
-- WHAT THIS FILE IS
--   The SQL used to extract the seizure / epilepsy cohort from MIMIC-IV on
--   PhysioNet's Google BigQuery, producing the five CSV files in this directory
--   (admissions.csv, diagnoses_icd.csv, edstays.csv, patients.csv,
--   procedures_icd.csv) that the notebook loads.
--
--   * Part A -- Cohort export queries (5): one per exported CSV. These are the
--     actual queries saved in the PhysioNet BigQuery console (Feb 2026), copied
--     here verbatim.
--   * Part B -- Analysis queries (3): the parameter-extraction queries as
--     documented in the notebook. In the shipped notebook the same logic is
--     reproduced in pandas over the CSVs above, so the pipeline runs without
--     live BigQuery access.
--
-- DATASET / ACCESS
--   Dataset : physionet-data.mimiciv_3_1_hosp.*  and  physionet-data.mimiciv_ed.*
--             (MIMIC-IV v3.1, hospital module; ED module for edstays)
--   Access  : Requires a credentialed PhysioNet account -- completed CITI
--             "Data or Specimens Only Research" training and a signed MIMIC-IV
--             Data Use Agreement -- linked to Google Cloud BigQuery.
--
-- COHORT DEFINITION
--   Seizure / epilepsy patients identified by ICD diagnosis codes:
--     G40.x  Epilepsy and recurrent seizures
--     G41.x  Status epilepticus
--     R56.x  Convulsions, not elsewhere classified
--   Cohort subjects = patients with >= 1 diagnosis in the above families. The
--   export then pulls the FULL record set (all admissions, diagnoses, and
--   procedures) for those subjects, so complete hospital history is available
--   for the 30-day readmission calculation; ED stays (A5) are restricted to the
--   seizure admissions themselves.
--
--   The seizure-diagnosis filter yields 10,050 distinct seizure admissions,
--   the N reported in the repository README. Absolute row counts depend on the
--   MIMIC-IV v3.1 snapshot at extraction time.
--
-- REPRODUCE
--   Run Part A in the PhysioNet BigQuery console and export each result set as
--   CSV into this directory. The notebook auto-detects the CSVs; if absent it
--   falls back to literature-calibrated synthetic data and runs identically.
-- =============================================================================


-- #############################################################################
-- PART A -- COHORT EXPORT QUERIES  (produce the CSVs in this directory)
-- #############################################################################

-- -----------------------------------------------------------------------------
-- A1 -> patients.csv
-- Demographics for the seizure cohort
-- Columns needed: subject_id, gender, anchor_age
-- -----------------------------------------------------------------------------
SELECT DISTINCT
    p.subject_id,
    p.gender,
    p.anchor_age
FROM `physionet-data.mimiciv_3_1_hosp.patients` p
WHERE p.subject_id IN (
    SELECT DISTINCT d.subject_id
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
    WHERE d.icd_code LIKE 'G40%'
       OR d.icd_code LIKE 'G41%'
       OR d.icd_code LIKE 'R56%'
);


-- -----------------------------------------------------------------------------
-- A2 -> admissions.csv
-- All admissions for patients who EVER had a seizure/epilepsy diagnosis
-- Columns needed: hadm_id, subject_id, admittime, dischtime
-- -----------------------------------------------------------------------------
SELECT
    a.hadm_id,
    a.subject_id,
    a.admittime,
    a.dischtime
FROM `physionet-data.mimiciv_3_1_hosp.admissions` a
WHERE a.subject_id IN (
    -- Patients with at least one seizure/epilepsy diagnosis
    SELECT DISTINCT d.subject_id
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
    WHERE d.icd_code LIKE 'G40%'
       OR d.icd_code LIKE 'G41%'
       OR d.icd_code LIKE 'R56%'
)
ORDER BY a.subject_id, a.admittime;


-- -----------------------------------------------------------------------------
-- A3 -> diagnoses_icd.csv
-- All diagnosis codes for the seizure cohort's admissions
-- Columns needed: hadm_id, icd_code, icd_version
-- -----------------------------------------------------------------------------
SELECT
    d.hadm_id,
    d.icd_code,
    d.icd_version
FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
WHERE d.subject_id IN (
    SELECT DISTINCT d2.subject_id
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d2
    WHERE d2.icd_code LIKE 'G40%'
       OR d2.icd_code LIKE 'G41%'
       OR d2.icd_code LIKE 'R56%'
)
ORDER BY d.hadm_id, d.seq_num;


-- -----------------------------------------------------------------------------
-- A4 -> procedures_icd.csv
-- All procedures for the seizure cohort's admissions
-- Columns needed: hadm_id, icd_code
-- -----------------------------------------------------------------------------
SELECT
    p.hadm_id,
    p.icd_code
FROM `physionet-data.mimiciv_3_1_hosp.procedures_icd` p
WHERE p.subject_id IN (
    SELECT DISTINCT d.subject_id
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
    WHERE d.icd_code LIKE 'G40%'
       OR d.icd_code LIKE 'G41%'
       OR d.icd_code LIKE 'R56%'
)
ORDER BY p.hadm_id;


-- -----------------------------------------------------------------------------
-- A5 -> edstays.csv
-- ED stays for patients with seizure/epilepsy diagnoses
-- Columns needed: hadm_id, intime, outtime
-- -----------------------------------------------------------------------------
SELECT DISTINCT
    e.hadm_id,
    e.intime,
    e.outtime
FROM `physionet-data.mimiciv_ed.edstays` e
INNER JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
    ON e.hadm_id = d.hadm_id
WHERE (d.icd_code LIKE 'G40%'    -- Epilepsy
    OR d.icd_code LIKE 'G41%'    -- Status epilepticus
    OR d.icd_code LIKE 'R56%')   -- Convulsions/seizures
  AND e.hadm_id IS NOT NULL;


-- #############################################################################
-- PART B -- ANALYSIS QUERIES  (as documented in the notebook)
--
-- These express the parameter-extraction logic in SQL. The shipped notebook
-- reproduces the same logic in pandas over the CSVs above so the pipeline runs
-- without live BigQuery access.
-- #############################################################################

-- Query 1: ED length of stay for seizure/epilepsy patients
-- Feeds: lognormal fit of ED LOS (hours) and hospital LOS (days).
SELECT
    e.stay_id,
    e.hadm_id,
    TIMESTAMP_DIFF(e.outtime, e.intime, HOUR)      AS ed_los_hours,
    TIMESTAMP_DIFF(a.dischtime, a.admittime, DAY)  AS hosp_los_days
FROM `physionet-data.mimiciv_ed.edstays` e
LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
    ON e.hadm_id = a.hadm_id
WHERE e.hadm_id IN (
    SELECT DISTINCT hadm_id
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
    WHERE (icd_version = 10 AND icd_code LIKE 'G40%')  -- G40: Epilepsy
       OR (icd_version = 10 AND icd_code LIKE 'R56%')  -- R56: Convulsions
)
AND e.outtime IS NOT NULL;


-- Query 2: EEG ordering rate
-- Feeds: probability that a seizure admission receives an EEG.
SELECT
    COUNT(DISTINCT CASE WHEN icd_code IN ('8914','9589') THEN hadm_id END)
        AS eeg_count,
    COUNT(DISTINCT hadm_id) AS total_seizure_admissions
FROM `physionet-data.mimiciv_3_1_hosp.procedures_icd`
WHERE hadm_id IN (
    SELECT DISTINCT hadm_id FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
    WHERE icd_code LIKE 'G40%' OR icd_code LIKE 'R56%'
);


-- Query 3: 30-day readmission rate
-- Feeds: baseline probability of readmission within 30 days of discharge.
SELECT
    a1.subject_id,
    COUNT(*) AS readmit_30d_count
FROM `physionet-data.mimiciv_3_1_hosp.admissions` a1
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a2
    ON a1.subject_id = a2.subject_id
    AND a2.admittime > a1.dischtime
    AND TIMESTAMP_DIFF(a2.admittime, a1.dischtime, DAY) <= 30
GROUP BY a1.subject_id;
