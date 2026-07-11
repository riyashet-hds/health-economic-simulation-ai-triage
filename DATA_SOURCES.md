# Data Sources

This document records the provenance, access requirements, and usage rationale for every data source feeding into the simulation.

## Source A: MIMIC-IV (Empirical Parameters)

**What it is:** De-identified electronic health records from >300,000 hospital admissions at Beth Israel Deaconess Medical Center, Boston, USA (2008–2019). Maintained by the MIT Laboratory for Computational Physiology.

**Access:** Requires credentialed access via [PhysioNet](https://physionet.org/content/mimiciv/). Researchers must complete CITI training and sign a data use agreement. No MIMIC-IV data is stored in this repository — only aggregated distributional parameters derived from the data.

**Tables used:**

| MIMIC-IV Table | Parameters Extracted |
|---|---|
| `mimic_hosp.admissions` + `mimic_ed.edstays` | ED length-of-stay distributions, admission patterns |
| `mimic_hosp.diagnoses_icd` | Diagnostic code co-occurrence, misdiagnosis rates for seizure/epilepsy cohort |
| `mimic_hosp.procedures_icd` | EEG and neuroimaging ordering rates, procedure timing |
| `mimic_hosp.admissions` | Hospital LOS distributions, 30-day readmission rates |
| `mimic_hosp.patients` + `admissions` | Age/sex demographics for seizure presentations |

**Cohort definition:** Admissions with ICD-10 codes G40.x (epilepsy) and R56.x (seizures), filtered from the full MIMIC-IV population.

**Why a US database for a UK model?** MIMIC-IV provides *distributional shapes* — how values like hospital length-of-stay are spread across patients. These clinical process patterns are informative across healthcare systems. UK-specific monetary values are applied separately via NHS Reference Costs. This hybrid approach (international clinical data + local cost data) is standard methodology in published health economic modelling.

**Citation:**
> Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67

---

## Source B: Published Clinical Literature (Epilepsy-Specific Parameters)

Some parameters are disease-specific and not captured in hospital-level records. These come from peer-reviewed literature and clinical guidelines.

| Parameter | Value | Source |
|---|---|---|
| UK epilepsy prevalence | ~633,000 people | Mbizvo et al., *Seizure* (2023) |
| Incidence | ~51 per 100,000/year | Mbizvo et al., *Seizure* (2023); NICE NG217 |
| Time to diagnosis (standard care) | Mean ~18 months, range 7–60+ months | Bonnett et al., *Epilepsia* (2020); Epilepsy Action (2023) |
| AI diagnostic sensitivity/specificity | Modelled as Beta distributions (see notebook §3) | Published EEG-AI classifier studies |
| QALY utility weights | Epilepsy health states | NICE technology appraisals |
| NICE WTP threshold | £20,000–£30,000 per QALY | NICE Methods Guide |

**Bayesian updating framework** is informed by:
> Terry, J. R. et al. (2024). "Updating Beliefs using New Evidence from a Diagnostic Decision-Support Aid in a Nurse-Led First-Seizure Clinic." Preprint, December 2024.

---

## Source C: NHS Reference Costs (UK-Specific Unit Costs)

Monetary values are drawn from NHS cost publications to ensure relevance to UK health economic decision-making.

| Cost Item | Approximate Value | Source |
|---|---|---|
| A&E attendance | ~£170–£220 | NHS National Cost Collection (latest available year) |
| Neurology outpatient | ~£180–£250 | NHS National Cost Collection |
| EEG procedure | ~£150–£200 | NHS National Cost Collection |
| Hospital bed-day | ~£350–£500 | NHS National Cost Collection |

**Note:** Costs are modelled as Gamma distributions to capture uncertainty, not as fixed point estimates. Distributional parameters are documented in the notebook (§3).

**Publication:**
> NHS England. National Cost Collection. Published annually. https://www.england.nhs.uk/costing-in-the-nhs/national-cost-collection/

---

## Data Directory Structure

```
data/
├── mimic_extracted/     # Aggregated MIMIC-IV parameters (CSVs, no PHI)
│                        # NOT included in repo — requires credentialed access
├── nhs_reference/       # NHS cost data (publicly available)
└── literature/          # Parameter source references and extraction notes
```

## Privacy and Ethics

- No individual patient data is stored in or distributed with this repository.
- MIMIC-IV access complies with the PhysioNet Credentialed Health Data Use Agreement.
- All simulation patients are entirely synthetic, generated from fitted probability distributions.
- Aggregated parameters (means, standard deviations, rates) cannot be used to re-identify individuals.
