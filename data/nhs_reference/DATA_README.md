# NHS Reference Cost Data

This directory should contain NHS National Cost Collection data used for unit cost parameters.

## Data Source

**NHS England 2023/24 National Cost Collection Data Publication**

The following files are required but not included in the repository due to their size (>100MB each):

| File | Size | Description |
|------|------|-------------|
| `Organisation_level_source_data_1_2324 v2(mff_adjusted).csv` | ~101 MB | MFF-adjusted organisation-level costs |
| `Organisation_level_source_data_2_2324 v2.csv` | ~99 MB | Organisation-level source data (part 2) |
| `Organisation_level_source_data_3_2324 v2(reference_tables)/` | ~15 files | XLSX reference tables (currencies, departments, services) |

## How to Obtain

1. Visit the [NHS England National Cost Collection](https://www.england.nhs.uk/costing-in-the-nhs/national-cost-collection/) page
2. Download the 2023/24 organisation-level source data files
3. Place them in this directory

## Note

The notebook uses specific unit costs extracted from these files (A&E attendance, neurology outpatient, EEG, inpatient bed-day). If the raw NHS data is not available, the notebook uses literature-derived cost estimates as defaults.
