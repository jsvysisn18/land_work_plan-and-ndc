# Policy Inputs and Core Spatial Allocation Modules

This repository provides the core policy XML inputs and standalone Python algorithmic modules accompanying the manuscript:

> **Linking land system regulation to climate mitigation: Alignment and divergence between China's spatial planning and the 2035 NDC**

## Repository Structure

### 1. GCAM-China v8.2 Policy XML Inputs (`gcam_policy_xml/`)
Corresponding to **Section 2.2** and **Supporting Information Texts S1–S2**:
- `BAU_NDC2035_CO2_cap_provincial_official_10pct.xml`: Provincial CO₂ emission constraints for 2035.
- `SSP2_HE2024_RELATIVE_AREA.xml`: Baseline-anchored relative forest area floor derived from the Pledged pathway of He et al. (2024) under SSP2.
- `SSP2_HE2024_STRUCTURAL_ZERO_FLOOR.xml`: Matched structural zero forest-floor control configuration.

### 2. Core Spatial Allocation Modules (`spatial_allocation_core/`)
Corresponding to **Sections 2.2 & 2.4** and **Supporting Information Texts S4–S6**:
- `two_stage_rf_model.py`: Two-stage Random Forest transition probability estimator, spatial-block probability calibration selector (prior-odds correction, Platt scaling, isotonic regression), and masked destination renormalization.
- `glu_quota_and_allocation.py`: GCAM-to-LUCC compositional quota reconciliation, largest-remainder integer closure, iterative shadow-price spatial allocation, and minimum-utility-loss residual repair.

## Data and Model Availability

- **Base Models**: The base GCAM-China v8.2 model is publicly available via the official release repository (<https://umd-cgs.github.io/metarepo_gcam-china/index.html>). The FLUS software is publicly available from its original developers.
- **Input Datasets**: Publicly available land-cover and auxiliary datasets (CLCD, WorldClim, WorldPop, HWSD v2.0, SRTM DEM, FAO GLW4, VIIRS, and GDP) can be obtained from their respective sources cited in the manuscript and Supporting Information (Table S1).
- **Intermediate Data & Full Workflow**: Full 1 km raster stacks, provincial planning target tables, and intermediate processing scripts are not redistributed here due to file size and third-party licensing constraints, and are available from the corresponding author upon reasonable request.
