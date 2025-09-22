# Supporting Information for MASLD Flavonoid Screen

This archive contains datasets, configuration files, and scripts required to reproduce the results of the manuscript.

## Contents

### Raw curated activity tables
- metabolites_<target>.csv: curated IC values from ChEMBL, cleaned and binarized (active/inactive).

### Prediction outputs
- Ranked hits, candidate tables, similarity/gap analysis, and product portfolios.

### Summary metrics
- Model performance (ROC-AUC, F1, MCC) and pipeline summaries.

### Configuration and provenance
- All parameters (YAML/JSON), clustering thresholds, concordance scores, and software versions.

### Scripts
- postprocess_hits.py: generates the reported tables and figures from prediction outputs.

## Data Sources
Raw activity records can be retrieved directly from the ChEMBL API (https://www.ebi.ac.uk/chembl).  

## Reproduction
Example command:
`ash
python Scripts/postprocess_hits.py --config Configs/config.yaml --predictions Datasets/advanced_predictions.csv
`

## Contact

Additional details are available from the corresponding author upon request.

## Data Citation
- Version: v1.0.0
- DOI: TBD
- Recommended citation:
  - Supporting Information for MASLD Flavonoid Screen, v1.0.0 (2025). DOI: TBD.
