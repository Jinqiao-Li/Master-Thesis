# Thesis: Work task classification from job ads onto O*NET: hierarchy-aware semantics and cross-lingual transfer approach

Jinqiao Li 

Supervise: Ann-Sophie Gnehm, Simon Clematide

Professor: Martin Volk

## Files Description

### Data

- `data_process.ipynb`
Merged all the  relevant data. extened mappings relationships from 'task to DWA' to IWA and GWA. Also merged the English version and German version (SJMM tranlated).

  Output file: task_to_GWA_IWA_DWA_DE.csv

- `stat.ipynb`
Some statistical analysis on the data used for later training and test.

- `DeepL.ipynb`
Translated original English data to German with DeeL API (formal).

  Input file: onet_occ_tasks_workactivities_en.jsonl

  Output file: onet_occ_tasks_workactivities_en2de_deepl.jsonl

### Models
- `baseline.ipynb`
- `model_measurement.ipynb`

