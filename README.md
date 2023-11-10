# Thesis: Work task classification from job ads onto O*NET: hierarchy-aware semantics and cross-lingual transfer approach

Jinqiao Li 

Supervise: Ann-Sophie Gnehm, Simon Clematide

Professor: Martin Volk

## Project Description

This project applied a hierarchy-aware and cross-lingual approach to classify job tasks (e.g.: *Verpackungsarbeiten allgemein und in Medizinaltechnik*) from German job advertisements using the ONET English ontology which is a complex ontology with three hierarchical level and fine-grained classes. Two methods, machine translation and multilingual models, are tested to bridge the language gap. The project consisted of two sets of experiments: local classifier experiments using transformer-based models at each hierarchical level, and global hierarchical models on the O*NET data. This work yields several key findings:

Firstly, domain adaptation proved effective, with job domain-specific language models outperforming general domain models. Translation quality also influenced classification performance, with DeepL outperforming the SJMM engine.

Secondly, state-of-the-art models (TextRNN, TextRCNN, HMCN, HiAGM) were used as global hierarchical models for task classification. These models effectively incorporated hierarchical information, addressing inconsistencies and overfitting through recursive regularization.

Furthermore, the best model configurations from both series of experiments are selected to predict job advertisement data, resulting in reliable classification using the O*NET hierarchical ontology. Human post-evaluation, conducted by a German-speaking domain expert, validates the accuracy of the models' predictions. Overall, while this project extensively tested the feasibility of hierarchy-aware classification models, the transformer-based flat model Job-GBERT proves to be a more suitable option for the hierarchical classification of Job Ads data, given its specificity.



### Data

- `task_train.csv` `task_val.csv` `task_test.csv`
These are the data used for training, validation and test

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

