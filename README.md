# Clinical Case Intake

Machine learning-assisted platform for patient intake, case categorization/routing and clinical workflow support.

This project evolves [PW_Triage_ML](https://github.com/cestdove/PW_Triage_ML), my previous thesis project exploring Support Vector Machines for ticket triage and text classification.

The model takes a short clinical-case description as input and predicts a numerical label representing one of five broad medical-condition categories.

The project uses the public *Medical Abstracts Text Classification Dataset*, available on Hugging Face and created by Tim Schopf, Daniel Braun, and Florian Matthes. The dataset consists of English medical abstracts rather than real patient tickets or clinical records.

The label used in particular are distingued as: 
| Label | Category | Original Train | Original Test |
|---:|---|---:|---:|
| 1 | Neoplasms | 2.530 | 633 |
| 2 | Digestive system diseases | 1.195 | 299 |
| 3 | Nervous system diseases | 1.540 | 385 |
| 4 | Cardiovascular diseases | 2.441 | 610 |
| 5 | General pathological conditions | 3.844 | 961 |


This project introduces more recent concepts as modern Neural Nets, transformers, and new NLP techniques like a BERT model being fine-tuned.  
