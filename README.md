# FlowARC
______
This repository provides training and evaluation codes for the paper – FlowARC: Deep Learning Based Automatic Detection of Abnormal B-cells in Acute Lymphoblastic Leukemia/Lymphoma. FlowARC automatically identifies abnormal B-cells of B-lymphoblastic Leukemia (B-ALL) in the initial diagnostic flow cytometry specimens as well as in the follow-up specimens.


##  Prerequisites
_______
##### R programming language 4.3.1
* flowCore (R package v2.14.2)
* flowDensity (R package v1.38.0)
##### Python 3.10.9
* PyTorch 2.01+cu117
* fcsparser 0.2.8
* pandas 1.5.3
* scikit-learn 1.2.1

## Required data
_____
The replication of the results of the FlowARC paper requires access to the original patient flow cytometry data, along with their manual annotations. The access to the PHI removed version of this data will be provided once the paper goes through peer-review. 

The dataset contains raw flow cytometry data files of all normal and abnormal/tumor cases. Additionally, flow cytometry files with just abnormal cells for annotated tumor cases are available. 

## Preprocessing
_______
* **preprocess_and_gate.R** : preprocessing of the raw files for all cases (ungated_all_cases folder in the original_data folder) and their subsequent B cell gating.
* **preprocess_only.R** : preprocessing of the annotated tumor cases

## Cell-level module
__________
* **train_evaluate_cell_level_module.py** : train and evaluate the Cell-level module for 100 epochs using annotated cells. The evaluation of validation cells and the test cells are output at every 5 epochs.
* **save_predictions_annotated_datasets.py** : save the predictions of the Cell-level module for all the annotated cells as temporary pandas-feather files.
* **generate_synthetic_samples.py** : generate and save training, validation and testing synthetic samples using the files containing annotated cells and their Cell-level module predictions.

## Sample-level module
________
* **train_evaluate_sample_level_module.py** : train and save 5 Sample-level modules for 100 epochs using the synthetic samples. The final output is the ROC curve plot for these 5 modules on the synthetic test samples.
* **train_evaluate_quantification_module.py** : train and save 5 quantification modules for 100 epochs using the synthetic samples. The final outputs are the $R^2$, MAE and MSE values on the synthetic test samples along with their 95% CI.
* **results_original_test_sample_level_module.py** : Evaluate the Sample-level module (one with the best AUROC on the synthetic validation samples) on the original patient test cases. The final output is the corresponding confusion matrix.
* **results_original_test_quantification_module.py** : Evaluate the quantification module (one with the best $R^2$  on the synthetic validation samples) on the original patient test cases. The final output is the corresponding predicted Vs true tumor population plot (Figure 4 in the paper).

## Folder structure
______________
* **code** : contains all scripts
* **model** : contains all trained models
* **original_data** : contains input raw and annotated flow cytometry files
* **tmp** : contains the temporary files – the annotated cells with predictions and synthetic samples

## License
________________
This project is under the CC-BY-NC 4.0 license. See LICENSE for details. (c) MSK