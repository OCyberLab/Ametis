
# Detailed README

## Running the Experiment: Best Features and Models Selection

This section provides detailed instructions on how to execute the "Best Features and Models Selection" experiment using the provided code and datasets. Additionally, guidance is provided for adapting the code to new datasets.

### 1. Running the Experiment on IoT-23 and CICIoT2023 Datasets

The repository includes scripts to perform feature selection experiments on the IoT-23 and CICIoT2023 datasets. The process of running these experiments involves selecting the best combinations of features and models, along with necessary preprocessing steps.

#### IoT-23 Dataset
To run the feature selection experiment for the IoT-23 dataset, use the `feature_selection_exp.py` script. Ensure that the IoT-23 dataset is available in the `datasets/IoT23/` folder. This dataset will be split into training and test sets for evaluation.

Command:
```bash
python feature_selection_exp.py
```

#### CICIoT2023 Dataset
To run the feature selection experiment for the CICIoT2023 dataset, use the `feature_selection_exp_CIC.py` script. Ensure that the CICIoT2023 dataset is located in the `datasets/CIC/` folder.

Command:
```bash
python feature_selection_exp_CIC.py
```

These scripts will automatically load the corresponding dataset, perform the necessary preprocessing, and execute the feature selection experiment to evaluate the impact on model performance.

### 2. Running the Experiment for BERT Model

For experiments involving the BERT model, separate scripts are provided, but the process is similar to the above.

#### IoT-23 Dataset with BERT
Use the `Bert_FS.py` script for running the BERT-based feature selection experiment on the IoT-23 dataset.

Command:
```bash
python Bert_FS.py
```

#### CICIoT2023 Dataset with BERT
Use the `Bert_FS_CIC.py` script for running the BERT-based feature selection experiment on the CICIoT2023 dataset.

Command:
```bash
python Bert_FS_CIC.py
```

### 3. Adapting the Framework to a New Dataset

If you want to run this experiment with a new dataset, follow these steps:

1. **Copy the Dataset Structure**: Duplicate the `CIC/` folder and rename it to reflect the name of your new dataset (e.g., `MyNewDataset/`). Ensure that the new folder contains subfolders for training and test datasets.

2. **Modify Preprocessing Functions**: Update the `main_code.py` script within your new dataset's folder to preprocess your dataset correctly. Key preprocessing tasks include:
   - Handling missing values
   - Encoding categorical features
   - Normalizing numerical data
   - Splitting data into training and test sets

3. **Update Feature Selection Script**: Modify the `feature_selection_exp.py` or `feature_selection_exp_CIC.py` script to load your dataset and implement any specific logic related to your data structure.

4. **Run the Experiment**: Once the preprocessing is complete and the feature selection script is updated, you can execute the script to evaluate feature selection and model performance on your new dataset.

### 4. Scenarios and Modules
In the provided scripts, different scenarios and modules (e.g., feature selection, data balancing, hyperparameter tuning) are implemented. The scenarios outlined in the paper can be linked to specific sections of the code as follows:

- **Data Preparation**: Handled in the `main_code.py` file, where preprocessing and dataset preparation functions are defined.
- **Feature Selection**: Performed by the `feature_selection_exp.py` and `feature_selection_exp_CIC.py` scripts, which evaluate different feature selection methods.
- **Anomaly Detection**: Integrated into the experiment scripts where the selected features are used to train models.
- **Hyperparameter Tuning**: Built into the scripts, particularly using KerasTuner for deep learning models.
- **Data Balancing**: Methods like SMOTE are incorporated into the experiment scripts where applicable.

By following the above steps, users can replicate the feature selection experiments or adapt them to new datasets, helping them optimize anomaly detection performance in various scenarios.
