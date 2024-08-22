import sys
sys.stdout.flush()

import itertools
import datetime
from main_code import print_green, check_row_exist, scale_data, balance_data_and_apply_feature_selection_bert, train_model
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os
import csv
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
from math import sqrt
from transformers.trainer_pt_utils import get_parameter_names
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset

# CHECK GPU availability
print_green("-----------------")
print_green("GPU AVAILABILITY:")
print_green("-----------------")

# Check available devices and use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: CUDA (GPU)")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")  # Use Apple Silicon GPU
    print(f"Using device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print(f"Using device: CPU")
print_green("-----------------")

# FUNCTIONS
def update_csv(file_path, row_data, column_names):
    new_row = pd.DataFrame([row_data], columns=column_names)
    if not os.path.exists(file_path):
        new_row.to_csv(file_path, index=False)
    else:
        df = pd.read_csv(file_path)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(file_path, index=False)


def clean_data_cic(df):
    # Missing Values
    df = df.replace('-', float('NaN'))
    df.fillna(0, inplace=True)

    for col in df.columns:
        if df[col].dtype == 'float64' and df[col].mean() == 0:
            df[col] = df[col].astype(pd.SparseDtype(float, fill_value=0))

    # Dealing with Duplicate Data
    df = df.drop_duplicates()

    return df

def scale_data_cic(df_train, test_path):
  scaler = MinMaxScaler(feature_range=(0, 1))

  y_train = np.where(df_train.label.str.contains('Benign'), 0, 1)
  x = df_train.drop(['label'], axis=1)
  
  # One-hot encoding for categorical data
  categorical_cols = x.select_dtypes(include=['object', 'category']).columns
  df = pd.get_dummies(x, columns=categorical_cols, drop_first=True)

  df_test = pd.read_csv(test_path)
  df_test = clean_data_cic(df_test)
  
  x_test = df_test.reindex(labels=x.columns,axis=1)
  x_test.replace(float('NaN'), 0, inplace=True)
  y_test = np.where(df_test.label.str.contains('Benign'), 0, 1)
  
  X_train_scaled = scaler.fit_transform(x)  # compute min, max of the training data and scale it
  X_test_scaled = scaler.transform(x_test)  # scale test data using the same min and max values

  return X_train_scaled, y_train, X_test_scaled, y_test

def preprocess_data_with_label_encoder(df):
    x = df_train.drop(['label'], axis=1)
    categorical_cols = x.select_dtypes(include=['object', 'category']).columns
    encoder = LabelEncoder()
    for column in symbolic_coloumns:
        df[column] = df[column].astype(str)
        df[column] = encoder.fit_transform(df[column])
    df.fillna(0, inplace=True)
    return df

def select_features(data, selected_features_indices):
    selected_features = data.columns[selected_features_indices]
    print(f"\033[92mSelected Columns: {selected_features}\033[0m")
    return data[selected_features].astype(str).agg(' '.join, axis=1)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)  # Ensure labels are long

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]
        
class TextDatasetHT(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)  # Ensure labels are long

        # Check if all encodings have the same length as labels
        encoding_lengths = [len(val) for val in self.encodings.values()]
        if not all(length == len(self.labels) for length in encoding_lengths):
            raise ValueError("The length of all encodings must match the length of labels")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]  # Labels are already torch.long
        return item

def compute_metrics_fn(eval_preds):
    metric = dict()
    y_pred = eval_preds.predictions.argmax(axis=-1)
    y_true = eval_preds.label_ids
    metric['accuracy'] = accuracy_score(y_true, y_pred)
    metric['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metric['f1'] = f1_score(y_true, y_pred)
    metric['f1_micro'] = f1_score(y_true, y_pred, average='micro')
    metric['precision'] = precision_score(y_true, y_pred)
    metric['recall'] = recall_score(y_true, y_pred)
    y_pred_proba = eval_preds.predictions[:, 1]
    metric['auc'] = roc_auc_score(y_true, y_pred_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metric['g_mean'] = sqrt((tp / (tp + fn)) * (tn / (tn + fp)))
    return metric

def run_bert_experiment(scenario, feature_selection, model_name, x_train, y_train, x_test, y_test, epochs, csv_file_path):

    if model_name == "BERT":
        # Your BERT training and evaluation code here
        print_green("Applying Feature Selection ....")
        x_train, x_test, y_train, feature_selection_total_time, selected_features_indices = balance_data_and_apply_feature_selection_bert(x_train, x_test, y_train, y_test, scenario, feature_selection)
        print_green("Feature Selection is Done")
        X_train_text = select_features(x_train, selected_features_indices)
        X_test_text = select_features(x_test, selected_features_indices)
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        train_dataset = TextDataset(X_train_text, pd.Series(y_train))
        test_dataset = TextDataset(X_test_text, pd.Series(y_test))
        num_train_epochs = epochs
        def custom_collate_fn(batch):
            texts = [item[0] for item in batch]
            labels = torch.tensor([item[1] for item in batch])
            tokenized_inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            return {
                "input_ids": tokenized_inputs["input_ids"],
                "attention_mask": tokenized_inputs["attention_mask"],
                "labels": labels
            }
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=16,
            evaluation_strategy="steps",
            save_strategy='epoch',
            logging_strategy='epoch'
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=custom_collate_fn,
            compute_metrics=compute_metrics_fn,
        )
        start_time = datetime.datetime.now()
        trainer.train()
        end_time = datetime.datetime.now()
        training_time = (end_time - start_time).total_seconds()
        test_results = trainer.evaluate()
        metrics = test_results
        print_green("Evaluation Results:")
        for key, value in metrics.items():
            print_green(f"{key}: {value:.4f}")
        row_data = [
            metrics['eval_accuracy'],
            metrics['eval_f1'],
            metrics['eval_f1_macro'],
            metrics['eval_f1_micro'],
            metrics['eval_g_mean'],
            metrics['eval_auc'],
            metrics['eval_recall'],
            metrics['eval_precision'],
            training_time,
            scenario,
            "BERT",
            feature_selection,
            train_dataset_name,
            len(x_train.columns),
            len(selected_features_indices),
            feature_selection_total_time,
            num_train_epochs
        ]
        update_csv(csv_file_path, row_data, column_names)
        
    else:
        print_green("Model not supported for this experiment")

# Hyperparamter TUNING
import optuna
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer

def objective(trial, train_dataset, eval_dataset, tokenizer, model_name):
    # Define the hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])

    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        evaluation_strategy="steps",
        save_strategy='epoch',
        logging_strategy='epoch',
        learning_rate=learning_rate,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_result = trainer.evaluate()

    # Return the evaluation metric (e.g., accuracy) to optimize
    return eval_result["eval_accuracy"]

def run_bert_experiment_with_hyperparameter_tuning(scenario, feature_selection, model_name, x_train, y_train, x_test, y_test, epochs, csv_file_path):
    start_time = datetime.datetime.now()
    # Prepare the datasets, model, and tokenizer

    print_green("Applying Feature Selection ....")
    x_train, x_test, y_train, feature_selection_total_time, selected_features_indices = balance_data_and_apply_feature_selection_bert(x_train, x_test, y_train, y_test, scenario, feature_selection, True)
    print_green("Feature Selection is Done")
    X_train_text = select_features(x_train, selected_features_indices)
    X_test_text = select_features(x_test, selected_features_indices)
    model_name = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(X_train_text.tolist(), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(X_test_text.tolist(), truncation=True, padding=True, max_length=512)

    print("Length of train_encodings:", {len(val) for val in train_encodings.values()})
    print("Length of y_train:", len(y_train))

    train_dataset = TextDatasetHT(train_encodings, y_train)
    test_dataset = TextDatasetHT(test_encodings, y_test)

    # Optimize hyperparameters
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_dataset, test_dataset, tokenizer, model_name), n_trials=5)

    # Retrieve the best hyperparameters
    best_trial = study.best_trial
    best_learning_rate = best_trial.params["learning_rate"]
    best_batch_size = best_trial.params["per_device_train_batch_size"]

    # Train the model with the best hyperparameters
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=best_batch_size,
        evaluation_strategy="steps",
        save_strategy='epoch',
        logging_strategy='epoch',
        learning_rate=best_learning_rate,
    )

    trainer = Trainer(
    model=AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics_fn,
    )

    trainer.train()
    end_time = datetime.datetime.now()
    training_time = (end_time - start_time).total_seconds()

    # Evaluate the model and save the results
    test_results = trainer.evaluate()
    metrics = test_results
    print_green("Evaluation Results:")
    for key, value in metrics.items():
        print_green(f"{key}: {value:.4f}")
    row_data = [
            metrics['eval_accuracy'],
            metrics['eval_f1'],
            metrics['eval_f1_macro'],
            metrics['eval_f1_micro'],
            metrics['eval_g_mean'],
            metrics['eval_auc'],
            metrics['eval_recall'],
            metrics['eval_precision'],
            training_time,
            scenario,
            "BERT_HT",
            feature_selection,
            train_dataset_name,
            len(x_train.columns),
            len(selected_features_indices),
            feature_selection_total_time,
            epochs
    ]
    update_csv(csv_file_path, row_data, column_names)
# MAIN CODE
import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description="Process the datasets and select best combinations.")
parser.add_argument('--scenario', type=str, required=True, help='Name of the scenario.')
parser.add_argument('--feature_selection', type=str, required=True, help='Name of the feature selection.')
parser.add_argument('--model_name', type=str, required=True, help='Name of the model.')

# Parse arguments
args = parser.parse_args()

# Use the arguments
scenario = args.scenario
feature_selection = args.feature_selection
model_name = args.model_name

csv_file_path = f"results/DA/result_data_augmentation_result_BERT_CIC_{scenario}_{feature_selection}_{model_name}.csv"
epochs = 1
column_names = ["Accuracy", "F1-Score", "F1-Macro", "F1-Micro", "G-Mean", "AUC", "Recall", "Precision", "Model Training Total Time", "Scenario", "Model", "Feature Selection", "Dataset", "Total Features", "Selected Features", "Feature Selection Time", "Epochs"]


datasets = ["CIC1", "CIC2", "CIC4", "CIC5", "CIC6"]
combinations = []
for r in range(1, len(datasets) + 1):
    combinations.extend(list(itertools.combinations(datasets, r)))

print_green("-----------------")
print(f"Number of combinations: {len(combinations)}")
print(combinations)
print_green("-----------------")

for i, combination in enumerate(combinations):
    print_green("-----------------")
    print(f"Experiment Progress: {round(100 * i / len(combinations), 2)} % complete")
    print_green("-----------------")
    dfs = []
    train_dataset_name = ", ".join(combination)
    if check_row_exist(csv_file_path, feature_selection, scenario, model_name, train_dataset_name):
        print_green("-----------------")
        print(f"Result exists for Dataset: {train_dataset_name}")
        print_green("-----------------")
        continue
    for dataset in combination:
        dataset_folder_path = 'datasets/CIC/'
        df = pd.read_csv(dataset_folder_path + dataset + '.csv')
        dfs.append(df)
        print_green("-----------------")
        print(f"Dataset loaded: {dataset}")
        print_green("-----------------")
    df_train = pd.concat(dfs, ignore_index=True)
    print_green("-----------------")
    print(f"Datasets in combination merged: {train_dataset_name}")
    print_green("-----------------")
    # Your BERT training and evaluation code here

    df_train = clean_data_cic(df_train)
    df_test = pd.read_csv(dataset_folder_path + "CIC_test.csv")
    df_test = clean_data_cic(df_test)
    y_train = np.where(df_train.label.str.contains('Benign'), 0, 1)
    x_train = df_train.drop(['label'], axis=1)
    y_test = np.where(df_test.label.str.contains('Benign'), 0, 1)
    x_test = df_test.reindex(labels=x_train.columns,axis=1)
    x_test.replace(float('NaN'), 0, inplace=True)

    print_green("-----------------")
    print(f"Train dataset preprocess is done. Shape: {x_train.shape}")
    print_green("-----------------")


    print_green("-----------------")
    print(f"Training Model ....")
    print_green("-----------------")

    if check_row_exist(csv_file_path, feature_selection, scenario, model_name, train_dataset_name):
        print_green("Result exists")
        continue
    else:
        if scenario in ["S5", "S6", "S7", "S8", "S9"]:
            run_bert_experiment_with_hyperparameter_tuning(scenario, feature_selection, model_name, x_train, y_train, x_test, y_test, epochs, csv_file_path)
        else:
            run_bert_experiment(scenario, feature_selection, model_name, x_train, y_train, x_test, y_test, epochs, csv_file_path)
    print_green("-----------------")
    print(f"Training Model is Done")
    print_green("-----------------")


#CALCULATE COMPLEXITY


import itertools
import dcm
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os
import time

from main_code import print_green, clean_data, scale_data, balance_data_and_apply_feature_selection

def L2(X, y):
  # Train a logistic regression model on the scaled training data
  logreg = LogisticRegression().fit(X, y)

  # Predict the classes for the training data using the trained model
  Y_pred = logreg.predict(X)

  # Calculate the L2 error rate
  n = len(y)
  L2 = sum(Y_pred != y) / n

  return L2

def check_row_exist(csv_path, feature_selection, scenario, train_dataset_name):
  """
  Check if the row exists in the feature_selection dataframe.

  Args:
    feature_selection: The feature_selection dataframe.
    scenario: The scenario.
    model_name: The model name.
    train_dataset_name: The train dataset name.

  Returns:
    A boolean value indicating whether the row exists or not.
  """

  if os.path.exists(csv_path):
    # Load the existing CSV file
    df = pd.read_csv(csv_path)
  else:
    return False

  row = df[
      (df['Scenario'] == scenario) &
      (df["Feature Selection"] == feature_selection) &
      (df["Dataset"] == train_dataset_name)]
  return len(row) > 0

def compute_data_complexity(datasets, csv_path, scenario, feature_selection, model_name):
    
    print_green("-----------------")
    print(f"Number of combinations: {len(combinations)}")
    print(combinations)
    print_green("-----------------")
    
    df_result = pd.read_csv(csv_path)
    
    df_result['CM'] = 0
    df_result['L2_Complexity'] = 0
    df_result['F1_Complexity'] = 0
    df_result['C1_Complexity'] = 0
    df_result['C2_Complexity'] = 0
    for index, combination in enumerate(combinations):
        print_green("-----------------")
        print(f"Experiment Progress: {round(100 * index / len(combinations), 2)} % complete")
        print_green("-----------------")
        dfs = []
        train_dataset_name = ", ".join(combination)
        for dataset in combination:
            dataset_folder_path = 'datasets/CIC/'
            df = pd.read_csv(dataset_folder_path + dataset + '.csv')
            dfs.append(df)
            print_green("-----------------")
            print(f"Dataset loaded: {dataset}")
            print_green("-----------------")
        df_train = pd.concat(dfs, ignore_index=True)
        print_green("-----------------")
        print(f"Datasets in combination merged: {train_dataset_name}")
        print_green("-----------------")
        df_train = clean_data_cic(df_train)
        df_test = pd.read_csv(dataset_folder_path + "CIC_test.csv")
        df_test = clean_data_cic(df_test)
        y_train = np.where(df_train.label.str.contains('Benign'), 0, 1)
        x_train = df_train.drop(['label'], axis=1)
        y_test = np.where(df_test.label.str.contains('Benign'), 0, 1)
        x_test = df_test.reindex(labels=x_train.columns,axis=1)
        x_test.replace(float('NaN'), 0, inplace=True)

        print_green("-----------------")
        print(f"Train dataset preprocess is done. Shape: {x_main.shape}")
        print_green("-----------------")

        print_green("-----------------")
        print(f"Applying Feature Selection ....")
        print_green("-----------------")
        x, x_test, y, feature_selection_total_time, selected_features_indices = balance_data_and_apply_feature_selection_bert(x_train, x_test, y_train, y_test, scenario, feature_selection, True)
        print_green("-----------------")
        print(f"Feature Selection is Done")
        print_green("-----------------")

        start_time = time.time()
        L2_value = L2(x, y)
        end_time = time.time()
        elapsed_time = (end_time - start_time)
        print_green("-----------------")
        print(f'L2 Complexity: {L2_value}')
        print_green(f"Time taken for L2 Complexity: {elapsed_time:.2f} seconds")
        print_green("-----------------")

        start_time = time.time()
        i, F1 = dcm.F1(x, y)
        end_time = time.time()
        elapsed_time = (end_time - start_time)
        print_green("-----------------")
        print(f'F1 Complexity: {F1}')
        print_green(f"Time taken for F1 Complexity: {elapsed_time:.2f} seconds")
        print_green("-----------------")

        start_time = time.time()
        C1, C2 = dcm.C12(x, y)
        end_time = time.time()
        elapsed_time = (end_time - start_time)
        print_green("-----------------")
        print(f'C1 Complexity: {C1}')
        print(f'C2 Complexity: {C2}')
        print_green(f"Time taken for C1 & C2: {elapsed_time:.2f} seconds")
        print_green("-----------------")
        
        # Assuming you have a DataFrame 'df' with a column 'label'
        count_label_1 = len(y[y == 1])
        count_label_0 = len(y[y == 0])

        # Calculate the ratio
        if count_label_0 > 0:  # To avoid division by zero
            ratio = count_label_1 / count_label_0
        else:
            ratio = 0  # Or some other value indicating that the ratio is not defined
        rows = x.shape[0]
        
        print_green(f"{ratio}")
        print_green(f"{rows}")
        # Calculate the number of normal instances
        normal_instances = rows / (1 + ratio)
        print_green(f"{normal_instances}")

        # For balanced data, the number of attack instances will be equal to the number of normal instances
        attack_instances_new = rows - normal_instances
        print_green(f"{attack_instances_new}")

        # Calculate the new total number of instances after oversampling and balancing
        rows = max(2*attack_instances_new, 2*normal_instances)
        ratio = 1
        print_green("-----------------")
        print_green("Data Ratio and Size")
        print_green("-----------------")
        print(f'# Attack {count_label_1}')
        print(f'# Normal {count_label_0}')
        print(f'Data Ratio(Attack / Normal) {ratio}')
        print(f'Data Rows {rows}')
        print_green("-----------------")
    
#        start_time = time.time()
#
#        error_rates = []
#        l = x.shape[0]
#        kList = [1, 5, 10, 20, 30, 50, 100]
#        minErrorRate = 99999999999999999999999999
#
#        for i in kList:
#            knn = KNeighborsClassifier(n_neighbors=i)
#            nbrs = knn.fit(x,y)
#            pred_i = knn.predict(x_test)
#            error_rate = np.mean(pred_i != y_test)
#            if error_rates != []:
#                if error_rate == error_rates[-1]:
#                    continue
#            if error_rate < minErrorRate:
#                k = i
#                minErrorRate = error_rate
#
#            error_rates.append(error_rate)
#            print_green("-----------------")
#            print("K = ", i)
#            print("Error rate = ", error_rate)
#            print_green("-----------------")
#
#        k = kList[error_rates.index(min(error_rates))]
#
#        print("K for min error rate = ", k)
#
#        kmeanModel = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
#        print("K neighbor model ....")
#        nbrs = kmeanModel.fit(x)
#        print("K neighbor model fitted")
#        distances, indices = nbrs.kneighbors(x)
#
#        print("Calculating CM ...")
#        count0 = int(pd.Series(y).value_counts()[0])
#        count1 = int(pd.Series(y).value_counts()[1])
#
#        if count0 < count1:
#            minClass = 0
#        else:
#            minClass = 1
#
#        for i in range(len(y)):
#            diffCount = 0
#            # wDiffCount = 0
#            if y[i] == minClass:
#                count = 0
#                # wCount = 0
#                for j in indices[i]:
#                    if y[j] == minClass:
#                        count += 1
#                        # wCount += distances[i]
#                if count/float(k) > 0.5:
#                    diffCount += 1
#            # if wCount/float(k) > 0.5:
#            #   wDiffCount += 1

#
#        CM = diffCount/(pd.Series(y).value_counts()[minClass])
        CM = 0
#        # wCM = wDiffCount/(pd.Series(y).value_counts()[minClass])
#
#        end_time = time.time()
#        elapsed_time = (end_time - start_time)
#        print_green("-----------------")
#        print(f'CM Complexity: {CM}')
#        print_green(f"Time taken for CM Complexity: {elapsed_time:.2f} seconds")
#        # print(f'wCM Complexity {wCM}')
#        print_green("-----------------")

  # print_green("-----------------")
  # print("Balancing with SMOTE")
  # print_green("-----------------")
  # train_class0_count = np.count_nonzero(train_df.label == 0)
  # train_class1_count = np.count_nonzero(train_df.label == 1)
  # print('Balancing train data ...')
  # X_train, Y_train, X_test, Y_test, X_train_scaled, X_test_scaled = scale_data(train_df, test_df)
  # smote = SMOTE()

  # X_smote, Y_smote = smote.fit_resample(X_train_scaled, Y_train)

  # print('Original dataset shape:\n', Y_train.value_counts())
  # print('Resample dataset shape:\n', Y_smote.value_counts())

  # X_train = X_smote
  # Y_train = Y_smote
  # L2_balanced = L2(X_train_scaled, Y_train)
  # print_green("-----------------")
  # print(f'L2 Complexity Balanced {L2_balanced}')
  # print_green("-----------------")
        # Create a new row with zeros
        column_names = ["Dataset", "Feature Selection", "Scenario", "L2_Complexity", "F1_Complexity", "C1_Complexity", "C2_Complexity", "CM", "Data Ratio", "# Rows"]
        df_result.loc[index, 'CM'] = CM
        df_result.loc[index, 'L2_Complexity'] = L2_value
        df_result.loc[index, 'F1_Complexity'] = F1
        df_result.loc[index, 'C1_Complexity'] = C1
        df_result.loc[index, 'C2_Complexity'] = C2
        df_result.loc[index, 'Data Ratio'] = ratio
        df_result.loc[index, '# Rows'] = rows

        df_result.to_csv(csv_path, index=False)


# Construct the dynamic file path
compute_data_complexity(datasets, csv_file_path, scenario, feature_selection, model_name)


