import sys
sys.stdout.flush()

import itertools
import datetime
from main_code import print_green, check_row_exist, scale_data, balance_data_and_apply_feature_selection_bert, train_model, clean_data, preprocess_data_with_label_encoder
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


#def clean_data(df):
#    # Missing Values
#    df = df.replace('-', float('NaN'))
#    nan_columns = ['local_resp', 'local_orig']
#    if 'id.orig_h' in df.columns:
#        df.rename(columns={'id.orig_h': 'id_orig_h'}, inplace=True)
#    if 'id.resp_h' in df.columns:
#        df.rename(columns={'id.resp_h': 'id_resp_h'}, inplace=True)
#    features_for_removal = ['id_orig_h', 'id_resp_h']
#    df.drop(features_for_removal, axis=1, inplace=True)
#    df["service"].fillna("No Service", inplace=True)
#    df["conn_state"].fillna("No State", inplace=True)
#    df.fillna(0, inplace=True)
#    for col in df.columns:
#        if df[col].dtype == 'float64' and df[col].mean() == 0:
#            df[col] = df[col].astype(pd.SparseDtype(float, fill_value=0))
#    # Dealing with Duplicate Data
#    df = df.drop_duplicates()
#    # Delete unnessary coloumns
#    df.drop(['ts', 'uid'], axis=1, inplace=True)
#    return df

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

def run_bert_experiment(scenario, feature_selection, model_name, train_dataset_name, epochs, csv_file_path):

    if model_name == "BERT":
        # Your BERT training and evaluation code here
        dataset_folder_path = 'datasets/IoT23/'
        df_train = pd.read_csv(dataset_folder_path + train_dataset_name + '.csv')
        df_train = clean_data(df_train)
        df_test = pd.read_csv(dataset_folder_path + 'test.csv')
        df_test = clean_data(df_test)
        y_train = np.where(df_train['tunnel_parents   label   detailed-label'].str.contains('Benign'), 0, 1)
        x_train = df_train.drop(['tunnel_parents   label   detailed-label'], axis=1)
        y_test = df_test['label']
        x_test = df_test.reindex(labels=x_train.columns,axis=1)
        x_test.replace(float('NaN'), 0, inplace=True)
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

def run_bert_experiment_with_hyperparameter_tuning(scenario, feature_selection, model_name, train_dataset_name, epochs, csv_file_path):
    start_time = datetime.datetime.now()
    # Prepare the datasets, model, and tokenizer
    dataset_folder_path = 'datasets/IoT23/'
    df_train = pd.read_csv(dataset_folder_path + train_dataset_name + '.csv')
    df_train = clean_data(df_train)
    df_test = pd.read_csv(dataset_folder_path + 'test.csv')
    df_test = clean_data(df_test)

    y_train = np.where(df_train['tunnel_parents   label   detailed-label'].str.contains('Benign'), 0, 1)
    x_train = df_train.drop(['tunnel_parents   label   detailed-label'], axis=1)
    y_test = df_test['label']
    x_test = df_test.reindex(labels=x_train.columns,axis=1)
    x_test.replace(float('NaN'), 0, inplace=True)

    print_green("Applying Feature Selection ....")
    x_train, x_test, y_train, feature_selection_total_time, selected_features_indices = balance_data_and_apply_feature_selection_bert(x_train, x_test, y_train, y_test, scenario, feature_selection)
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
    

column_names = ["Accuracy", "F1-Score", "F1-Macro", "F1-Micro", "G-Mean", "AUC", "Recall", "Precision", "Model Training Total Time", "Scenario", "Model", "Feature Selection", "Dataset", "Total Features", "Selected Features", "Feature Selection Time", "Epochs"]

#for scenario in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S9", "S8"]:
#    feature_selection_list = ["NO FS"] if scenario in ["S0", "S2", "S5", "S7"] else ["trank", "RF", "trank", "MI", "Chi2", "PCA", "PSO", "SelectKBest"]
#    if scenario in ["S5", "S6", "S7", "S8", "S9"]:
#        model_list = ["BERT_HT"]
#    else:
#        model_list = ["BERT"]
#
#    print_green("-----------------------------")
#    print_green(scenario)
#    print_green(str(model_list))
#    print_green(str(feature_selection_list))
#    print_green("-------------")
#
#    for feature_selection in feature_selection_list:
#        print_green("**************")
#        print_green(str(feature_selection))
#        print_green("**************")
#        for model_name in model_list:
#            if check_row_exist(csv_file_path, feature_selection, scenario, model_name, train_dataset_name):
#                print_green("Result exists")
#                continue
#            else:
#                if scenario in ["S5", "S6", "S7", "S8", "S9"]:
#                    run_bert_experiment_with_hyperparameter_tuning(scenario, feature_selection, model_name, train_dataset_name, epochs, csv_file_path)
#                else:
#                    run_bert_experiment(scenario, feature_selection, model_name, train_dataset_name, epochs, csv_file_path)
