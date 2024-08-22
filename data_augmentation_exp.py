import sys
sys.stdout.flush()

"""# Data Augmentation EXP"""

import itertools
import datetime
from main_code import print_green, check_row_exist, clean_data, scale_data, balance_data_and_apply_feature_selection, train_model, update_csv
import pandas as pd

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
if feature_selection == "NA":
    feature_selection = "NO FS"
model_name = args.model_name

print_green(f"Feature Selection {feature_selection} - Model {model_name} - Scenario {scenario}")

csv_file_path = f"results/DA/result_data_augmentation_result_IOT23_{scenario}_{feature_selection}_{model_name}.csv"
epochs = 200
column_names = ["Accuracy", "F1-Score", "F1-Macro", "F1-Micro", "G-Mean", "AUC", "Recall", "Precision", "Model Training Total Time", "Scenario", "Model", "Feature Selection", "Dataset", "Total Features", "Selected Features", "Feature Selection Time", "Epochs"]

datasets = ["8-1", "3-1", "1-1", "20-1", "42-1"]
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
    dataset_folder_path = 'datasets/IoT23/'
    df = pd.read_csv(dataset_folder_path + dataset + '.csv')
    dfs.append(df)
    print_green("-----------------")
    print(f"Dataset loaded: {dataset}")
    print_green("-----------------")
  df_train = pd.concat(dfs, ignore_index=True)
  print_green("-----------------")
  print(f"Datasets in combination merged: {train_dataset_name}")
  print_green("-----------------")
  df_train = clean_data(df_train)
  x_main, y_main, x_test_main, y_test = scale_data(df_train, 'datasets/IoT23/test.csv')
  print_green("-----------------")
  print(f"Train dataset preprocess is done. Shape: {x_main.shape}")
  print_green("-----------------")

  print_green("-----------------")
  print(f"Applying Feature Selection ....")
  print_green("-----------------")
  x, x_test, y, feature_selection_total_time, selected_features_indices = balance_data_and_apply_feature_selection(x_main, x_test_main, y_main, y_test, scenario, feature_selection)
  print_green("-----------------")
  print(f"Feature Selection is Done")
  print_green("-----------------")


  print_green("-----------------")
  print(f"Training Model ....")
  print_green("-----------------")
  result, auc_score, selected_features, _, keras_tuner_total_time = train_model(model_name, x, y, x_test, y_test, train_dataset_name, scenario, epochs, feature_selection)
  update_csv(csv_file_path, column_names, model_name, result, auc_score, keras_tuner_total_time, feature_selection, train_dataset_name, selected_features, x_main.shape[1], feature_selection_total_time, scenario, epochs)
  print_green("-----------------")
  print(f"Training Model is Done")
  print_green("-----------------")

  print_green("-----------------")
  print(f"Model AUC Score: {auc_score}")
  print(f"Result: {result}")
  print_green("-----------------")

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
            dataset_folder_path = 'datasets/IoT23/'
            df = pd.read_csv(dataset_folder_path + dataset + '.csv')
            dfs.append(df)
            print_green("-----------------")
            print(f"Dataset loaded: {dataset}")
            print_green("-----------------")
        df_train = pd.concat(dfs, ignore_index=True)
        print_green("-----------------")
        print(f"Datasets in combination merged: {train_dataset_name}")
        print_green("-----------------")
        df_train = clean_data(df_train)
        x_main, y_main, x_test_main, y_test = scale_data(df_train, 'datasets/IoT23/test.csv')
        print_green("-----------------")
        print(f"Train dataset preprocess is done. Shape: {x_main.shape}")
        print_green("-----------------")

        print_green("-----------------")
        print(f"Applying Feature Selection ....")
        print_green("-----------------")
        x, x_test, y, feature_selection_total_time, selected_features_indices = balance_data_and_apply_feature_selection(x_main, x_test_main, y_main, y_test, scenario, feature_selection)
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


