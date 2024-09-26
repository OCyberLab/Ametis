import sys
sys.stdout.flush()


"""# Feature Selection EXP"""

from main_code import print_green, check_row_exist, balance_data_and_apply_feature_selection, train_model, update_csv, clean_data_cic, scale_data_cic
import pandas as pd
import datetime
import os
import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description="Process the datasets and select best combinations.")
parser.add_argument('--train_dataset_name', type=str, required=True, help='Name of the training dataset.')

# Parse arguments
args = parser.parse_args()

# Use the arguments
train_dataset_name = args.train_dataset_name

parser = argparse.ArgumentParser(description="Process some inputs for the experiment.")

# Add arguments
parser.add_argument('train_dataset_name', type=str, help='Enter the train dataset name')

train_dataset_path = "datasets/CIC/"
epochs = 200

# Construct the dynamic file path
csv_path = "results" + "/result_" + train_dataset_name + ".csv"
print_green(csv_path)

df = pd.read_csv(train_dataset_path + train_dataset_name + '.csv')
df_train = clean_data_cic(df)
print(df_train.columns)
x_main, y_main, x_test_main, y_test = scale_data_cic(df_train, train_dataset_path + "CIC_test.csv")

column_names = ["Accuracy", "F1-Score", "F1-Macro", "F1-Micro", "G-Mean", "AUC", "Recall", "Precision", "Model Training Total Time", "Scenario", "Model", "Feature Selection", "Dataset", "Total Features", "Selected Features", "Feature Selection Time", "Epochs"]

# Check if the CSV file exists
if os.path.exists(csv_path):
    print_green("**************")
    print_green("Load the existing CSV file")
    print_green("**************")
    df = pd.read_csv(csv_path)
else:
    print_green("**************")
    print_green("Create a new CSV file with specified column names")
    print_green("**************")
    df = pd.DataFrame(columns=column_names)
    df.to_csv(csv_path, index=False)

for scenario in ["S0", "S1" , "S2", "S3", "S4", "S5", "S7", "S6", "S8", "S9"]:
  if scenario in ["S0", "S2", "S5", "S7"]:
    feature_selection_list = ["NO FS"]
  else:
    feature_selection_list = ["PCA", "RF", "trank", "MI", "Chi2", "L1", "L2", "SelectKBest", "PSO"]

  if scenario in ["S5", "S6", "S7", "S8", "S9"]:
    model_list = ["CNN_HT", "XGBoost_HT", "IF_HT", "AE_HT", "NN_HT"]
  else:
    model_list = ["CNN", "XGBoost", "IF", "AE", "NN"]

  print_green("-----------------------------")
  print_green(scenario)
  print_green(str(model_list))
  print_green(str(feature_selection_list))
  print_green("-------------")
  for feature_selection in feature_selection_list:
    # if feature_selection in ["L1", "L2"] and "AE_HT" in model_list:
    #   continue

    print_green("**************")
    print_green(str(feature_selection))
    print_green("**************")
    count = 0
    for model_name in model_list:
      if check_row_exist(csv_path, feature_selection, scenario, model_name, train_dataset_name):
        count += 1
    if count == len(model_list):
      continue
    x, x_test, y, feature_selection_total_time, selected_features_indices = balance_data_and_apply_feature_selection(x_main, x_test_main, y_main, y_test, scenario, feature_selection)
    for model_name in model_list:
      print_green("++++++++++++")
      print_green(scenario)
      print_green(model_name)
      print_green(feature_selection)
      if model_name in ["XGBoost", "XGBoost_HT", "RF", "RF_HT", "IF", "IF_HT"] and feature_selection in ["L1", "L2"]:
        result = {'accuracy': 0, 'gmean': 0, 'f1_micro': 0, 'f1_macro': 0, 'weighted avg': {'f1-score': 0, 'recall': 0, 'precision': 0}}
        update_csv(csv_path, column_names, model_name, result, 0, 0, feature_selection, train_dataset_name, x_main.shape[1], x_main.shape[1], 0, scenario, epochs)
        print_green("Model and Feature Selection not compatible, 0 added to result!")
        continue


      if check_row_exist(csv_path, feature_selection, scenario, model_name, train_dataset_name):
        print_green("result exists")
      else:
        if feature_selection in ["L1", "L2"] and model_name in ["AE_HT"]:
          result, auc_score, selected_features, _, keras_tuner_total_time = train_model(model_name, x, y, x_test, y_test, train_dataset_name, scenario, epochs, "NO FS")
          update_csv(csv_path, column_names, model_name, result, auc_score, keras_tuner_total_time, feature_selection, train_dataset_name, selected_features, x_main.shape[1], feature_selection_total_time, scenario, epochs)
        else:
          result, auc_score, selected_features, _, keras_tuner_total_time = train_model(model_name, x, y, x_test, y_test, train_dataset_name, scenario, epochs, feature_selection)
          update_csv(csv_path, column_names, model_name, result, auc_score, keras_tuner_total_time, feature_selection, train_dataset_name, selected_features, x_main.shape[1], feature_selection_total_time, scenario, epochs)
      print_green("++++++++++++")
