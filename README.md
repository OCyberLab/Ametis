
# README

## Overview
This repository contains the source code accompanying the paper titled: **An Empirical Study on Learning Models and Data Augmentation for IoT Anomaly Detection**.

## Abstract
Among many other security applications, anomaly detection is one of the biggest users of deep learning methods.
This growing popularity is mainly driven by two common beliefs: (i) its ability to manage complicated patterns inside large datasets (given a large amount of data) and (ii) its no need of separate feature engineering (as it is done within the model learning). In this study, we question both of those beliefs and revisit the effectiveness of feature selection and data augmentation in the performance of popular deep-learning based anomaly detection approaches. Additionally, we study the impact of other important factors of any learning based anomaly detection approaches including learning models (both traditional Ml and deep learning), data balancing techniques, hyper parameter tuning, etc. on their performance. From this study, we first report that those common beliefs are not always true - which necessitates a framework that can evaluate the usefulness of features and data for specific use cases (varying the data and need). Then, we propose a new framework that can fill in this gap and assist the data users and anomaly detection tools to perform better by selectively choosing all the configurations (such as, features, models, hyper parameter, balanced data, augmented data). Finally, we demonstrate the effectiveness of our framework using two major IoT datasets.

## Prerequisites
This code is developed using Python 3.11.5 and requires a GPU for running deep learning models efficiently. Ensure you have the appropriate hardware setup.

## Environment Setup
To replicate the environment used in this project, use the following Conda environment file. This ensures that all dependencies are correctly installed.

### Conda Environment
Create the environment using the following command:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate gpuenv
```

**Notes:**
- If you are using a GPU that supports CUDA, consider switching to `tensorflow-gpu` instead of `tensorflow-macos` and `tensorflow-metal`.
- Ensure that the versions of your dependencies are compatible with each other, especially TensorFlow, Keras, and the GPU drivers.

## Running the Code
To run the experiments, use the following commands:

```bash
# Insert specific instructions on how to run your code
```

## Citation
If you find this work useful, please cite our paper:

```bibtex
@inproceedings{toghiankhorasgani2024empirical,
  title={An Empirical Study on Learning Models and Data Augmentation for IoT Anomaly Detection},
  author={Toghiani Khorasgani, Alireza and Shirani, Paria and Majumdar, Suryadipta},
  booktitle={2024 IEEE Conference on Communications and Network Security (CNS)},
  year={2024},
  organization={IEEE}
}
```

## Contact
For any inquiries or issues, please contact alirezatoghyiani@gmail.com.
