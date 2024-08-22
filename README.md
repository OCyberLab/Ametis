
# Thesis Source Code - README

## Overview
This repository contains the source code accompanying the thesis titled: **[Insert Your Thesis Title Here]**. The primary focus of the thesis is on machine learning models and data augmentation for IoT anomaly detection.

## Prerequisites
This code is developed using Python 3.11.5 and requires a GPU for running deep learning models efficiently. Ensure you have the appropriate hardware setup.

## Environment Setup
To replicate the environment used in this project, use the following Conda environment file. This ensures that all dependencies are correctly installed.

### Conda Environment
Create the environment using the following command:

```bash
conda env create -f gpuenv.yml
```

Activate the environment:

```bash
conda activate gpuenv
```

### `gpuenv.yml`
This is the content of the `gpuenv.yml` file:

```yaml
name: gpuenv
channels:
  - defaults
dependencies:
  - python=3.11.5
  - pip
  - pip:
    - imbalanced-learn==0.10.1
    - keras
    - keras-tuner
    - matplotlib
    - numpy  # Consider specifying a version  
    - oauthlib==3.2.2
    - opt-einsum==3.3.0
    - packaging==23.1
    - pandas
    - scikit-learn==1.2.0
    - scipy
    - tensorflow-macos
    - tensorflow-metal
    - data-complexity
    - xgboost
    - prompt_toolkit
    - mlxtend
    - boruta
    - skrebate
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

## License
[Insert any relevant licensing information]

## Contact
For any inquiries or issues, please contact [Your Email].
