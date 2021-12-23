# US Census Income Analysis

Predicting high / low salary band for a person based on US census data.

# Python Environment

This repository uses Python 3.8.

The Python packages are stored in the `env.yml` file. These can be installed using `conda` by running:

```shell
conda env create -f env.yml
```

# Workflow

There are three main pieces of this project:

1. [Exploratory Data Analysis](eda.ipynb).

2. [Data Processing](processing.py)

   - This handles all the loading, processing and transformations associated with the project.
   - Once executed it will run the following:

     a) Create a report of missing values within the raw dataset.

     b) Encode the salary threshold labels into a `[0,1]` format.

     c) One-hot encode and scale the relevant feature variables.

     d) Apply PCA-based dimensionality reduction (if requested)

     e) Split and Save the train and test datasets to a folder called `data/processed/<tag>/`.

```shell
python3 -m processing --tag my_baseline --use_pca False
```

3. [Modelling](modelling.py)

   - This handles all of the model training and evaluation associated with this project.
   - Once executed it will run the following:

     a) Load the training and test datasets from the local storage `data/processed/<data>` folder.

     b) Fit the name `--algo` model to the dataset (see `MODELS` for a range of choices).

     c) Evaluate the training, validation and test performance and save the results to `results/<tag>.json`.

     d) Save the trained model under a directory called `models/<tag>.pkl`

```shell
python3 -m modelling --data baseline --algo Ridge --tag baseline_model
```

# Code Formatting

This respository uses `black` and `isort` for code and package import formatting.
To run these execute the following commands in the terminal;

- `black <file_name>` or `black .` for all files.
- `isort <file_name>` or `isort .` for all files.
