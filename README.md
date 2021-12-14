# us-census

Predicting high/low salaries from US census data.

# Python Environment

This repository uses Python 3.8.

The Python packages are stored in the `env.yml` file. These can be installed using `conda` by running:

```shell
conda env create -f env.yml
```

# Workflow

There are three main pieces of this project:

1. [Exploratory Data Analysis notebook](eda.ipynb).

2. [Data Processing script](processing.py)

   - This handles all the loading, processing and transformations associated with the project.
   - You can run the script by executing `python3 -m processing --name my_baseline` to create a new training/test set under the `data/processing/my-baseline` folder.

3. [Modelling](modelling.py)

   - scfsc
   - sdcsa

# Code Formatting

This respository uses `black` and `isort` for code and package import formatting.
To run these execute the following commands in the terminal;

- `black <file_name>` or `black .` for all files.
- `isort <file_name>` or `isort .` for all files.
