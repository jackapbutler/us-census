"""Module for fitting and evaluating models on US census data"""
import abc
import argparse
import os
from typing import Dict
import pickle
import json
import pandas as pd
import numpy as np
import sklearn.metrics as sk_metrics
import sklearn.linear_model as sk_lin
import sklearn.linear_model._stochastic_gradient as sk_sgd
import sklearn.gaussian_process as sk_gpc
import sklearn.neighbors as sk_neigh
import sklearn.neural_network as sk_nn
import sklearn.tree as sk_tree
import sklearn.svm as sk_svm
import sklearn.ensemble as sk_ens

import processing

# OR use voting classifiers
# OR use deep learning

TRAINING_DIR = "data/processed"
RESULTS_DIR = "results"
MODELS_DIR = "models"

# models must have a fit and predict method
MODELS = {
    "Logistic": sk_lin.LogisticRegression(),
    "Ridge": sk_lin.RidgeClassifier(),
    "KNN": sk_neigh.KNeighborsClassifier(),
    "MLP": sk_nn.MLPClassifier(),
    "Tree": sk_tree.DecisionTreeClassifier(),
    "Tree+": sk_tree.ExtraTreeClassifier(),
    "SGD": sk_sgd.SGDClassifier(),
    "GPC": sk_gpc.GaussianProcessClassifier(),
    "SVC": sk_svm.SVC(),
    "AdaBoost": sk_ens.AdaBoostClassifier(),
    "Bagging": sk_ens.BaggingClassifier(),
    "RandForest": sk_ens.RandomForestClassifier(),
}

# overall accuracy, accuracy on segments (remove bias)
METRICS = {
    "Accuracy": sk_metrics.accuracy_score,
    "Precision": sk_metrics.precision_score,
    "Recall": sk_metrics.recall_score,
    "F1 Score": sk_metrics.f1_score,
}


class Model(abc.ABC):
    """Abstract base class for implementing machine learning models"""

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def fit(self, X, y):
        """Abstract method for fitting a machine learning model"""
        pass

    @abc.abstractmethod
    def predict(self, y):
        """Abstract method for predicting the output from the model"""
        pass


class Experiment:
    """Class to run a machine learning experiment"""

    def __init__(self, data_dir, algorithm_name, tag) -> None:
        self.data_dir: str = data_dir
        self.dataset_name: str = self.data_dir.split("/")[-1]
        self.algorithm_name: str = algorithm_name
        self.algorithm: Model = MODELS.get(self.algorithm_name)
        self.tag: str = tag

    def run(self):
        """Run experiment"""
        self.load_training_datasets()

        print(f"Fitting algorithm {self.algorithm_name} to the data.")
        self.trained_model = self.algorithm.fit(self.X_tr, self.y_tr)

        print("\n Using the trained model to predict the test set.")
        self.y_te_pred = self.trained_model.predict(self.X_te)
        self.y_tr_pred = self.trained_model.predict(self.X_tr)

        self.save_results()
        self.save_model()

    def load_training_datasets(self):
        """Load training and test datasets from local storage"""
        folder = f"{TRAINING_DIR}/{self.data_dir}/"
        print(f"Loading training datasets from {folder}")

        if os.path.isdir(folder):
            self.X_tr: pd.DataFrame = pd.read_csv(f"{folder}/X_tr.csv")
            self.y_tr: np.ndarray = pd.read_csv(f"{folder}/y_tr.csv").values.ravel()
            self.X_te: pd.DataFrame = pd.read_csv(f"{folder}/X_te.csv")
            self.y_te: np.ndarray = pd.read_csv(f"{folder}/y_te.csv").values.ravel()

        else:
            raise FileNotFoundError(
                f"Can't find training datasets folder called {folder}."
            )

    def load_original_dataset(self):
        """Load the original preprocessed dataset"""
        self.original_df: pd.DataFrame = pd.read_csv(processing.RAW_DATA)

    def generate_metrics(
        self,
        truth: np.ndarray,
        preds: np.ndarray,
    ) -> Dict:
        """Generate a dictionary of common classification metrics"""
        return {name: str(metric(truth, preds)) for name, metric in METRICS.items()}

    def save_results(self):
        """Saving the experiment results to a certain folder"""
        results_dict: Dict = {
            "dataset": self.data_dir,
            "algorithm": self.algorithm_name,
            "training": self.generate_metrics(self.y_tr, self.y_tr_pred),
            "testing": self.generate_metrics(self.y_te, self.y_te_pred),
        }
        self.save_dict_to_json(results_dict)

    def save_dict_to_json(self, j_dict: Dict):
        """Save a Python results dictionary to a JSON file"""
        file_name = f"{RESULTS_DIR}/{self.tag}.json"
        with open(file_name, "w") as fid:
            json.dump(j_dict, fid)

        print(f"\n Results have been saved to: {file_name}")
        print(j_dict)

    def save_model(self):
        """Save trained model to local storage"""
        file_name = f"{MODELS_DIR}/{self.tag}.pkl"
        with open(file_name, "wb") as fid:
            pickle.dump(self.trained_model, fid)

        print(f"\n Model has been saved to: {file_name}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default="baseline",
        help="Provide the subdirectory name for the training/split datasets.",
    )

    parser.add_argument(
        "--algo",
        type=str,
        default="Logistic",
        help="Provide a specific algorithm you wish to fit to the data.",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="baseline",
        help="Provide the tag name associated with this experiment.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    experiment = Experiment(args.data, args.algo, args.tag)
    experiment.run()
