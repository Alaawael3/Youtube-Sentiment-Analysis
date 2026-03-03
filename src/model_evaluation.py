import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)
from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pickle
import yaml
import logging
import os
import json

# logging configuration
logger = logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("model_evaluation_errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        df.dropna(inplace=True)  # Drop rows with NaN values
        logger.debug("Data loaded and NaNs dropped from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise


def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        logger.debug("Model loaded from %s", model_path)
        return model
    except Exception as e:
        logger.error("Error loading model from %s: %s", model_path, e)
        raise


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, "rb") as file:
            vectorizer = pickle.load(file)
        logger.debug("TF-IDF vectorizer loaded from %s", vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error("Error loading vectorizer from %s: %s", vectorizer_path, e)
        raise


def evaluate_model(model, x_test, y_test):
    """Evaluate the model and log classification metrics and confusion matrix."""
    try:
        y_pred = model.predict(x_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        logger.debug("Model evaluation completed")

        return report, cm

    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise


def mlflow_log_metrics(report, cm, data_name="test_data") -> None:
    """Log classification metrics and confusion matrix to MLflow."""
    try:
        mlflow.log_metrics(
            {
                "accuracy": report["accuracy"],
                "precision": report["weighted avg"]["precision"],
                "recall": report["weighted avg"]["recall"],
                "f1_score": report["weighted avg"]["f1-score"],
            }
        )

        mlflow.log_artifact("confusion_matrix.png")

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        cm_file_path = f"confusion_matrix_{data_name}.png"
        plt.savefig(cm_file_path)
        mlflow.log_artifact(cm_file_path)

        logger.debug("Metrics and confusion matrix logged to MLflow")

    except Exception as e:
        logger.error("Error logging metrics to MLflow: %s", e)


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save model information to a YAML file."""
    try:
        model_info = {"run_id": run_id, "model_path": model_path}
        with open(file_path, "w") as file:
            json.dump(model_info, file)
        logger.debug("Model information saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving model information to %s: %s", file_path, e)
        raise


def main():
    mlflow.set_tracking_uri(
        "http://ec2-16-171-25-99.eu-north-1.compute.amazonaws.com:5000/"
    )
    mlflow.set_experiment("dvc-pipeline-runs")

    with mlflow.start_run() as run:
        try:
            params = load_params(os.path.join(ROOT_DIR, "params.yaml"))

            for key, value in params.items():
                mlflow.log_param(key, value)

            test_data = load_data(os.path.join(ROOT_DIR, "data/processed/test.csv"))

            model = load_model(os.path.join(ROOT_DIR, "model/lgbm_model.pkl"))
            vectorizer = load_vectorizer(
                os.path.join(ROOT_DIR, "model/tfidf_vectorizer.pkl")
            )

            x_test = vectorizer.transform(test_data["clean_comment"])
            y_test = test_data["category"].values

            # MLflow needs a concrete example of what input looks like when someone loads the model later.
            input_example = pd.DataFrame(
                x_test[:5].toarray(), columns=vectorizer.get_feature_names_out()
            )
            output_example = infer_signature(input_example, model.predict(x_test[:5]))

            mlflow.sklearn.log_model(
                model,
                "lgbm_model",
                signature=output_example,
                input_example=input_example,
            )

            # Save model info
            # artifact_uri = mlflow.get_artifact_uri()
            model_path = "lgbm_model"
            save_model_info(run.info.run_id, model_path, "experiment_info.json")

            # Log the vectorizer as an artifact
            mlflow.log_artifact(os.path.join(ROOT_DIR, "model/tfidf_vectorizer.pkl"))

            report, cm = evaluate_model(model, x_test=x_test, y_test=y_test)

            mlflow_log_metrics(report=report, cm=cm)

            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
