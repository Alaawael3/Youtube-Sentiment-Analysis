import pandas as pd
import logging
import yaml
import pickle
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# logging configuration
logger = logging.getLogger("model_building")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("model_building_errors.log")
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
        df = pd.read_csv(file_path)
        df.dropna(inplace=True) # Drop rows with NaN values
        logger.debug("Data loaded and NaNs dropped from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise


def apply_vectorization(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int = 5000, ngram: tuple = (1, 1)) -> tuple:
    """Apply TF-IDF vectorization to the text data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram)
        X_train = vectorizer.fit_transform(train_data['clean_comment'])
        X_test = vectorizer.transform(test_data['clean_comment'])
        logger.debug("TF-IDF vectorization applied to the text data")
        
        with open(os.path.join(ROOT_DIR, "model/tfidf_vectorizer.pkl"), "wb") as f:
            pickle.dump(vectorizer, f)
        logger.debug("TF-IDF vectorizer saved successfully to ../model/tfidf_vectorizer.pkl")
        return X_train, X_test
    except Exception as e:
        logger.error("Error during TF-IDF vectorization: %s", e)
        raise


def train_model(x_train, y_train, model) :
    try:
        # model = model(**param)
        model.fit(x_train, y_train)
        logger.debug("Model trained successfully with parameters: %s", model.get_params())
        return model
    except Exception as e:
        logger.error("Error during model training: %s", e)
        raise


def save_model(model, model_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug("Model saved successfully to %s", model_path)
    except Exception as e:
        logger.error("Error saving the model: %s", e)
        raise


def main():
    try:
        logger.debug("Starting model building process...")
        
        # Load parameters
        params = load_params(os.path.join(ROOT_DIR, "params.yaml"))
        
        # Load data
        train_data = load_data(os.path.join(ROOT_DIR, "data/processed/train.csv"))
        test_data = load_data(os.path.join(ROOT_DIR, "data/processed/test.csv"))
        
        # Apply vectorization
        max_features = params['vectorization']['max_features']
        ngram_range = tuple(params['vectorization']['ngram_range'])
        X_train, X_test = apply_vectorization(train_data, test_data, max_features=max_features, ngram=ngram_range)
        y_train = train_data['category'].values
        
        # Train model
        model_params = params['model']
        model = LGBMClassifier(**model_params)
        model = train_model(X_train, y_train, model)
        model_name = 'lgbm_model'
        
        # Save model
        os.makedirs(os.path.join(ROOT_DIR, "model"), exist_ok=True)
        save_model(model, os.path.join(ROOT_DIR, f"model/{model_name}.pkl"))

    except Exception as e:
        logger.error("Failed to complete the model building process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()