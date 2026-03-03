import pandas as pd
import os
import logging


logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")
if not logger.handlers:
    console_handler = (
        logging.StreamHandler()
    )  # Create a console handler that prints the logs in the terminal
    console_handler.setLevel("DEBUG")

    file_handler = logging.FileHandler(
        "errors.log"
    )  # Create a file handler that writes the logs to a file named 'errors.log'
    file_handler.setLevel("ERROR")

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )  # Create a formatter that specifies the format of the logs, including the timestamp, logger name, log level, and log message -> 2026-02-24 15:30:22,123 - data_ingestion - ERROR - File not found
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)



def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(
            "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
        )
        logger.info(f"Data loaded successfully from {data_url}")
        return data
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing the CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, duplicates, and empty strings."""
    try:
        # Removing missing values
        df.dropna(inplace=True)
        # Removing duplicates
        df.drop_duplicates(inplace=True)
        # Removing empty strings
        df = df[df["clean_comment"].str.strip() != ""]
        logger.info(
            "Data preprocessing completed successfully: Missing values, duplicates, and empty strings removed"
        )
        return df
    except Exception as e:
        logger.error(f"An error occurred while preprocessing data: {e}")
        raise


def save_data(data: pd.DataFrame, data_path: str) -> None:
    """Save the preprocessed data to a CSV file."""
    try:
        data_path = os.path.join(data_path, "raw")
        os.makedirs(data_path, exist_ok=True)
        data.to_csv(os.path.join(data_path, "data.csv"), index=False)
        logger.info(f"Data saved successfully to {data_path}")
        
    except Exception as e:
        logger.error(f"An error occurred while saving data: {e}")
        raise


def main():
    try:
        url = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
        data = load_data(url)
        
        preprocessed_data = preprocess_data(data)
        save_data(preprocessed_data, "data")
        
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        raise


if __name__ == "__main__":
    main()