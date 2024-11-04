import os
import sys
import pandas as pd
from google.cloud import bigquery
from real_estate.exception import realEstateException
from real_estate.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from real_estate.components.data_transformation import DataTransformation
from real_estate.components.model_trainer import ModelTrainer

from real_estate.components.data_transformation import DataTransformation
from real_estate.components.data_transformation import DataTransformationConfig

from real_estate.components.model_trainer import ModelTrainerConfig
from real_estate.components.model_trainer import ModelTrainer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set Google Cloud credentials from environment
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if credentials_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
else:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")




@dataclass
class DataIngestionConfig:
    feature_store_path: str = os.path.join('artifacts', 'feature_store', "real_estate.csv")
    train_data_path: str = os.path.join('artifacts', 'ingested', "train.csv")
    test_data_path: str = os.path.join('artifacts', 'ingested', "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Get BigQuery configurations from environment variables
            BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID")
            BQ_DATASET_ID = os.getenv("BQ_DATASET_ID")
            BQ_TABLE_ID = os.getenv("BQ_TABLE_ID")

            
            # Read data from BigQuery
            query = f"SELECT * FROM {BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}"
            df = pd.read_gbq(query, project_id=BQ_PROJECT_ID)

            logging.info("Fetched the dataset from BigQuery as dataframe")

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.feature_store_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the full dataset to feature_store as real_estate.csv
            df.to_csv(self.ingestion_config.feature_store_path, index=False, header=True)
            logging.info("Saved the full dataset to feature_store/real_estate.csv")

            # Split data into train and test sets
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test data to ingested folder
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise realEstateException(e, sys)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            # Step 1: Data Ingestion
            train_data, test_data = self.data_ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            train_arr, test_arr, _ = self.data_transformation.initiate_data_transformation(train_data, test_data)

            # Step 3: Model Training
            model_result = self.model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info("Pipeline execution completed successfully")
            return model_result

        except Exception as e:
            raise realEstateException(e, sys)
