# from real_estate.logger import logging
# from real_estate.exception import realEstateException
# import sys
 

from real_estate.components.data_ingestion import TrainPipeline


if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = TrainPipeline()

    # Run the pipeline
    pipeline.run_pipeline()

# from real_estate.utils.main_utils import load_object
# preprocessor = load_object(file_path="artifacts/ingested/processor.pkl")
# print(f"Loaded preprocessor type: {type(preprocessor)}")

