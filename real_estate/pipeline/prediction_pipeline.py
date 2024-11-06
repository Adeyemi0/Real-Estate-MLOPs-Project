import sys
import pandas as pd
from real_estate.exception import realEstateException
from real_estate.utils.main_utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Paths for model and processor
            model_path = os.path.join("artifacts", "trained_model", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "ingested", "top_features_processor.pkl")
            
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            
            # Mapping dictionary for home_type
            home_type_mapping = {
                'CONDO': 1,
                'SINGLE_FAMILY': 2,
                'TOWNHOUSE': 3,
                'MULTI_FAMILY': 4,
                'MANUFACTURED': 5,
                'LOT': 6,
                'APARTMENT': 7
            }
            
            # Encode 'home_type' using the mapping
            features['home_type'] = features['home_type'].map(home_type_mapping)
            
            # Transform the features using the preprocessor
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise realEstateException(e, sys)

class CustomData:
    def __init__(self, home_type: str, zestimate: float, tax_assessed_value: float, baths: float, area: float, lot_area_value: float, beds: float, days_on_zillow: float, rent_zestimate: float, price_change: float):
        self.home_type = home_type
        self.zestimate = zestimate
        self.tax_assessed_value = tax_assessed_value
        self.baths = baths
        self.area = area
        self.lot_area_value = lot_area_value
        self.beds = beds
        self.days_on_zillow = days_on_zillow
        self.price_change = price_change
        self.rent_zestimate = rent_zestimate

    def get_data_as_data_frame(self):
        try:
            # Arrange the dictionary to match the required feature order
            custom_data_input_dict = {
                "days_on_zillow": [self.days_on_zillow],
                "zestimate": [self.zestimate],
                "rent_zestimate": [self.rent_zestimate],
                "area": [self.area],
                "beds": [self.beds],
                "baths": [self.baths],
                "price_change": [self.price_change],
                "tax_assessed_value": [self.tax_assessed_value],
                "lot_area_value": [self.lot_area_value],
                "home_type": [self.home_type],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise realEstateException(e, sys)
