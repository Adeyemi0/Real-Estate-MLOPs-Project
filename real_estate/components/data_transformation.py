import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  
from real_estate.exception import realEstateException
from dataclasses import dataclass
from real_estate.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = "artifacts/ingested/processor.pkl"
    train_data_path: str = "artifacts/ingested/train_data.csv"
    test_data_path: str = "artifacts/ingested/test_data.csv"
    dataset_path: str = "artifacts/feature_store/real_estate.csv"

class DataTransformation:
    def __init__(self, dataset_path="artifacts/feature_store/real_estate.csv"):
        self.dataset_path = dataset_path
        self.config = DataTransformationConfig()
        self.target_column = 'price'
        self.input_features = ['days_on_zillow', 'zestimate', 'rent_zestimate', 'area', 'beds', 'baths', 'price_change', 
                               'tax_assessed_value', 'lot_area_value', 'home_type']
        
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            # Set the dataset paths to the provided train and test paths
            self.train_data_path = train_data_path
            self.test_data_path = test_data_path
            
            # Apply transformations and return the processed data
            train_arr, test_arr, preprocessor_path = self.apply_transformations()

            return train_arr, test_arr, preprocessor_path
        except Exception as e:
            raise realEstateException(f"Error in initiate_data_transformation: {e}")
        
    def load_data(self):
        try:
            df = pd.read_csv(self.dataset_path)
            return df
        except Exception as e:
            raise realEstateException(f"Error loading data: {e}")
        
    def filter_and_clean_data(self, df):
        try:

            # Filter for sales listings only (if applicable to your data)
            df = df[df['listing_type'] == 'sales']
            df.drop('listing_type', axis=1, inplace=True)

            # Filter columns to keep only the required input features
            df = df[self.input_features + [self.target_column]]
            
            # Drop duplicates based on essential columns
            df_cleaned = df.drop_duplicates(subset=self.input_features).reset_index(drop=True)
            
            # Replace "NaN" strings with actual NaN values and fill remaining missing values with empty strings
            df_cleaned.replace("NaN", pd.NA, inplace=True)
            df_cleaned.fillna("", inplace=True)
            
            return df_cleaned
        except Exception as e:
            raise realEstateException(f"Error in data filtering and cleaning: {e}")

    def convert_numeric_columns(self, df):
        try:
            # Convert relevant columns to numeric types, ignoring errors
            numeric_columns = ['zestimate', 'rent_zestimate', 'beds', 'baths', 'price_change', 
                               'tax_assessed_value', 'lot_area_value', 'area']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            return df
        except Exception as e:
            raise realEstateException(f"Error converting numeric columns: {e}")
        

    def remove_outliers_iqr(self, df, column):
        try:
            # Remove outliers using the IQR method
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
            return df
        except Exception as e:
            raise realEstateException(f"Error removing outliers: {e}")
        

    def encode_and_scale(self, df):
        try:
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
            
            # # Apply mapping to 'home_type' instead of Label Encoding
            df['home_type'] = df['home_type'].map(home_type_mapping)
            
            df.fillna(0, inplace=True)
            
            return df
        except Exception as e:
            raise realEstateException(f"Error encoding and scaling: {e}")
        

    def apply_transformations(self):
        try:
            df = self.load_data()
            
            # Apply filtering, cleaning, and transformations only on the selected columns
            df = self.filter_and_clean_data(df)
            df = self.convert_numeric_columns(df)
            
            # Log the columns after filtering and cleaning
            logging.info(f"Columns after filtering and cleaning: {df.columns.tolist()}")
            
            # Remove outliers from the specified numeric columns
            for column in ['area', 'lot_area_value', 'zestimate', 'rent_zestimate', 'tax_assessed_value']:
                df = self.remove_outliers_iqr(df, column)
                
            # Encode and scale the data
            df = self.encode_and_scale(df)

            # Split the data into train and test sets
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            # Log the final input features and target variable
            logging.info(f"Final input features: {train_df.drop(columns=[self.target_column]).columns.tolist()}")

            # Separate input features and target variable
            input_feature_train_df = train_df.drop(columns=[self.target_column])
            target_feature_train_df = train_df[self.target_column]
            input_feature_test_df = test_df.drop(columns=[self.target_column])
            target_feature_test_df = test_df[self.target_column]

            # Preprocess using StandardScaler
            scaler = StandardScaler()
            input_feature_train_arr = scaler.fit_transform(input_feature_train_df)
            input_feature_test_arr = scaler.transform(input_feature_test_df)

            # Concatenate processed input features and target columns
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.to_numpy()]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df.to_numpy()]

            # Save the preprocessing object
            joblib.dump(scaler, self.config.preprocessor_obj_file_path)

            # Save processed train and test data to CSV
            columns = self.input_features + [self.target_column]
            pd.DataFrame(train_arr, columns=columns).to_csv(self.config.train_data_path, index=False)
            pd.DataFrame(test_arr, columns=columns).to_csv(self.config.test_data_path, index=False)

            return train_arr, test_arr, self.config.preprocessor_obj_file_path
        except Exception as e:
            raise realEstateException(f"Error in apply_transformations: {e}")
