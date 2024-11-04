import os
import sys

import numpy as np
import dill
import yaml
from pandas import DataFrame

from real_estate.exception import realEstateException
from real_estate.logger import logging
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import joblib


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        
        # Iterate over models and their corresponding parameters
        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")
            
            # Get parameters for the current model
            para = param.get(model_name, {})
            
            if para:
                logging.info(f"Performing GridSearchCV for {model_name} with params: {para}")
                
                # Initialize GridSearchCV with parameters
                gs = GridSearchCV(model, para, cv=3)
                gs.fit(X_train, y_train)

                # Set the best found parameters
                best_params = gs.best_params_
                logging.info(f"Best params for {model_name}: {best_params}")
                
                model.set_params(**best_params)
            else:
                logging.warning(f"No parameters provided for {model_name}. Using default model.")

            # Train the model with the best parameters or defaults
            model.fit(X_train, y_train)

            # Predict and calculate R2 scores
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            logging.info(f"{model_name} - Train R2: {train_model_score}, Test R2: {test_model_score}")

            # Store test model score in the report
            report[model_name] = test_model_score

        return report

    except Exception as e:
        logging.error(f"Error in evaluate_models: {str(e)}")
        raise realEstateException(e, sys)

    

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise realEstateException(e, sys) from e
    


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise realEstateException(e, sys) from e
    




def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of utils")
    try:
        obj = joblib.load(file_path)
        logging.info("Exited the load_object method of utils")
        return obj
    except Exception as e:
        raise realEstateException(e, sys) from e

    


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise realEstateException(e, sys) from e
    



def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise realEstateException(e, sys) from e




def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise realEstateException(e, sys) from e



def drop_columns(df: DataFrame, cols: list)-> DataFrame:

    """
    drop the columns form a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped
    """
    logging.info("Entered drop_columns methon of utils")

    try:
        df = df.drop(columns=cols, axis=1)

        logging.info("Exited the drop_columns method of utils")
        
        return df
    except Exception as e:
        raise realEstateException(e, sys) from e