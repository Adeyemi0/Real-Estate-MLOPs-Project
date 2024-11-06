import os
import sys
from dataclasses import dataclass
import pickle  # For loading the saved model and features

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import pandas as pd
from sklearn.compose import ColumnTransformer
from real_estate.exception import realEstateException
from real_estate.logger import logging
from real_estate.utils.main_utils import save_object, evaluate_models, load_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "trained_model", "model.pkl")
    top_features_file_path = os.path.join("artifacts", "trained_model", "top_features.pkl")
    preprocessor_file_path = os.path.join("artifacts", "ingested", "top_features_processor.pkl")  # Path for modified preprocessor

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Load the modified preprocessor with only top features
            preprocessor = load_object(self.model_trainer_config.preprocessor_file_path)
            
            # Apply the preprocessor to the training and testing data
            X_train = preprocessor.transform(X_train)
            X_test = preprocessor.transform(X_test)

            # Initialize models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=0),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Define parameters for each model (unchanged)
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # DataFrame to store feature importances
            feature_importances = pd.DataFrame(index=[f'feature_{i}' for i in range(X_train.shape[1])])

            # Dictionary to store top 8 features for each model
            top_8_features = {}

            # Train each model and collect feature importance (if available)
            for model_name, model in models.items():
                logging.info(f"Training {model_name}")
                model.fit(X_train, y_train)

                # If the model has feature importances, capture the top 8
                if hasattr(model, 'feature_importances_'):
                    feature_importances[model_name] = model.feature_importances_
                    top_8 = feature_importances[model_name].nlargest(8).index.tolist()
                    top_8_features[model_name] = [int(i.split('_')[1]) for i in top_8]
                    logging.info(f"Top 8 features for {model_name}: {top_8}")
                else:
                    logging.warning(f"{model_name} does not support feature importances.")
                    top_8_features[model_name] = None

            # Save the top features for the best model
            best_model_name = max(top_8_features, key=lambda k: (top_8_features[k] is not None))
            with open(self.model_trainer_config.top_features_file_path, 'wb') as f:
                pickle.dump(top_8_features[best_model_name], f)

            # Evaluate all models and store results
            model_report = {}

            for model_name, model in models.items():
                top_features = top_8_features.get(model_name)

                if top_features is not None:
                    X_train_selected = X_train[:, top_features]
                    X_test_selected = X_test[:, top_features]
                else:
                    X_train_selected = X_train
                    X_test_selected = X_test

                # Evaluate the model
                report = evaluate_models(X_train_selected, y_train, X_test_selected, y_test, 
                                         models={model_name: model}, param=params.get(model_name))
                model_report.update(report)

            # Get the best model based on score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise realEstateException("No best model found")

            logging.info(f"Best found model using top features: {best_model_name}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test_selected)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise realEstateException(e, sys)

# Function to load the model and make predictions using only the top features
def predict_with_top_features(input_data):
    # Load the trained model
    with open(ModelTrainerConfig.trained_model_file_path, 'rb') as f:
        model = pickle.load(f)

    # Load the top features
    with open(ModelTrainerConfig.top_features_file_path, 'rb') as f:
        top_features = pickle.load(f)

    # Prepare the input data (ensure it has the same order as the training data)
    input_data_selected = input_data.iloc[:, top_features]

    # Make predictions
    predictions = model.predict(input_data_selected)
    return predictions
