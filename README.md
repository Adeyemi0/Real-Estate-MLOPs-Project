# Real Estate Price Prediction Pipeline

This repository contains an end-to-end automated pipeline for predicting real estate prices based on Zillow data, developed to provide data-driven insights for property value assessment.

## Objective

Create an automated system for accurate property price prediction, enabling stakeholders to make informed real estate decisions.

## Key Components

- **Data Scraper**: A web scraper was developed and deployed on Google Virtual Machine (VM) to extract property listing data from Zillow, capturing over 25,000 sales listings within a month.
- **Data Storage**: Scraped data is stored in Google BigQuery for secure, scalable access and future analysis.
- **Model Training**: Multiple models were trained, and hyperparameter tuning was applied using GridSearchCV to improve accuracy, achieving a best R² score of 0.92.
- **API Development**: A client-facing API was developed using Flask to provide real-time predictions based on current listing data.
- **Deployment**: The model was deployed on Render, ensuring reliable, scalable access for real-time predictions.
- **Outcome**: The pipeline provides actionable insights into real estate trends, allowing clients to make more accurate, data-backed property valuations.

## Deployment

Access the deployed application at: [Real Estate Price Prediction App](https://zillow-ml.onrender.com/)

## Resources Used

- **Python Version**: 3.8
- **Packages**:
- **For Web Framework Requirements**: pip install -r requirements.txt
- **Scraper Github**: https://github.com/Adeyemi0/zillow_scraper

## Data Schema

The following fields are extracted from Zillow:

- `address`
- `days_on_zillow`
- `zestimate`
- `rent_zestimate`
- `longitude`
- `latitude`
- `area`
- `price`
- `img_src`
- `beds`
- `baths`
- `price_change`
- `tax_assessed_value`
- `lot_area_value`
- `home_type`
- `living_area`
- `detail_url`
- `listing_type`
- `scraped_date`

## Data Processing and Feature Engineering

```plaintext
Data Cleaning:
    - Removed rows with 0 beds or baths
    - Removed duplicate entries
    - Removed outliers
    - Filled missing values

Feature Engineering:
    - Created a new column `city`
    - Encoded categorical variables
    - Scaled numerical data for improved model performance
```

## Model Performance
Models were tested with the following R² scores:
```plaintext
- Decision Tree Regressor: 0.864
- Gradient Boosting Regressor: 0.796
- Linear Regression: 0.352
- XGBRegressor: 0.917
- CatBoost Regressor: 0.909
- AdaBoost Regressor: 0.492
```
## Productization

The model is accessible through a Flask API, enabling real-time property price predictions. Deployed on Render, it provides reliable, scalable access for users.
