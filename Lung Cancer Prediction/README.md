# Disease Prediction Analysis - README

## Project Overview

This project performs an end-to-end analysis on a disease-related dataset containing lung cancer prediction data. The analysis includes data cleaning, exploratory data analysis, feature engineering, model development, and deployment preparation.

## Dataset

The dataset (`lcs_synthetic_20000.csv`) contains 20,000 synthetic records with the following features:
- **GENDER**: Patient gender (M/F)
- **AGE**: Patient age
- **SMOKING**: Smoking status (1/2)
- **YELLOW_FINGERS**: Yellow fingers indicator (1/2)
- **ANXIETY**: Anxiety level (1/2)
- **PEER_PRESSURE**: Peer pressure indicator (1/2)
- **CHRONIC DISEASE**: Chronic disease indicator (1/2)
- **FATIGUE**: Fatigue level (1/2)
- **ALLERGY**: Allergy indicator (1/2)
- **WHEEZING**: Wheezing indicator (1/2)
- **ALCOHOL CONSUMING**: Alcohol consumption (1/2)
- **COUGHING**: Coughing indicator (1/2)
- **SHORTNESS OF BREATH**: Shortness of breath indicator (1/2)
- **SWALLOWING DIFFICULTY**: Swallowing difficulty indicator (1/2)
- **CHEST PAIN**: Chest pain indicator (1/2)
- **LUNG_CANCER**: Target variable (YES/NO)

## Project Structure

```
├── disease_prediction.ipynb    # Main Jupyter Notebook with complete analysis
├── app.py                     # Flask API for model deployment
├── final_model.pkl          # Trained model 
├── scaler.pkl              # Feature scaler 
├── lcs_synthetic_20000.csv # Original dataset
└── README.md              # This file
```

## Analysis Summary

### Data Cleaning
- Removed 429 duplicate rows from the original 20,000 records
- Converted categorical variables to numerical format
- No missing values were found in the dataset

### Exploratory Data Analysis
- Performed univariate analysis on all features
- Created correlation matrix to identify feature relationships
- Analyzed bivariate relationships with the target variable
- Generated multiple visualization types including histograms, boxplots, and count plots

### Feature Engineering
- Created a new feature: `AGE_SMOKING_INTERACTION` (AGE × SMOKING)
- Applied feature selection using SelectKBest with f_classif
- Selected top 10 features for model training

### Model Development
Three machine learning algorithms were compared:
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Gradient Boosting Classifier**

#### Model Performance
All models achieved excellent performance metrics:
- **Precision**: 0.869 (exceeds requirement of ≥0.3)
- **Recall**: 1.000 (exceeds requirement of ≥0.3)

The **Logistic Regression** model was selected as the final model based on performance and simplicity.

#### Selected Features
The final model uses these 10 features:
1. GENDER
2. AGE
3. ANXIETY
4. FATIGUE
5. ALLERGY
6. ALCOHOL CONSUMING
7. SHORTNESS OF BREATH
8. SWALLOWING DIFFICULTY
9. CHEST PAIN
10. AGE_SMOKING_INTERACTION

## Usage

### Running the Jupyter Notebook
1. Ensure the dataset file `lcs_synthetic_20000.csv` is in the same directory
2. Open `disease_prediction.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells to reproduce the complete analysis

### Using the Flask API
1. Navigate to the `disease_prediction_app` directory
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Flask app: `python app.py`
4. The API will be available at `http://localhost:5001`

#### API Endpoints
- **GET /**: API information and feature list
- **GET /health**: Health check endpoint
- **POST /predict**: Make predictions

#### Example API Usage
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "GENDER": 1,
    "AGE": 65,
    "ANXIETY": 2,
    "FATIGUE ": 2,
    "ALLERGY ": 1,
    "ALCOHOL CONSUMING": 2,
    "SHORTNESS OF BREATH": 2,
    "SWALLOWING DIFFICULTY": 1,
    "CHEST PAIN": 2,
    "SMOKING": 2
  }'
```

## Dependencies

### Python Packages
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- flask
- flask-cors
- pickle

### Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn flask flask-cors
```

## Model Files

- `final_model.pkl`: Trained Logistic Regression model
- `scaler.pkl`: StandardScaler for feature normalization

## Notes

- The model achieves high precision and recall, indicating excellent performance on this synthetic dataset
- The Flask API is configured for development use; for production deployment, consider using a WSGI server like Gunicorn

