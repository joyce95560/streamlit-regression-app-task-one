import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline model
# Ensure this path is correct relative to where you deploy your Streamlit app
@st.cache_resource
def load_model():
    try:
        model = joblib.load('linear_regression_pipeline_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file 'linear_regression_pipeline_model.joblib' not found. Please ensure it's in the same directory.")
        st.stop()

pipeline = load_model()

st.title('House Price Prediction App')
st.write('Enter house features to predict its sale price.')

# Collect user input for features
# This is a simplified example; you'll need to add all features your model expects
# It's crucial that the input features match the features the model was trained on.

# Example of how to get input for numerical features
mssubclass = st.number_input('MSSubClass (e.g., 60 for 2-STORY 1946 & Newer)', min_value=20, max_value=190, value=60)
lotarea = st.number_input('LotArea (square feet)', min_value=1000, max_value=200000, value=8450)
overallqual = st.slider('OverallQual (Overall material and finish quality)', min_value=1, max_value=10, value=7)
overallcond = st.slider('OverallCond (Overall condition rating)', min_value=1, max_value=9, value=5)
yearbuilt = st.number_input('YearBuilt', min_value=1800, max_value=2023, value=2003)
yearremodadd = st.number_input('YearRemodAdd (Remodel year if different from construction year)', min_value=1950, max_value=2023, value=2003)
grlinarea = st.number_input('GrLivArea (Above grade (ground) living area square feet)', min_value=300, max_value=6000, value=1710)
garagecars = st.slider('GarageCars (Size of garage in car capacity)', min_value=0, max_value=4, value=2)
garagearea = st.number_input('GarageArea (Size of garage in square feet)', min_value=0, max_value=1500, value=548)

# For simplicity, we'll assume default/median values for other features for this example
# In a real app, you'd collect input for all relevant features or use sensible defaults

# Create a dictionary for prediction input
# You MUST include all columns that were in your X_train, even if you are using default values
# The column names and their order must match what the pipeline expects.
input_data = {
    'MSSubClass': mssubclass,
    'MSZoning': 'RL', # Example: assuming default 'RL'
    'LotFrontage': 65.0, # Example: using a typical value
    'LotArea': lotarea,
    'Street': 'Pave', # Example: assuming default 'Pave'
    'Alley': 'None', # Example: imputed 'None'
    'LotShape': 'Reg',
    'LandContour': 'Lvl',
    'Utilities': 'AllPub',
    'LotConfig': 'Inside',
    'LandSlope': 'Gtl',
    'Neighborhood': 'CollgCr',
    'Condition1': 'Norm',
    'Condition2': 'Norm',
    'BldgType': '1Fam',
    'HouseStyle': '2Story',
    'OverallQual': overallqual,
    'OverallCond': overallcond,
    'YearBuilt': yearbuilt,
    'YearRemodAdd': yearremodadd,
    'RoofStyle': 'Gable',
    'RoofMatl': 'CompShg',
    'Exterior1st': 'VinylSd',
    'Exterior2nd': 'VinylSd',
    'MasVnrType': 'None', # Example: imputed 'None'
    'MasVnrArea': 0.0, # Example: imputed 0.0
    'ExterQual': 'Gd',
    'ExterCond': 'TA',
    'Foundation': 'PConc',
    'BsmtQual': 'TA', # Example: imputed 'TA'
    'BsmtCond': 'TA', # Example: imputed 'TA'
    'BsmtExposure': 'No', # Example: imputed 'No'
    'BsmtFinType1': 'Unf', # Example: imputed 'Unf'
    'BsmtFinSF1': 0,
    'BsmtFinType2': 'Unf', # Example: imputed 'Unf'
    'BsmtFinSF2': 0,
    'BsmtUnfSF': 0,
    'TotalBsmtSF': 0,
    'Heating': 'GasA',
    'HeatingQC': 'Ex',
    'CentralAir': 'Y',
    'Electrical': 'SBrkr',
    '1stFlrSF': 856,
    '2ndFlrSF': 854,
    'LowQualFinSF': 0,
    'GrLivArea': grlinarea,
    'BsmtFullBath': 1,
    'BsmtHalfBath': 0,
    'FullBath': 2,
    'HalfBath': 1,
    'BedroomAbvGr': 3,
    'KitchenAbvGr': 1,
    'KitchenQual': 'Gd',
    'TotRmsAbvGrd': 8,
    'Functional': 'Typ',
    'Fireplaces': 0,
    'FireplaceQu': 'None', # Example: imputed 'None'
    'GarageType': 'Attchd', # Example: imputed 'Attchd'
    'GarageYrBlt': yearbuilt,
    'GarageFinish': 'RFn', # Example: imputed 'RFn'
    'GarageCars': garagecars,
    'GarageArea': garagearea,
    'GarageQual': 'TA', # Example: imputed 'TA'
    'GarageCond': 'TA', # Example: imputed 'TA'
    'PavedDrive': 'Y',
    'WoodDeckSF': 0,
    'OpenPorchSF': 0,
    'EnclosedPorch': 0,
    '3SsnPorch': 0,
    'ScreenPorch': 0,
    'PoolArea': 0,
    'PoolQC': 'None', # Example: imputed 'None'
    'Fence': 'None', # Example: imputed 'None'
    'MiscFeature': 'None', # Example: imputed 'None'
    'MiscVal': 0,
    'MoSold': 2,
    'YrSold': 2008,
    'SaleType': 'WD',
    'SaleCondition': 'Normal'
}

# Create a DataFrame from the input data
# Ensure the order of columns matches the training data
input_df = pd.DataFrame([input_data])

if st.button('Predict Sale Price'):
    try:
        # Ensure 'Id' column is handled correctly if it was in X_train but not used for prediction
        # For this example, we assume 'Id' was dropped in preprocessing or is added before prediction
        # If 'Id' was part of X_train for the pipeline, you might need to add a dummy 'Id' here.
        # Given the pipeline structure, it should handle columns it was trained on.

        # If the pipeline expects 'Id' but it's not a feature for prediction,
        # we need to ensure its presence for the ColumnTransformer and then that the regressor ignores it.
        # A simpler approach is to ensure 'Id' is dropped from X BEFORE the pipeline is defined if it's not a feature.
        # In our case, the pipeline was defined to exclude 'Id' from numerical_cols_for_pipeline, so it should be fine.

        prediction = pipeline.predict(input_df)
        st.success(f'The predicted Sale Price is: ${prediction[0]:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.write("\n\n--- Developed with Streamlit --- ")
