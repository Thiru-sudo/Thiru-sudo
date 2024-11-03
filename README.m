!pip install catboost
!pip install "dask[dataframe]"
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor, BaggingRegressor, GradientBoostingRegressor
import warnings
from google.colab import files
warnings.filterwarnings('ignore')  # Ignore warnings for cleaner output
import time
from scipy.stats import spearmanr

# Function to upload and load the CSV file
def upload_and_load():
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]
    df = pd.read_csv(file_name)
    return df

# Function to perform data imputation testing and select the best method
def test_imputation_methods(df, required_columns):
    numeric_df = df.select_dtypes(include=[np.number])

    imputation_methods = {
        'Mean+Forward Fill': df.fillna(numeric_df.mean()).fillna(method='ffill'),
        'Linear Interpolation+KNN': df.interpolate().fillna(pd.DataFrame(KNNImputer().fit_transform(numeric_df), columns=numeric_df.columns, index=df.index)),
        'ARIMA+Forward Fill': df.interpolate(method='linear').fillna(method='ffill'),
        'Mean+Spline Interpolation': df.fillna(numeric_df.mean()).interpolate(method='spline', order=2)
    }

    best_method, best_df = min(imputation_methods.items(), key=lambda x: x[1][required_columns].isna().sum().sum())
    print(f"Best imputation method selected: {best_method}")
    return best_df

# Function to perform normalization testing and select the best method
def test_normalization_methods(X):
    # Small constant to avoid log of zero
    epsilon = 1e-6

    # Define normalization methods with preprocessed X to avoid errors
    X_log_safe = np.log1p(np.clip(X, epsilon, None))  # Clip values to avoid log of zero or negative values

    normalization_methods = {
        'MinMax+Log': MinMaxScaler().fit_transform(X_log_safe),
        'Z-Score+Robust': RobustScaler().fit_transform(StandardScaler().fit_transform(X)),
        'Log+Z-Score': StandardScaler().fit_transform(X_log_safe),
        'MaxAbs+MinMax': MinMaxScaler().fit_transform(MaxAbsScaler().fit_transform(X))
    }

    # Select the best normalization method based on standard deviation
    best_method, best_X = min(normalization_methods.items(), key=lambda x: np.std(x[1]))
    print(f"Best normalization method selected: {best_method}")
    return best_X

# Main function to process data and train models
def load_dataset():
    df = upload_and_load()  # Assuming the upload_and_load function loads the dataset

    required_columns = ['CampusKey', 'SiteKey', 'Timestamp', 'SolarGeneration']

    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns in the dataset: {missing_columns}")
        return None, None, None, None, None

    # Drop rows with NaN values in required columns
    df = df.dropna(subset=required_columns)

    # Ensure SolarGeneration column is numeric and impute missing values
    df['SolarGeneration'] = pd.to_numeric(df['SolarGeneration'], errors='coerce')
    df = test_imputation_methods(df, required_columns)

    # Convert Timestamp to datetime and extract basic time features
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    df['year'] = df['Timestamp'].dt.year
    df['month'] = df['Timestamp'].dt.month
    df['day'] = df['Timestamp'].dt.day
    df['hour'] = df['Timestamp'].dt.hour
    df['minute'] = df['Timestamp'].dt.minute
    df['second'] = df['Timestamp'].dt.second

    # Define features and target variable (without feature engineering)
    X = df[[ 'month', 'day', 'hour', 'minute']]
    y = df['SolarGeneration']

    # Normalize features and split the data
    X = test_normalization_methods(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, df


# Initialize models dictionary
def get_models():
    return {
        'Linear Regression': LinearRegression(),
        'Lasso': Lasso(),
        'Ridge': Ridge(),
        'ElasticNet': ElasticNet(),
        'SVR': SVR(),
        'Decision Tree': DecisionTreeRegressor(),
        'KNN': KNeighborsRegressor(),
        'RandomForest': RandomForestRegressor(),
        'GradientBoosting': GradientBoostingRegressor(),
        'AdaBoost': AdaBoostRegressor(),
        'XGBoost': XGBRegressor(),
        'LightGBM': LGBMRegressor(),
        'CatBoost': CatBoostRegressor(verbose=0),
    }


# Function to calculate Index of Agreement (IA)
def ia(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((np.abs(y_true - np.mean(y_true)) + np.abs(y_pred - np.mean(y_true))) ** 2)
    return 1 - (numerator / denominator)

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Evaluation function for model metrics
def evaluate_model(model, X_train, X_test, y_train, y_test):
   # Start time measurement
    start_time = time.time()

    # Fit the model
    model.fit(X_train, y_train)

    # Stop time measurement
    end_time = time.time()

    # Calculate training time
    time_taken = end_time - start_time  # in seconds
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return {
        'MAE': mean_absolute_error(y_test, y_pred_test),
        'MSE': mean_squared_error(y_test, y_pred_test),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'R^2': r2_score(y_test, y_pred_test),
         'IA': ia(y_test, y_pred_test),  # Ensure IA function is correctly defined
        'Training Time (s)': time_taken  # Now time_taken is defined
    }

# Parameter tuning with hybrid search (random, grid, or both)
def tune_model(model, X_train, y_train, X_test, y_test, search_type):
    metrics = {}
    # Define parameter grids
    param_grids = {
       LinearRegression: {
        'fit_intercept': [True, False],
        'positive': [True, False],
    },
    SVR: {
        'C': [0.001, 0.1, 1],                  # A reduced set of C values for a reasonable range
        'epsilon': [0.1, 0.5, 1],              # Only a few options to control the margin of tolerance
        'kernel': ['rbf', 'linear'],        # Focus on faster kernels
        'gamma': ['scale', 'auto', 0.1],    # Reduce gamma options to fewer values
    },
    Ridge: {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
        'max_iter': [500, 1000, 5000],  # Adjust maximum iterations for convergence
        'tol': [1e-4, 1e-3, 1e-2],       # Tolerance for convergence
    },
    Lasso: {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
        'max_iter': [1000, 5000, 10000],  # More iterations to help Lasso converge
        'tol': [1e-4, 1e-3, 1e-2],
        'selection': ['cyclic', 'random'],  # Selection method
    },
    ElasticNet: {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
        'l1_ratio': [0.1, 0.25, 0.5, 0.75, 1],
        'max_iter': [1000, 5000, 10000],
        'tol': [1e-4, 1e-3, 1e-2],
        'selection': ['cyclic', 'random'],
    },
        DecisionTreeRegressor: {'max_depth': [10, 20, 30, 40, 50], 'min_samples_split': [10, 20, 30]},
        KNeighborsRegressor: {'n_neighbors': [3, 6, 9, 12, 15], 'weights': ['distance']},
        RandomForestRegressor: {'n_estimators': [50, 100, 150, 200], 'max_depth': [10, 20, 30]},
        GradientBoostingRegressor: {'n_estimators': [50, 100, 150, 200], 'learning_rate': [0.01, 0.1, 1], 'max_depth': [3, 6, 9, 12]},
        XGBRegressor: {'n_estimators': [50, 100, 150, 200], 'learning_rate': [0.01, 0.1, 1], 'max_depth': [3, 6, 9, 12]},
        AdaBoostRegressor: {'n_estimators': [50, 100, 150, 200], 'learning_rate': [0.01, 0.1, 1]},
        LGBMRegressor: {'n_estimators': [50, 100, 150, 200, 250, 300], 'learning_rate': [0.01, 0.1, 0.3], 'max_depth': [4, 6, 8, 10]},
        CatBoostRegressor: {'iterations': [50, 100, 150, 200], 'learning_rate': [0.01, 0.1, 1], 'depth': [3, 6, 9, 12]},
    }

     # Get the parameter grid based on the model type
    param_grid = param_grids.get(type(model), {})

    if search_type == "random":
        search = RandomizedSearchCV(model, param_grid, cv=3, n_iter=3, scoring='r2', random_state=42)
        search.fit(X_train, y_train)
        metrics = evaluate_model(search.best_estimator_, X_train, X_test, y_train, y_test)

    elif search_type == "grid":
        search = GridSearchCV(model, param_grid, cv=3, scoring='r2')
        search.fit(X_train, y_train)
        metrics = evaluate_model(search.best_estimator_, X_train, X_test, y_train, y_test)

    else:  # Hybrid
        random_search = RandomizedSearchCV(model, param_grid, cv=3, n_iter=3, scoring='r2', random_state=42)
        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_

        # Narrow down the grid search around best params
        search = GridSearchCV(model, {k: [best_params[k]] for k in best_params}, cv=3, scoring='r2')
        search.fit(X_train, y_train)
        metrics = evaluate_model(search.best_estimator_, X_train, X_test, y_train, y_test)

    print(f"Best parameters for {type(model).__name__}: {search.best_params_}")
    return metrics


# Tuning and evaluation with CSV download
def tune_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = {
        'Grid Search': {},
        'Random Search': {},
        'Hybrid Search': {}
    }

    for name, model in models.items():
        print(f"Tuning {name} with Random Search...")
        results['Random Search'][name] = tune_model(model, X_train, y_train, X_test, y_test, search_type="random")

        print(f"Tuning {name} with Grid Search...")
        results['Grid Search'][name] = tune_model(model, X_train, y_train, X_test, y_test, search_type="grid")

        print(f"Tuning {name} with Hybrid Search...")
        results['Hybrid Search'][name] = tune_model(model, X_train, y_train, X_test, y_test, search_type="hybrid")

    # Create DataFrames for each search type
    for search_type in results.keys():
        results_df = pd.DataFrame(results[search_type]).transpose()
        results_df.to_csv(f"{search_type}_tuning_metrics.csv")
        # Download CSV file in Colab
        # Uncomment the line below if running in a Colab environment
        files.download(f"{search_type}_tuning_metrics.csv")

    return results

# Spearman correlation function
def spearman_correlation(X, y):
    correlations = []
    for col in X.columns:
        corr, _ = spearmanr(X[col], y)
        correlations.append(corr)
    return pd.Series(correlations, index=X.columns)

# Function to plot Spearman correlation matrix
def plot_correlation_matrix(X, y, feature_names, target_name):
    # Combine features and target into a DataFrame
    combined_df = pd.concat([X, y], axis=1)
    spearman_corr = combined_df.corr(method='spearman')

    plt.figure(figsize=(10, 8))
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Spearman Correlation Matrix')
    plt.show()


# Main function
def main():
    # Load and preprocess the dataset
    X_train, X_test, y_train, y_test, df = load_dataset()
    if X_train is None or y_train is None:
        print("Data loading failed due to missing columns or invalid data.")
        return

    # Convert X_train to DataFrame if necessary
    feature_names = ['month', 'day', 'hour', 'minute']  # Replace with actual names
    X_train = pd.DataFrame(X_train, columns=feature_names)
    target_name = 'SolarGeneration'

    # Plot Spearman correlation matrix
    plot_correlation_matrix(X_train, y_train, feature_names, target_name)

    # Display Spearman correlation of features with target variable
    print("Calculating Spearman correlation with target variable...")
    spearman_corr = spearman_correlation(X_train, y_train)
    print("Spearman Correlation:\n", spearman_corr)

    # Initialize models
    models = get_models()

    # Tune and evaluate models
    print("Tuning and evaluating models...")
    results_df = tune_and_evaluate(models, X_train, X_test, y_train, y_test)
    print("Results:\n", results_df)

# Execute the main function
main()
