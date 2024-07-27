import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import ipywidgets as widgets
from IPython.display import display

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")

def preprocess_data(data):
    try:
        X = data.drop(['total_runs'], axis=1)
        y = data['total_run ' ]
        venue_encoder = LabelEncoder()
        batting_team_encoder = LabelEncoder()
        bowling_team_encoder = LabelEncoder()
        striker_encoder = LabelEncoder()
        bowler_encoder = LabelEncoder()
        
        X['venue'] = venue_encoder.fit_transform(X['venue'])
        X['batting_team'] = batting_team_encoder.fit_transform(X['batting_team'])
        X['bowling_team'] = bowling_team_encoder.fit_transform(X['bowling_team'])
        X['striker'] = striker_encoder.fit_transform(X['striker'])
        X['bowler'] = bowler_encoder.fit_transform(X['bowler'])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y
    except Exception as e:
        print(f"Error preprocessing data: {e}")

def split_data(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error splitting data: {e}")

def train_model(X_train, y_train):
    try:
        model = RandomForestRegressor()
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10]
        }
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    except Exception as e:
        print(f"Error training model: {e}")
 def evaluate_model(model, X_test, y_test):
    try:
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return mse, mae, r2
    except Exception as e:
        print(f"Error evaluating model: {e}")

def create_widget(model, venue_encoder, batting_team_encoder, bowling_team
