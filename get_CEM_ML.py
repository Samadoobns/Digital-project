"""
import os
print(os.getcwd())  
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
#**************************#READ AND SPLIT DATA*********************//
X_full = pd.read_excel(r"C:\Users\samad\OneDrive\Bureau\ml\electrical_motors\data_pn-fm-machines.xlsx")
#print(X_full.head) 
X_full.dropna(axis=0, subset=['CEM'], inplace=True)
y = X_full.CEM
X_full.drop(['CEM','CEM_Numerique_vrai','Erreur','Erreur en %'], axis=1, inplace=True)

#print(machines_data.shape)
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)
print("train set dim",X_train_full.shape)
print("full dim",X_full.shape)

#**************************#PREPROCESSING DATA / MODEL*********************//
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='most_frequent')
# Bundle preprocessing for numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, X_train_full.columns)])

model_1 = RandomForestRegressor(n_estimators=100, random_state=42)
# Bundle preprocessing and modeling code in a pipeline
my_pipline = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model_1)])
# Preprocessing of training data, fit model 
my_pipline.fit(X_train_full, y_train)
preds = my_pipline.predict(X_valid_full)
from sklearn.metrics import r2_score
# Evaluate the model
score = r2_score(y_valid, preds)
print('score1 :', score)
from sklearn.ensemble import GradientBoostingRegressor

model_2 = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=50, random_state=42,verbose=1)

my_pipline = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model_2)])
# Preprocessing of training data, fit model 98//*-
my_pipline.fit(X_train_full, y_train)
preds = my_pipline.predict(X_valid_full)

# Evaluate the model
score = r2_score(y_valid, preds)
print('score2 :', score)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
numerical_features = X_train_full.select_dtypes(include=['number']).columns.tolist()

preprocessor1 = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features)
])

model = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=1000, random_state=42)

my_pipline = Pipeline(steps=[('preprocessor', preprocessor1),
                      ('model', model)])
# Preprocessing of training data, fit model
my_pipline.fit(X_train_full, y_train)
preds = my_pipline.predict(X_valid_full)

# Evaluate the model
score = r2_score(y_valid, preds)
print('score3 :', score)
print(preds - y_valid)