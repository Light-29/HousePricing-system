import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import os

# 1. Load Dataset
# Note: Ensure 'train.csv' is in your working directory
df = pd.read_csv('train.csv')

# 2. Feature Selection (Selecting 6 from the recommended 9)
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
target = 'SalePrice'

X = df[features]
y = df[target]

# 3. Data Preprocessing
# Handling missing values
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# 5. Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluation
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# 7. Save Model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/house_price_model.pkl')
joblib.dump(imputer, 'model/imputer.pkl') # Save imputer to handle web inputs
print("Model saved successfully in /model/ folder.")