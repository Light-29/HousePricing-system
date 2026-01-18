import os
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# --- FIX START ---
# This finds the EXACT folder where app.py is sitting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# This creates a perfect path to your model files
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'house_price_model.pkl')
IMPUTER_PATH = os.path.join(BASE_DIR, 'model', 'imputer.pkl')
# --- FIX END ---

# Load the model
model = joblib.load(MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)

print(f"Success! Model loaded from: {MODEL_PATH}")

@app.route('/')
def home():
    return render_template('index.html')

# ... (the rest of your predict function remains the same)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form fields
        input_data = [
            float(request.form['OverallQual']),
            float(request.form['GrLivArea']),
            float(request.form['GarageCars']),
            float(request.form['TotalBsmtSF']),
            float(request.form['FullBath']),
            float(request.form['YearBuilt'])
        ]
        
        # Convert to numpy array and reshape for the model
        final_features = np.array([input_data])
        
        # Apply the same imputer used during training (safety check)
        final_features = imputer.transform(final_features)
        
        # Make prediction
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        return render_template('index.html', 
                               prediction_text=f'Estimated House Price: ${output:,.2f}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)