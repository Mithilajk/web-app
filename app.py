from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features from form
        features = [
            float(request.form['Income']),
            float(request.form['Age']),
            float(request.form['Household_Size']),
            float(request.form['Education_Years']),
            float(request.form['Employment_Status']),
            float(request.form['Location_Urban']),
            float(request.form['Married']),
            float(request.form['Number_of_Dependents']),
            float(request.form['Owns_House']),
            float(request.form['Monthly_Installments']),
        ]

        # Make prediction
        final_features = np.array([features])  # shape must be [1, 10]
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Estimated Future Expense: ₹{output}')

    except Exception as e:
        # Handle errors gracefully
        return render_template('index.html', prediction_text=f'Error in prediction: {str(e)}')

# Only needed for local testing; Render will ignore this because it uses Gunicorn
if __name__ == "__main__":
    app.run(debug=True)
