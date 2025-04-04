from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load the trained model
# model = joblib.load("term_deposit_model.pkl")
model = joblib.load("models/lr_model.pkl")

# Create Flask app
app = Flask(__name__)

@app.route("/")
def home():
     return render_template("index.html")

# Define the API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Extract input data
        features = np.array([[
            data["age"], 
            data["job"], 
            data["marital"], 
            data["education"], 
            data["default"], 
            data["housing"], 
            data["loan"]
        ]])
        
        # Make prediction
        prediction = model.predict(features)
        result = "yes" if prediction[0] == 1 else "no"
        
        return jsonify({"subscribed": result})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

