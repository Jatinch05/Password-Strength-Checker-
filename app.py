from flask import Flask, render_template, request
import joblib
import pandas as pd
from project import predict_password_strength, get_password_feedback,shannon_entropy,count_digits,count_lowercase,count_uppercase,count_special,detect_arithmetic_sequence,detect_dictionary_words,detect_keyboard_pattern_by_grid,detect_repetitive_patterns

app = Flask(__name__)

# Load your pre-trained model
rf_model = joblib.load("Random Forest_model.joblib") 
  

@app.route("/", methods=["GET", "POST"])
def index():
    strength = None
    feedback = []
    password = ""
    recommendations = []

    if request.method == "POST":
        password = request.form["password"]
        # Get the password strength prediction and feedback
        strength,feedback = predict_password_strength(password, rf_model)
        recommendations = get_password_feedback(password)
        strength = int(strength)
        if strength == 0:
            strength = "Weak"
        elif strength == 1:
            strength = "Medium"
        else:
            strength = "Strong"

    return render_template("index.html", strength=strength, recommendations=recommendations, password=password,feedback =feedback
    )

if __name__ == "__main__":
    app.run(debug=True)
