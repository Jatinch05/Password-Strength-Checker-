# Password-Strength-Checker-and-Recommendations-System
**Password Strength Prediction Project** 
# 📋 Overview:
This project is designed to assess the strength of passwords using machine learning models. It provides users with real-time feedback on password strength and offers suggestions to improve weak or medium-strength passwords. The project is implemented using Python, Flask, and HTML/CSS, and uses various models like Random Forest, XGBoost, and Support Vector Machines (SVM) to classify passwords into categories: Weak, Medium, or Strong.
# 🚀 Features:
**Password Strength Prediction:** Classifies passwords into Weak, Medium, or Strong categories.
**Interactive Recommendations:** Provides suggestions to strengthen passwords based on various factors like length, character composition, entropy, and patterns.
**Real-Time Feedback:** Evaluates passwords for dictionary words, keyboard patterns, repetitive characters, sequences, and more.
Web Interface: User-friendly web interface built using Flask and HTML/CSS.
# 📂 Project Structure:

├── app.py                # Main Flask application              

├── project.py            # Core functions for password analysis

├── Random Forest_model.joblib  # Pre-trained Random Forest model

├── templates/

│   └── index.html        # Web page template

├── static/

│   └── styles.css        # CSS for the web page


├── requirements.txt      # List of dependencies
├── README.md             # Project documentation

└── dataset.csv           # Dataset used for training the models
# 🛠️ Installation:
**Prerequisites**
Make sure you have the following installed:

Python 3.x

pip package manager

Step 1: Clone the Repository 

git clone https://github.com/Jatinch05/Password-Strength-Checker-

cd password-strength-prediction

Step 2: Create a Virtual Environment (Optional but Recommended)

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

Step 3: Install the Dependencies

pip install -r requirements.txt

Step 4: Download the Pre-trained Model

Ensure that you have the pre-trained model file (Random Forest_model.joblib) in the root directory. If not, retrain the model using the dataset provided.


Step 5: Run the Application

python app.py

Step 6: Access the Web Interface

Click on the link generated after running the file


# 🗃️ Dataset:
The project uses a password dataset containing passwords and their corresponding strength labels (Weak, Medium, Strong). If you want to use your own dataset, ensure it has the following format:

password,strength

example123,Weak

SecurePass!23,Strong

...
# ⚙️ Machine Learning Models
This project uses multiple ML models to classify passwords:

Random Forest: The primary model used in the web application.
XGBoost
Support Vector Machine (SVM)
Logistic Regression
Decision Trees

**Feature Engineering**
The following features are extracted from passwords for prediction:

Password length
Number of lowercase, uppercase, digits, and special characters
Shannon entropy
Detection of repetitive patterns, sequences, keyboard patterns, and dictionary words

**Model Evaluation**

The models were evaluated using metrics like:

Accuracy
Precision, Recall, F1-score
# 📈 Evaluation Metrics
Accuracy: ~97.4% for Random Forest, SVM, and XGBoost models.

Precision, Recall, F1-score: Measured across each password strength class.

# 🖥️ Web Interface
The web interface allows users to input a password and get instant feedback on its strength, along with recommendations for improvement.

# 🛡️ Security Considerations
The password input is processed server-side; however, the project is intended for educational purposes and should not be used for handling sensitive information in a production environment.
# 🤝 Contribution
Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

# 📄 License
This project is licensed under the MIT License.
