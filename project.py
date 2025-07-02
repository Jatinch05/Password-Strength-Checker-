    # %%
import pandas as pd
import math
import re
import nltk
import os
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))
from nltk.corpus import words
from collections import Counter
from functools import lru_cache
merged_df = pd.read_csv("final_dataset.csv")  
def password_length(password):
        return len(password)

    # Function to count character types
def count_lowercase(password):
        return sum(1 for c in password if c.islower())

def count_uppercase(password):
        return sum(1 for c in password if c.isupper())

def count_digits(password):
        return sum(1 for c in password if c.isdigit())

def count_special(password):
        return sum(1 for c in password if not c.isalnum())

    # Shannon entropy calculation
def shannon_entropy(password):
        length = len(password)
        if length == 0:
            return 0
        counts = Counter(password)
        probabilities = [count / length for count in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probabilities)
        return entropy

    # Check for repetitive patterns 
def detect_repetitive_patterns(password):
        repetitive = re.search(r'(.)\1{2,}', password)
        return 1 if repetitive else 0

    #Checking for sequences
def detect_arithmetic_sequence(password):
        length = len(password)
        if length < 3:
            return 0
        for i in range(length - 2):
            # Calculate the step between the first and second, then second and third characters
            step1 = ord(password[i+1]) - ord(password[i])
            step2 = ord(password[i+2]) - ord(password[i+1])

            if step1 == step2 and step1 != 0:
                return 1

        return 0

    #Checking for words in dictionary
# nltk.download('words')
english_words = set(words.words())

@lru_cache(maxsize=1000)
def detect_dictionary_words(password):
        # Check only substrings of 4 or more characters
        for i in range(len(password) - 3):  # Substring length threshold
            for j in range(i + 4, len(password) + 1):  # Only 4+ character substrings
                if password[i:j].lower() in english_words:
                    return 1
        return 0


keyboard_grid = {
        'q': (0, 0), 'w': (0, 1), 'e': (0, 2), 'r': (0, 3), 't': (0, 4), 'y': (0, 5), 'u': (0, 6), 'i': (0, 7), 'o': (0, 8), 'p': (0, 9),
        'a': (1, 0), 's': (1, 1), 'd': (1, 2), 'f': (1, 3), 'g': (1, 4), 'h': (1, 5), 'j': (1, 6), 'k': (1, 7), 'l': (1, 8),
        'z': (2, 0), 'x': (2, 1), 'c': (2, 2), 'v': (2, 3), 'b': (2, 4), 'n': (2, 5), 'm': (2, 6),
    }

def detect_keyboard_pattern_by_grid(password):
        password = password.lower()
        adjacent_count = 0

        for i in range(len(password) - 1):
            if password[i] not in keyboard_grid or password[i + 1] not in keyboard_grid:
                adjacent_count = 0
                continue

            pos1 = keyboard_grid[password[i]]
            pos2 = keyboard_grid[password[i + 1]]

            # Calculate movement on the grid
            row_move = abs(pos1[0] - pos2[0])
            col_move = abs(pos1[1] - pos2[1])

            # Check for direct horizontal, vertical, or diagonal adjacency
            if  (row_move <= 1 and col_move <= 1):
                if (password[i].isalpha() == password[i + 1].isalpha()) or (password[i].isdigit() == password[i + 1].isdigit()):
                    adjacent_count += 1
                if adjacent_count >= 2:  # At least three consecutive adjacent characters
                    return 1
            else:
                adjacent_count = 0  # Reset if not directly adjacent

        return 0

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import math
    import re
    import nltk
    from nltk.corpus import words
    from collections import Counter
    from functools import lru_cache
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings 
    warnings.filterwarnings('ignore')

    # %%

    df = pd.read_csv('data.csv',on_bad_lines='skip')

    # %%
    df.head()

    # %%
    df.info()

    # %%
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # %% [markdown]
    # # Feature Engineering #

    # %%
    # Function to calculate password length
    

    # %%
    #Feature Engineering
    df['password length']  = df['password'].apply(password_length)
    df['lower case']  = df['password'].apply(count_lowercase)
    df['upper case']  = df['password'].apply(count_uppercase)
    df['digits_count']  = df['password'].apply(count_digits)
    df['special characters']  = df['password'].apply(count_special)
    df['entropy']  = df['password'].apply(shannon_entropy)
    df['repetitive patterns']  = df['password'].apply(detect_repetitive_patterns)
    df['sequence']  = df['password'].apply(detect_arithmetic_sequence)
    df['keyboard patterns']  = df['password'].apply(detect_keyboard_pattern_by_grid)
    df['dictionary']  = df['password'].apply(detect_dictionary_words)

    # %%
    df

    # %% [markdown]
    # # Exploratory Data Analysis #

    # %%
    strength = df['strength']
    fig, axes = plt.subplots(nrows=10,ncols=1,figsize=(20,50))
    for i,col in enumerate(df.drop(columns=['password','strength']).columns):
        sns.scatterplot(y=strength,x=df[col],ax=axes[i])
    plt.show()

    # %%
    sns.heatmap(data=df.drop(columns=['password']).corr(),annot=True)
    plt.show()

    # %% [markdown]
    # # Model Training using Logistic Regression

    # %%
    X = df.drop(columns=['password','strength'])
    y = df['strength']

    # %%
    X

    # %%
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Multinomial Logistic Regression
    lm_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    lm_model.fit(X_train, y_train)

    # Predictions
    y_pred = lm_model.predict(X_test)

    # Evaluation
    from sklearn.metrics import accuracy_score

    print(f'Accuracy {accuracy_score(y_test,y_pred)}')

    # %%
    # Sample password to predict strength
    new_password = "&*+!`~?f"

    # Extract each feature in the specified order
    new_password_features = {
        "password length": len(new_password),
        "lower case": sum(1 for c in new_password if c.islower()),
        "upper case": sum(1 for c in new_password if c.isupper()),
        "digits_count": sum(1 for c in new_password if c.isdigit()),
        "special characters": sum(1 for c in new_password if c.isalnum()),
        "entropy": shannon_entropy(new_password),
        "repetitive patterns": detect_repetitive_patterns(new_password),
        "sequence": detect_arithmetic_sequence(new_password),
        "keyboard patterns": detect_keyboard_pattern_by_grid(new_password),
        "dictionary": detect_dictionary_words(new_password),
    }

    # Convert features to DataFrame format (order of columns as specified)
    import pandas as pd
    new_password_df = pd.DataFrame([new_password_features])

    # Predict strength using the trained model
    predicted_strength = lm_model.predict(new_password_df)
    print(f"Predicted Password Strength: {predicted_strength[0]}")


    # %% [markdown]
    # # Checking for Overfitting

    # %%
    from sklearn.model_selection import learning_curve

    # Generate learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        lm_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0]
    )




    # %%
    # Calculate the mean and standard deviation for train and test scores
    train_mean = np.abs(train_scores.mean(axis=1))
    train_std = np.abs(train_scores.std(axis=1))
    test_mean = np.abs(test_scores.mean(axis=1))
    test_std = np.abs(test_scores.std(axis=1))

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="r", label="Validation score")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.title("Learning Curve")
    plt.show()

    # %%
    train_accuracy = lm_model.score(X_train, y_train)
    print(f"Training Accuracy: {train_accuracy:.2f}")

    # Evaluate on testing set
    test_accuracy = lm_model.score(X_test, y_test)
    print(f"Testing Accuracy: {test_accuracy:.2f}")

    # %% [markdown]
    # **No major signs of overfitting as both the training accuracy and validation accuracy lines are converging. Also the there isn't much difference in the Training Accuracy score and the Testing Accuracy scores**

    # %%
    from sklearn.metrics import classification_report

    print(classification_report(y_test, y_pred))

    # %% [markdown]
    # **I suspected that the scores are true good to be true. So i tested the model by only giving the length of the password as an independent feature. I noticed 100% accuracy. So then I came to know that the dataset I used mostly classified the strengths of the passwords based on their lengths. So I found another dataset which contains 'Top 100 commonly used passwords'. Most of the passwords in this dataset are weak from what I saw. But there were some strong passwords. So I am going to filter this dataset to contain only weak passwords and combine it with the original dataset**

    # %%
    with open(r"C:\Users\jatin\Downloads\100k-most-used-passwords-NCSC.txt", "r", encoding='utf-8') as file:
        passwords = [line.strip() for line in file.readlines()]

    # %%
    passwords

    # %%
    common_passwords = pd.DataFrame(passwords,columns=['password'])

    # %%
    common_passwords['strength'] = 0

    # %%
    common_passwords.info()

    # %%
    common_passwords['password length']  = common_passwords['password'].apply(password_length)
    common_passwords['lower case']  = common_passwords['password'].apply(count_lowercase)
    common_passwords['upper case']  = common_passwords['password'].apply(count_uppercase)
    common_passwords['digits_count']  = common_passwords['password'].apply(count_digits)
    common_passwords['special characters']  = common_passwords['password'].apply(count_special)
    common_passwords['entropy']  = common_passwords['password'].apply(shannon_entropy)
    common_passwords['repetitive patterns']  = common_passwords['password'].apply(detect_repetitive_patterns)
    common_passwords['sequence']  = common_passwords['password'].apply(detect_arithmetic_sequence)
    common_passwords['keyboard patterns']  = common_passwords['password'].apply(detect_keyboard_pattern_by_grid)
    common_passwords['dictionary']  = common_passwords['password'].apply(detect_dictionary_words)

    # %%
    common_passwords

    # %%
    common_passwords = common_passwords[~common_passwords['password'].str.contains('¿½') == True]
    common_passwords.info()

    # %%
    sns.boxplot(common_passwords.select_dtypes(include=[int,float]))
    plt.show()

    # %%
    for column in common_passwords.select_dtypes(include=[int,float]):
        q1 = common_passwords[column].quantile(0.25)
        q3 = common_passwords[column].quantile(0.75)
        iqr = q3 - q1
        outliers = ((common_passwords[column].astype(float) <= (q1 - 1.5 * iqr)) | (common_passwords[column].astype(float) >= (q3 + 1.5 * iqr)))
        filtered_passwords = common_passwords[common_passwords[column] != outliers] 

    # %%
    filtered_passwords

    # %%
    sns.boxplot(filtered_passwords.select_dtypes(include=[int,float]))
    plt.show()

    # %%
    filtered_passwords = filtered_passwords[~(filtered_passwords['entropy'] > 4)]
    filtered_passwords = filtered_passwords[~(filtered_passwords['password length'] > 20)]

    # %%
    filtered_passwords.info()

    # %%
    filtered_passwords = filtered_passwords[~filtered_passwords['password'].str.contains('¿') == True]
    filtered_passwords.info()

    # %%
    print(filtered_passwords['password length'].max())
    print(filtered_passwords['entropy'].max())
    print(filtered_passwords['upper case'].max())

    # %%
    merged_df = pd.concat([df,filtered_passwords],ignore_index=True)
    merged_df

    # %%
    X = merged_df.drop(columns=['password','strength'])
    y = merged_df['strength']

    # %%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Multinomial Logistic Regression
    lm_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    lm_model.fit(X_train, y_train)

    # Predictions
    y_pred = lm_model.predict(X_test)

    # Evaluation
    from sklearn.metrics import accuracy_score

    print(f'Accuracy {accuracy_score(y_test,y_pred)}')

    # %%
    from sklearn.model_selection import learning_curve

    # Generate learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        lm_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0]
    )




    # %%
    # Calculate the mean and standard deviation for train and test scores
    train_mean = np.abs(train_scores.mean(axis=1))
    train_std = np.abs(train_scores.std(axis=1))
    test_mean = np.abs(test_scores.mean(axis=1))
    test_std = np.abs(test_scores.std(axis=1))

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="r", label="Validation score")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.title("Learning Curve")
    plt.show()

    # %%
    train_accuracy = lm_model.score(X_train, y_train)
    print(f"Training Accuracy: {train_accuracy:.2f}")

    # Evaluate on testing set
    test_accuracy = lm_model.score(X_test, y_test)
    print(f"Testing Accuracy: {test_accuracy:.2f}")

    # %% [markdown]
    # **No signs of Overfitting**

    # %% [markdown]
    # # Model Training Using Decision Tree Classifier

    # %%
    from sklearn.tree import DecisionTreeClassifier


    # %%
    from sklearn.model_selection import GridSearchCV
    # Set up the parameter grid
    dt_param_grid = {
        'max_depth': [None, 2, 4, 6, 8, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    }

    # Create a Decision Tree model
    dt_model = DecisionTreeClassifier(random_state=42)

    # Set up GridSearchCV
    dt_grid_search = GridSearchCV(estimator=dt_model, param_grid=dt_param_grid, 
                                cv=5, n_jobs=-1, verbose=1)

    # Fit GridSearchCV
    dt_grid_search.fit(X_train, y_train)

    print("Best parameters for Decision Tree:", dt_grid_search.best_params_)
    print("Best score for Decision Tree:", dt_grid_search.best_score_)


    # %%
    dtree_model = DecisionTreeClassifier(criterion='gini',max_depth=None,max_features='sqrt',min_samples_leaf=4,min_samples_split=10)
    dtree_model.fit(X_train, y_train)

    # Predictions
    y_pred = dtree_model.predict(X_test)

    # Evaluation
    from sklearn.metrics import accuracy_score

    print(f'Accuracy {accuracy_score(y_test,y_pred)}')

    # %%
    # Generate learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        dtree_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0]
    )


    # %%
    # Calculate the mean and standard deviation for train and test scores
    train_mean = np.abs(train_scores.mean(axis=1))
    train_std = np.abs(train_scores.std(axis=1))
    test_mean = np.abs(test_scores.mean(axis=1))
    test_std = np.abs(test_scores.std(axis=1))

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="r", label="Validation score")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.title("Learning Curve")
    plt.show()

    # %% [markdown]
    # **No overfitting**

    # %% [markdown]
    # # Model Training Using Random Forest Classifier

    # %%
    from sklearn.ensemble import RandomForestClassifier


    # %%
    from sklearn.model_selection import RandomizedSearchCV

    # Define the parameter distributions
    param_dist = {
        'n_estimators': np.arange(50, 501, 50),         
        'max_depth': list(np.arange(10, 101, 10)),  
        'min_samples_split': np.arange(2, 11),         
        'min_samples_split': np.arange(2, 11),         
        'max_features': ['auto', 'sqrt', 'log2'],     
        'bootstrap': [True, False],                    
        'criterion': ['gini', 'entropy']  }             
    # Initialize the RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)

    # Set up RandomizedSearchCV
    rf_random_search = RandomizedSearchCV(estimator=rf,
                                        param_distributions=param_dist,
                                        n_iter=50,                 
                                        scoring='accuracy',        
                                        cv=3,                      
                                        random_state=42,
                                        n_jobs=-1,                 
                                        verbose=2)

    # Fit RandomizedSearchCV to the data
    rf_random_search.fit(X_train, y_train)

    # Print the best parameters and best score
    print("Best parameters found: ", rf_random_search.best_params_)
    print("Best score: ", rf_random_search.best_score_)

    # %%
    rf_model = RandomForestClassifier(criterion='gini',max_depth=10,n_estimators=500,random_state=42,max_features='log2',min_samples_split=2,min_samples_leaf=4,bootstrap=True)
    rf_model.fit(X_train,y_train)
    y_pred = rf_model.predict(X_test)

    # Evaluation
    from sklearn.metrics import accuracy_score

    print(f'Accuracy {accuracy_score(y_test,y_pred)}')

    # %%
    # Generate learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        rf_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0]
    )


    # %%
    # Calculate the mean and standard deviation for train and test scores
    train_mean = np.abs(train_scores.mean(axis=1))
    train_std = np.abs(train_scores.std(axis=1))
    test_mean = np.abs(test_scores.mean(axis=1))
    test_std = np.abs(test_scores.std(axis=1))

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="r", label="Validation score")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.title("Learning Curve")
    plt.show()

    # %% [markdown]
    # **No major signs of overfitting**

    # %% [markdown]
    # # Model Training using XGBoost Classifier

    # %%
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)


    # %%
    params = {
        'objective': 'multi:softmax',  # for multi-class classification
        'num_class': len(y.unique()),  # number of unique classes in target
        'max_depth': 6,                # depth of each tree
        'learning_rate': 0.1,          # step size shrinkage
        'n_estimators': 100,           # number of boosting rounds
        'eval_metric': 'mlogloss',     # evaluation metric for multi-class classification
        'seed': 42
    }


    # %%
    xgb_model = xgb.train(params, dtrain, num_boost_round=100)


    # %%
    y_pred = xgb_model.predict(dtest)


    # %%
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # %%
    from sklearn.model_selection import RandomizedSearchCV
    from xgboost import XGBClassifier

    # Set up a more extensive parameter grid
    param_dist = {
        'max_depth': [3, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 150, 200],
        'subsample': [0.6, 0.7, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.1, 0.5, 1.0]
    }
    xgb_clf = XGBClassifier(objective='multi:softmax', num_class=len(y.unique()), seed=42)

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=xgb_clf, param_distributions=param_dist, n_iter=20, scoring='accuracy', cv=3, verbose=1, random_state=42)

    # Fit the model
    random_search.fit(X_train, y_train)

    # Get the best parameters and best score
    print("Best Parameters:", random_search.best_params_)
    print("Best Score:", random_search.best_score_)


    # %%
    # Generate learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        random_search, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1,
        train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0]
    )


    # %%
    # Calculate the mean and standard deviation for train and test scores
    train_mean = np.abs(train_scores.mean(axis=1))
    train_std = np.abs(train_scores.std(axis=1))
    test_mean = np.abs(test_scores.mean(axis=1))
    test_std = np.abs(test_scores.std(axis=1))

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="r", label="Validation score")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.title("Learning Curve")
    plt.show()

    # %%
    # Sample password to predict strength
    new_password = "!@#$%^&*()_+qaz"

    # Extract each feature in the specified order
    new_password_features = {
        "password length": len(new_password),
        "lower case": sum(1 for c in new_password if c.islower()),
        "upper case": sum(1 for c in new_password if c.isupper()),
        "digits_count": sum(1 for c in new_password if c.isdigit()),
        "special characters": sum(1 for c in new_password if c.isalnum()),
        "entropy": shannon_entropy(new_password),
        "repetitive patterns": detect_repetitive_patterns(new_password),
        "sequence": detect_arithmetic_sequence(new_password),
        "keyboard patterns": detect_keyboard_pattern_by_grid(new_password),
        "dictionary": detect_dictionary_words(new_password),
    }
    # Convert features to DataFrame format (order of columns as specified)
    import pandas as pd
    new_password_df = pd.DataFrame([new_password_features])

    # Predict strength using the trained model
    predicted_strength = xgb_model.predict(xgb.DMatrix(new_password_df))
    print(f"Predicted Password Strength: {predicted_strength[0]}")


    # %%
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    svm_model = SVC(kernel='rbf', C=1.0)  

    # Train (fit) the SVM model
    svm_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm_model.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))


    # %%
    train_sizes, train_scores, test_scores = learning_curve(
        svm_model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1,
        train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0]
    )

    # %%
    # Calculate the mean and standard deviation for train and test scores
    train_mean = np.abs(train_scores.mean(axis=1))
    train_std = np.abs(train_scores.std(axis=1))
    test_mean = np.abs(test_scores.mean(axis=1))
    test_std = np.abs(test_scores.std(axis=1))

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="r", label="Validation score")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.title("Learning Curve")
    plt.show()

    # %%



    # password = input("Enter your password: ")
    # assess_password(password, svm_model)

    # %% [markdown]
    # 

    # %%
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    models = {
        "Logistic Regression":lm_model,
        "Decision Tree": dtree_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "SVM": svm_model
    }

    results = []

    def evaluate_model(name, model, X_train, X_test, y_train, y_test):
        # model.fit(X_train, y_train)  # Train model
        if model == xgb_model:
            predictions = model.predict(dtest)
        else:
            predictions = model.predict(X_test)  # Predict on test set
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        
        # Append results to dictionary
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    # Evaluate each model
    for name, model in models.items():
        evaluate_model(name, model, X_train, X_test, y_train, y_test)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Print the results
    print(results_df)

    results_df = results_df.sort_values(by="Accuracy", ascending=False)
    print("\nSorted by Accuracy:")
    print(results_df)


    # %%
    import matplotlib.pyplot as plt

    # Plot comparison of models
    results_df.plot(x="Model", kind="bar", figsize=(15, 6))
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()


    # %%
    from joblib import dump, load

    # Save each model individually
    for name, model in models.items():
        dump(model, f'{name}_model.joblib')

    # %%


def predict_password_strength(password, model):
    features = {
    "password_length": len(password),
    "lower_case_count": sum(1 for c in password if c.islower()),
    "upper_case_count": sum(1 for c in password if c.isupper()),
    "digits_count": sum(1 for c in password if c.isdigit()),
    "special_characters_count": sum(1 for c in password if c.isalnum()),
    "entropy": shannon_entropy(password),
    "repetitive_patterns_present": detect_repetitive_patterns(password),
    "sequence_present": detect_arithmetic_sequence(password),
    "keyboard_pattern_present": detect_keyboard_pattern_by_grid(password),
    "dictionary_words_present": detect_dictionary_words(password),
    }
    feedback = []
    dict_features = features
    features = pd.DataFrame([features])
    for feature,value in dict_features.items():
        if feature == "repetitive patterns" or feature == "sequence" or feature == "keyboard patterns" or feature == "dictionary":
            if value == 1:
                 value = "Yes"
            else:
                value = "No"
        feedback.append(f'{feature}: {value}')
    # print(features)
    strength_prediction = model.predict(features)[0]  
    return strength_prediction,feedback

def get_password_feedback(password):
    recommendations = []
        
    if len(password) < merged_df['password_length'].mean():
        recommendations.append("Consider increasing the password length.")
            
    if count_uppercase(password) < merged_df['upper_case_count'].mean():
        recommendations.append("Add uppercase letters to make the password stronger.")

    if count_lowercase(password) < merged_df['lower_case_count'].mean():
        recommendations.append("Add lowercase letters to make the password stronger.")

    if count_digits(password) < merged_df['digits_count'].mean():
        recommendations.append("Add numbers to increase password strength.")

    if count_special(password) < merged_df['special_characters_count'].mean():
        recommendations.append("Include special characters like @, $, %, &, etc. to enhance security.")
        
    if detect_repetitive_patterns(password):
        recommendations.append("Avoid repetitive patterns in the password, such as 'aaa' or '111'.")

    if detect_keyboard_pattern_by_grid(password):  
        recommendations.append("Avoid common keyboard patterns like '1234' or 'qwerty'.")

    if detect_dictionary_words(password):  
        recommendations.append("Avoid using common words found in dictionaries.")
        
    if shannon_entropy(password) < merged_df['entropy'].mean():
        recommendations.append("Try increasing the randomness of the characters in your password")

    if detect_arithmetic_sequence(password):
        recommendations.append("Avoid using sequences like 1234,2468,aceg,abc and etc in your password")
                
    return recommendations

def assess_password(password, model):
    strength = predict_password_strength(password, model)
    feedback = get_password_feedback(password)
        
    if strength == 0:
        print("Password Strength: Weak. Recommendations: \n")
        for i, recommendation in enumerate(feedback, 1):
            print(f"{i}. {recommendation}")

    elif strength == 1:
        print("Password Strength: Medium.\n Recommendations: \n")
        for i, recommendation in enumerate(feedback, 1):
            print(f"{i}. {recommendation}")
    else:
        print("Password Strength: Strong\nYour password is secure.")
