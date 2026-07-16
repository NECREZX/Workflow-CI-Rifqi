import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def run_modelling():
    print("Loading data...")
    df = pd.read_csv("churn_preprocessing/churn_processed.csv")
    X = df.drop("Exited", axis=1)
    y = df["Exited"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Starting MLflow autolog run...")
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="Basic_RandomForest_Model"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"Model Accuracy: {score}")

if __name__ == "__main__":
    run_modelling()
