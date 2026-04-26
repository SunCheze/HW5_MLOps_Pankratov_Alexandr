import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import yaml
import os

def main():
    os.makedirs("models", exist_ok=True)
    
    # Читаем параметры из params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    # Чтение обработанных данных
    df = pd.read_csv("data/processed/iris_processed.csv")
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'sepal_ratio']]
    y = df['species']

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params["train"]["test_size"], 
        random_state=params["train"]["random_state"]
    )

    # Настройка MLflow (по умолчанию логирует локально в папку mlruns)
    mlflow.set_experiment("Iris_LogisticRegression")

    with mlflow.start_run():
        # Логируем параметры из YAML
        mlflow.log_params(params["train"])
        
        # Обучение модели
        model = LogisticRegression(max_iter=params["train"]["max_iter"])
        model.fit(X_train, y_train)
        
        # Оценка
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Логирование метрик и модели
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        print(f"Model trained. Accuracy: {acc}")

    # Сохранение модели локально (артефакт для DVC)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()
