import pandas as pd
import os
import yaml

# Читаем параметры
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

def main():
    os.makedirs("data/processed", exist_ok=True)
    
    # Чтение сырых данных
    df = pd.read_csv("data/raw/iris.csv")
    
    # Простая фича-инжиниринг (эмуляция Feature Store логики)
    df['sepal_ratio'] = df['sepal_length'] / df['sepal_width']
    
    # Сохраняем обработанные данные
    df.to_csv("data/processed/iris_processed.csv", index=False)
    print("Данные успешно подготовлены и сохранены.")

if __name__ == "__main__":
    main()
