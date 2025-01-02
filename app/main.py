from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import torch
import pandas as pd
import knn_cpp

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]

# Generar datos de ejemplo
def generate_data(num_samples=100, num_classes=3):
    torch.manual_seed(42)
    X = torch.rand((num_samples, 2)) * 10
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y

# Dividir datos en entrenamiento y prueba
def split_data(X, y, test_ratio=0.2):
    num_test = int(len(X) * test_ratio)
    indices = torch.randperm(len(X))
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]

@app.post("/knn")
def calculo():
    # Generar y dividir datos
    X, y = generate_data(num_samples=200, num_classes=3)
    X_train, y_train, X_test, y_test = split_data(X, y)

    # Convertir datos a listas para el módulo C++
    train_data = X_train.tolist()
    train_labels = y_train.tolist()
    test_data = X_test.tolist()

    # Llamar al módulo C++ para KNN
    k = 3
    predictions = knn_cpp.knn(train_data, train_labels, test_data, k)

    # Evaluar resultados
    accuracy = sum([1 if pred == true else 0 for pred, 
                    true in zip(predictions, y_test.tolist())]) / len(y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Visualización
    train_df = pd.DataFrame(X_train.numpy(), columns=["x1", "x2"])
    train_df['label'] = y_train.numpy()

    test_df = pd.DataFrame(X_test.numpy(), columns=["x1", "x2"])
    test_df['predicted'] = predictions

    plt.figure(figsize=(10, 6))
    plt.scatter(train_df['x1'], train_df['x2'], c=train_df['label'], cmap='viridis', label='Training Data')
    plt.scatter(test_df['x1'], test_df['x2'], c=test_df['predicted'], cmap='plasma', marker='x', label='Test Predictions')
    plt.legend()
    plt.title("KNN Classification Results")
    output_file = "knn_results.png"
    plt.savefig(output_file)
    plt.close()
    # Regresar el archivo como respuesta
    return FileResponse(output_file, media_type="image/png", filename=output_file)



    