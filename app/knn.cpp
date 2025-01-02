#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <tuple>

namespace py = pybind11;

// Calcular la distancia euclidiana entre dos puntos
double euclidean_distance(const std::vector<double>& point1, const std::vector<double>& point2) {
    double sum = 0.0;
    for (size_t i = 0; i < point1.size(); ++i) {
        sum += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return std::sqrt(sum);
}

// Buscar los K vecinos más cercanos
std::vector<int> knn(const std::vector<std::vector<double>>& train_data, 
                     const std::vector<int>& train_labels, 
                     const std::vector<std::vector<double>>& test_data, 
                     int k) {
    std::vector<int> predictions;

    for (const auto& test_point : test_data) {
        // Calcular las distancias a todos los puntos de entrenamiento
        std::vector<std::pair<double, int>> distances;
        for (size_t i = 0; i < train_data.size(); ++i) {
            double distance = euclidean_distance(test_point, train_data[i]);
            distances.push_back({distance, train_labels[i]});
        }

        // Ordenar las distancias
        std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        // Obtener los K vecinos más cercanos
        std::vector<int> k_neighbors;
        for (int i = 0; i < k; ++i) {
            k_neighbors.push_back(distances[i].second);
        }

        // Predecir la clase como la moda de los K vecinos
        std::vector<int> class_count(10, 0); // Asume que las clases son 0-9
        for (int label : k_neighbors) {
            class_count[label]++;
        }

        int predicted_class = std::distance(class_count.begin(), 
                                            std::max_element(class_count.begin(), class_count.end()));
        predictions.push_back(predicted_class);
    }

    return predictions;
}

// Exponer la función a Python
PYBIND11_MODULE(knn_cpp, m) {
    m.def("knn", &knn, "K-Nearest Neighbors Classification",
          py::arg("train_data"), py::arg("train_labels"), py::arg("test_data"), py::arg("k"));
}
