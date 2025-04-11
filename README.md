# Tubes_ML_1

## Deskripsi
Ini adalah kode implementasi Feed Forward Neural Network untuk Tugas Besar IF3270 Machine Learning

## Cara menjalankan program
Clone repository
```
https://github.com/Devinzenzhang/Tubes_ML_1
```
Buka test.ipynb untuk testing

Contoh
```
Model = FFNN(784, [3, 3, 3], 10, ["sigmoid", "sigmoid", "sigmoid", "softmax"], "mse", [("uniform", -1, 1, 42, "he") for _ in range (4)], regularization="L2", lambda_=0.001)
```
parameternya input size, hidden layer sizes (jumlah neuron setiap hidden layer), output size, activation function (setiap layer), loss function, weight initalization (setiap layer), regularization, lambda.

## Pembagian Tugas
| NIM            | Nama                                  | Pembagian Tugas                                                                     |
|----------------|---------------------------------------|-------------------------------------------------------------------------------------|
|12821046        |Fardhan Indrayesa                      |Value, ValueTensor, Regularization, Bonus automatic differentiation, Testing, Laporan|
|13522064        |Devinzen                               |Layer, Model, Bonus Xavier dan He initialization, Testing                            |
