# Import library
import numpy as np
import Perceptron as p

# Inisialisasi input (X1, X2) dan target (t) untuk masalah Logika OR (Bipolar)
X = np.array([
    [ 1,  1], 
    [ 1, -1], 
    [-1,  1], 
    [-1, -1]
])

t = np.array([
    [ 1], 
    [ 1], 
    [ 1], 
    [-1]
])

# Pemanggilan model Perceptron
# Learning rate = 0.1, Max epoch = 10
model = p.Perceptron(alpha=0.1, epoch=10)

# Jalankan proses training
model.fit(X, t)