# Import library
import numpy as np
import Backpropagation as b

# Inisialisasi input (X1, X2) dan target (t) untuk masalah Logika XOR (Bipolar)
X = np.array([
    [ 1,  1], 
    [ 1, -1], 
    [-1,  1], 
    [-1, -1]
])

t = np.array([
    [-1], 
    [ 1], 
    [ 1], 
    [-1]
])

# Pemanggilan model Backpropagation
# Alpha = 0.3, Max epoch = 1000, Target Error = 0.001
model = b.Backpropagation(alpha=0.3, epoch=1000, target_error=0.001)

# Jalankan proses training
model.fit(X, t)