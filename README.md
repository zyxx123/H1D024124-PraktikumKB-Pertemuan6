# Praktikum Kecerdasan Buatan - Pertemuan 6  
## Jaringan Syaraf Tiruan (JST)

Repositori ini berisi implementasi algoritma **Jaringan Syaraf Tiruan (JST)** menggunakan bahasa pemrograman Python. Praktikum ini mencakup dua model pembelajaran JST:

1. **Perceptron** untuk memecahkan masalah logika OR (*Single-Layer Perceptron*).
2. **Backpropagation** untuk memecahkan masalah logika XOR (*Multi-Layer Perceptron* dengan hidden layer).

Seluruh algoritma diimplementasikan **from scratch** menggunakan pustaka dasar:
- `numpy` untuk komputasi numerik.
- `matplotlib` untuk visualisasi.

## Struktur File

| Nama File | Deskripsi |
|--------|--------|
| `Perceptron.py` | Berisi class `Perceptron` yang mengimplementasikan arsitektur JST Perceptron, fungsi aktivasi, update bobot menggunakan delta rule, dan fungsi visualisasi. |
| `Perceptron_or.py` | Program utama untuk menjalankan model Perceptron pada kasus logika OR. |
| `Backpropagation.py` | Berisi class `Backpropagation` yang mengimplementasikan feedforward dan backward propagation dengan arsitektur 2-2-1. |
| `Backpropagation_xor.py` | Program utama untuk menjalankan model Backpropagation pada kasus logika XOR. |
| `HasilPerceptron.txt` | File log hasil training Perceptron, berisi epoch, bobot, bias, prediksi, dan nilai SSE. |
| `hasilBackpropagation.txt` | File log hasil training Backpropagation, berisi detail propagasi maju, propagasi mundur, update bobot, dan SSE tiap epoch. |

## Persyaratan (Prerequisites)

Pastikan telah terinstal:

- Python 3.x
- `numpy`
- `matplotlib`

Install library yang dibutuhkan dengan perintah berikut:

```bash
pip install numpy matplotlib
````

## Cara Menjalankan Program

### 1. Menjalankan Model Perceptron (Logika OR)

```bash
python Perceptron_or.py
```

#### Hasil yang Diharapkan

* Program akan melatih model untuk mengenali pola logika OR.
* Muncul jendela visualisasi decision boundary pada setiap epoch.
* File `HasilPerceptron.txt` akan dibuat atau diperbarui secara otomatis.
* Training berhenti ketika nilai **SSE = 0**.

---

### 2. Menjalankan Model Backpropagation (Logika XOR)

```bash
python Backpropagation_xor.py
```

#### Hasil yang Diharapkan

* Program akan melatih model untuk mengenali pola logika XOR.
* Muncul grafik penurunan error (SSE) terhadap epoch.
* File `hasilBackpropagation.txt` akan dibuat atau diperbarui secara otomatis.
* File log mencatat detail feedforward, backward propagation, dan update bobot.

---

## Dataset yang Digunakan

### Data OR

| x1 | x2 | Target |
| -- | -- | ------ |
| 0  | 0  | 0      |
| 0  | 1  | 1      |
| 1  | 0  | 1      |
| 1  | 1  | 1      |

### Data XOR

| x1 | x2 | Target |
| -- | -- | ------ |
| 0  | 0  | 0      |
| 0  | 1  | 1      |
| 1  | 0  | 1      |
| 1  | 1  | 0      |

---

## Arsitektur Jaringan

### Perceptron

* Input Layer: 2 neuron
* Output Layer: 1 neuron
* Fungsi Aktivasi: Step Function

### Backpropagation

* Input Layer: 2 neuron
* Hidden Layer: 2 neuron
* Output Layer: 1 neuron
* Fungsi Aktivasi: Sigmoid

---

## Visualisasi

### Perceptron

Menampilkan:

* Titik data training.
* Garis decision boundary.
* Perubahan garis pemisah pada setiap epoch.

### Backpropagation

Menampilkan:

* Grafik hubungan antara epoch dan SSE.
* Menunjukkan proses konvergensi jaringan.

---

## Catatan

* Proyek ini dibuat untuk memenuhi tugas **Praktikum Kecerdasan Buatan (KB) Pertemuan 6**.
* Seluruh algoritma diimplementasikan tanpa menggunakan framework machine learning seperti:

  * Scikit-Learn
  * TensorFlow
  * PyTorch
* Implementasi disesuaikan dengan modul praktikum.


## Author

**Zaki Fatah**
Mahasiswa Informatika

