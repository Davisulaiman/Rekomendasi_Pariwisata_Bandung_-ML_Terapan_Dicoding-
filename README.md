# Laporan Proyek Machine Learning - Sistem Rekomendasi Tempat Wisata di Kota Bandung

## Project Overview

Kota Bandung merupakan salah satu destinasi wisata unggulan di Indonesia. Kota ini menawarkan beragam pilihan tempat wisata mulai dari alam, budaya, kuliner, hingga rekreasi keluarga. Dengan banyaknya pilihan yang tersedia, wisatawan sering kali merasa kesulitan memilih destinasi yang sesuai dengan preferensi mereka. Oleh karena itu, dibutuhkan sistem rekomendasi yang dapat membantu pengguna memilih tempat wisata yang sesuai dengan minat mereka.

Proyek ini membangun sistem rekomendasi tempat wisata di Kota Bandung menggunakan dua pendekatan utama:

* **Content-Based Filtering**: memberikan rekomendasi berdasarkan informasi konten dari tempat wisata (kategori, deskripsi).
* **Collaborative Filtering**: memberikan rekomendasi berdasarkan pola interaksi pengguna lain yang memiliki preferensi serupa.

Referensi:

* Chalkiadakis et al. (2023). *A Novel Hybrid Recommender System for the Tourism Domain*. Algorithms, 16(215).
* Margaris et al. (2025). *Using Prediction Confidence Factors to Enhance Collaborative Filtering Recommendation Quality*. Technologies, 13(181).

## Business Understanding

### Problem Statements

* Bagaimana memberikan rekomendasi tempat wisata di Bandung yang sesuai dengan minat pengguna?
* Bagaimana memanfaatkan data konten dan interaksi pengguna untuk meningkatkan akurasi sistem rekomendasi?

### Goals

* Menghasilkan sistem rekomendasi top-N yang akurat dan personal untuk tempat wisata di Bandung.
* Membangun dua pendekatan sistem rekomendasi: content-based dan collaborative filtering.

### Solution Statements

* **Content-Based Filtering**: TF-IDF + cosine similarity dari konten tempat wisata.
* **Collaborative Filtering**: matriks user-item dan similaritas antar item.

## Data Understanding

Dataset terdiri dari:

1. `user.csv`

   * Jumlah: 300 pengguna
   * Kolom: `User_Id`, `Location`, `Age`
   * Tidak ada missing values atau duplikasi

2. `tourism_with_id.csv`

   * Jumlah: 437 tempat wisata
   * Kolom: `Place_Id`, `Place_Name`, `Description`, `Category`, `City`, `Price`, `Rating`, `Time_Minutes`, `Coordinate`, `Lat`, `Long`
   * Missing values: `Time_Minutes` sebagian besar kosong
   * Duplikasi: tidak ada
   * Fitur `Unnamed: 11` dan `Unnamed: 12` dihapus karena tidak relevan

3. `tourism_rating.csv`

   * Jumlah: 10.000 entri
   * Kolom: `User_Id`, `Place_Id`, `Place_Ratings`
   * Tidak ada missing values atau duplikasi

**Summary Insight:**

* Distribusi rating 1–5 merata dengan kecenderungan ke nilai tinggi.
* Kota difokuskan pada `City == Bandung`.
* Semua fitur relevan dijelaskan dan digunakan sesuai kebutuhan model.

## Data Preparation

Urutan tahapan:

1. **Cleaning**

   * Menghapus kolom tidak relevan (`Unnamed: 11`, `Unnamed: 12`)
   * Menangani missing values pada `Time_Minutes` dengan imputasi rata-rata/median atau menghapus baris jika sangat tidak lengkap
   * Filter tempat wisata hanya `City == "Bandung"`

2. **Preprocessing Teks**

   * Stemming dan stopword removal dengan Sastrawi

3. **Ekstraksi Fitur**

   * TF-IDF untuk `Category` dan `Description`

4. **Pembentukan Matriks**

   * Matriks user-item dari `tourism_rating.csv`

## Modeling

### Content-Based Filtering

* **Prinsip:** Menggunakan cosine similarity antar TF-IDF vektor konten
* **Fungsi:** Memberikan rekomendasi berdasarkan kesamaan konten
* **Parameter:** TF-IDF max\_features = 5000, ngram\_range = (1,2)

### Collaborative Filtering

* **Prinsip:** Item-Based k-NN pada matriks user-item
* **Parameter:**

  * K = 20 (jumlah tetangga terdekat)
  * Similarity metric = cosine
* **Alternatif:** Matrix factorization (Singular Value Decomposition)

### Perbandingan:

| Pendekatan              | Kelebihan                                         | Kekurangan                           |
| ----------------------- | ------------------------------------------------- | ------------------------------------ |
| Content-Based Filtering | Tidak butuh data pengguna lain (cold-start ready) | Terbatas pada item serupa            |
| Collaborative Filtering | Menangkap pola preferensi kompleks                | Tidak cocok untuk pengguna/item baru |

## Evaluation

### Metrik Evaluasi:

* **Precision\@N**: relevansi dalam top-N hasil
* **Recall\@N**: seberapa besar item relevan ditemukan
* **MAP (Mean Average Precision)**: evaluasi ranking

### Hasil Evaluasi:

* Content-Based unggul untuk cold-start (pengguna baru)
* Collaborative unggul untuk pengguna aktif

### Hubungan dengan Business Understanding:

* Model mampu menjawab dua permasalahan utama
* Meningkatkan pengalaman pengguna dalam memilih destinasi
* Content-Based menjawab kebutuhan wisatawan baru
* Collaborative meningkatkan loyalitas pengguna aktif

## Contoh Tempat Wisata di Bandung

* Tebing Keraton
* Orchid Forest Cikole
* Dusun Bambu
* Lembang Park and Zoo
* Farm House Lembang
* Kawah Putih
* Taman Hutan Raya Djuanda

---

**Catatan Tambahan:**

* Rencana pengembangan: Hybrid Recommender, Visualisasi rekomendasi dengan Streamlit/Dash
