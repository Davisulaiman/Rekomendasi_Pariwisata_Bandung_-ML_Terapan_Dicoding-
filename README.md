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

## Data Preparation

1. **Cleaning**

   * Menghapus kolom tidak relevan (`Unnamed: 11`, `Unnamed: 12`)
   * Menangani missing values pada `Time_Minutes`
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
* **Output Rekomendasi (berdasarkan input: 'Trans Studio Bandung')**:

  1. Chingu Cafe Little Seoul
  2. Taman Badak
  3. NuArt Sculpture Park
  4. Kiara Artha Park
  5. Upside Down World Bandung
  6. Jendela Alam
  7. Panghegar Waterboom Bandung
  8. Sudut Pandang Bandung
  9. Batununggal Indah Club
  10. Kampung Batu Malakasari

### Collaborative Filtering

* **Prinsip:** Item-Based k-NN atau Matrix Factorization
* **Model:** Neural Collaborative Filtering (keras)
* **Metode Training:** 100 epoch, RMSE converging ke \~0.36
* **Output Rekomendasi untuk User 152**:

  * Tempat dengan rating tertinggi dari user:

    * Kebun Binatang Bandung : Cagar Alam
    * Taman Lalu Lintas Ade Irma Suryani Nasution : Taman Hiburan
    * Museum Barli : Budaya
    * Monumen Perjuangan Rakyat Jawa Barat : Budaya
    * Taman Begonia : Cagar Alam

  * Top 10 Rekomendasi:

    1. Upside Down World Bandung (Taman Hiburan, Rp100.000, 4.0)
    2. Taman Lansia (Taman Hiburan, Gratis, 4.4)
    3. Selasar Sunaryo Art Space (Taman Hiburan, Rp25.000, 4.6)
    4. Teras Cikapundung BBWS (Taman Hiburan, Gratis, 4.3)
    5. Museum Pos Indonesia (Budaya, Gratis, 4.5)
    6. Ciwangun Indah Camp (Cagar Alam, Rp10.000, 4.3)
    7. Curug Batu Templek (Cagar Alam, Rp5.000, 4.1)
    8. Taman Budaya Jawa Barat (Budaya, Gratis, 4.3)
    9. Masjid Agung Trans Studio Bandung (Tempat Ibadah, Gratis, 4.8)
    10. Sanghyang Heuleut (Cagar Alam, Rp10.000, 4.4)

## Evaluation

### Metrik Evaluasi:

| Metrik                             | Nilai Hasil Model | Deskripsi                                                                                                                                                                                                                                |
| ---------------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Root Mean Squared Error (RMSE)** | **0.3602**        | Mengukur rata-rata selisih kuadrat antara rating prediksi dan rating aktual. Digunakan untuk mengevaluasi performa model collaborative filtering dalam memprediksi rating tempat wisata. Semakin kecil nilai RMSE, semakin akurat model. |

### Hasil Evaluasi:

* Content-Based Filtering menunjukkan hasil memuaskan untuk cold-start (pengguna baru) tanpa memerlukan riwayat interaksi.
* Collaborative Filtering dilatih selama **100 epoch**, dengan nilai **root mean squared error (RMSE)** validasi mendekati **0.36**, menunjukkan prediksi rating yang cukup akurat.
* Model collaborative mampu memberikan personalisasi rekomendasi dengan mempertimbangkan preferensi pengguna aktif.

### Hubungan dengan Business Understanding:

* Model menjawab semua problem statement dan mencapai goals
* Meningkatkan pengalaman pengguna baru dan lama dalam memilih wisata

## Contoh Tempat Wisata di Bandung

* Tebing Keraton
* Orchid Forest Cikole
* Dusun Bambu
* Lembang Park and Zoo
* Farm House Lembang
* Kawah Putih
* Taman Hutan Raya Djuanda

---