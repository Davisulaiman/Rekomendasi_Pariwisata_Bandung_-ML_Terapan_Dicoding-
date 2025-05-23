# Laporan Proyek Machine Learning – Rekomendasi Wisata Kota Bandung

![Iconic Bandung](assets/iconic_bandung.png)

## Domain Proyek

Bandung, ibu kota Provinsi Jawa Barat, dikenal sebagai salah satu destinasi wisata utama di Indonesia. Kota ini menawarkan ragam pengalaman mulai dari keindahan alam pegunungan, kekayaan budaya lokal, hingga tempat hiburan dan belanja modern yang menarik perhatian wisatawan dari dalam maupun luar negeri. Terkenalnya tempat-tempat seperti Lembang, Dago, Braga, dan Trans Studio menjadikan Bandung sebagai kota wisata dengan kunjungan tinggi setiap tahunnya.

Namun, banyaknya pilihan destinasi ini kerap membuat wisatawan kebingungan dalam menentukan tujuan wisata yang sesuai dengan preferensi pribadi. Hal ini menciptakan kebutuhan akan sistem rekomendasi cerdas yang dapat membantu menyaring informasi dan menyarankan tempat wisata yang paling relevan dan menarik bagi masing-masing individu.

Proyek ini mengembangkan sistem rekomendasi wisata menggunakan dua pendekatan utama: **Content-Based Filtering (CBF)** dan **Collaborative Filtering (CF)**. Pendekatan ini mengadaptasi strategi machine learning yang umum digunakan dalam berbagai bidang, termasuk sektor kesehatan seperti prediksi risiko kanker paru-paru. Dalam konteks wisata, sistem ini mampu memberikan rekomendasi berdasarkan kemiripan konten antar tempat wisata maupun dari pola perilaku pengguna lain yang serupa.

**Referensi Ilmiah:**

* Oktaviani et al., 2023. *Rekomendasi Destinasi Wisata Kota Bandung Berbasis Collaborative Filtering dan Content-Based Filtering*. Jurnal Teknologi Informasi dan Ilmu Komputer, 10(2): 252–259. [https://doi.org/10.25126/jtiik.202310252](https://doi.org/10.25126/jtiik.202310252)
* Chalkiadakis et al., 2023. *A Novel Hybrid Recommender System for the Tourism Domain*. Algorithms, 16(215). [https://doi.org/10.3390/a16040215](https://doi.org/10.3390/a16040215)
* Margaris et al., 2025. *Using Prediction Confidence Factors to Enhance Collaborative Filtering Recommendation Quality*. Technologies, 13(181). [https://doi.org/10.3390/technologies13050181](https://doi.org/10.3390/technologies13050181)

---

## Business Understanding

### Problem Statements

* Bagaimana memberikan rekomendasi wisata yang relevan bagi pengguna baru dan lama?
* Bagaimana meningkatkan kualitas personalisasi dalam sistem rekomendasi?

### Goals

* Membangun dua model rekomendasi (CBF dan CF) untuk tempat wisata di Bandung.
* Mengevaluasi performa model berdasarkan hasil top-N rekomendasi dan nilai RMSE.

### Solution Statements

* **CBF:** Menggunakan cosine similarity antar TF-IDF deskripsi/kategori wisata.
* **CF:** Menggunakan deep learning embedding dan matrix factorization, serta confidence factor (jumlah tetangga, rerata rating pengguna dan tempat).

---

## Data Understanding

### Dataset:

* `tourism_with_id.csv`: informasi tempat wisata
* `tourism_rating.csv`: data interaksi rating pengguna
* `user.csv`: data demografi pengguna

### Struktur Data:

* Tempat wisata diklasifikasikan berdasarkan kategori.
* Rating berkisar dari 1–5.
* Pengguna berasal dari berbagai kota dan usia berbeda.

### Visualisasi:

* ![Distribusi Rating](assets/rating_distribution.png)

  *Gambar 1. Histogram distribusi rating wisatawan.*

---

## Data Preparation

| No | Langkah              | Deskripsi                                                  |
| -- | -------------------- | ---------------------------------------------------------- |
| 1  | Data Cleaning        | Menghapus baris kosong dan data duplikat                   |
| 2  | Label Encoding       | ID pengguna dan tempat diubah menjadi numerik              |
| 3  | TF-IDF Vectorization | Untuk deskripsi dan kategori tempat wisata                 |
| 4  | Split Data           | Membagi data rating untuk pelatihan dan pengujian model CF |

---

## Modeling

### Content-Based Filtering (CBF)

* TF-IDF digunakan untuk menghitung similarity antar tempat wisata.
* Cosine similarity digunakan untuk menilai kemiripan:

$\text{cosine}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$

* Rekomendasi berdasarkan tempat dengan deskripsi/kategori serupa.

### Collaborative Filtering (CF)

* Menggunakan `RecommenderNet` berbasis embedding.
* Confidence scoring digunakan dari Margaris et al. (2025):

  * Jumlah tetangga
  * Rata-rata rating pengguna
  * Rata-rata rating tempat
* Evaluasi dengan RMSE:

$RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }$

---

## Evaluation

### CBF

* Relevansi rekomendasi dilihat dari kemiripan konten.
* Cocok untuk pengguna baru.

### CF

* RMSE < 0.25 dicapai setelah pelatihan.
* Plot evaluasi:

  ![RMSE Loss Plot](assets/rmse_loss.png)

  *Gambar 3. Grafik training/validation loss model CF.*

---

## Output Rekomendasi

### Content-Based Filtering

Rekomendasi untuk pengguna yang menyukai **Trans Studio Bandung**:

1. Sudut Pandang Bandung
2. Kiara Artha Park
3. Panghegar Waterboom
4. Chingu Cafe

![CBF Output](assets/cbf_output.png)

*Gambar 4. Contoh hasil rekomendasi sistem CBF berdasarkan preferensi input.*

### Collaborative Filtering

Rekomendasi top-5 berdasarkan histori pengguna:

1. Dago Dreampark
2. The Lodge Maribaya
3. Lembang Park & Zoo
4. Farmhouse Susu Lembang
5. Floating Market Lembang

![CF Output](assets/cf_output.png)

*Gambar 5. Contoh hasil rekomendasi sistem CF berdasarkan data interaksi pengguna.*

---

## Analisis Hasil Modeling

### 1. Performa Model

* **CBF** sangat baik dalam memberikan rekomendasi yang mirip secara konten.
* **CF** menghasilkan prediksi yang akurat dengan RMSE yang rendah.

### 2. Tantangan

* **CBF:** Kurang efektif jika metadata tidak lengkap atau deskripsi tidak akurat.
* **CF:** Performa menurun jika data pengguna minim (cold start).

### 3. Model Terbaik

* Untuk pengguna baru, **CBF** lebih stabil.
* Untuk pengguna lama, **CF** unggul dalam personalisasi.

---

## Keterkaitan dengan Business Understanding

### Apakah Model Menjawab Problem Statements?

* Ya. Kedua pendekatan mengakomodasi kebutuhan pengguna baru dan lama.

### Apakah Model Mencapai Goals?

* Tercapai. Evaluasi menunjukkan performa memuaskan.

---

## Rekomendasi dan Langkah Selanjutnya

1. **Integrasi Hybrid Model**:

   * Gabungkan pendekatan CBF dan CF secara dinamis.

2. **Peningkatan Metadata**:

   * Tambahkan informasi visual, lokasi, ulasan, dan rating waktu nyata.

3. **Evaluasi Lanjutan**:

   * Uji langsung ke pengguna untuk menilai kepuasan dan relevansi.

4. **Pengembangan Aplikasi**:

   * Kembangkan antarmuka rekomendasi berbasis web atau mobile.

---

## Kesimpulan

1. Proyek ini berhasil mengembangkan sistem rekomendasi wisata berbasis machine learning yang menggabungkan pendekatan Content-Based Filtering dan Collaborative Filtering.
2. Model **CBF** efektif dalam memberikan rekomendasi berdasarkan konten bagi pengguna baru, sedangkan **CF** menunjukkan keunggulan dalam memberikan rekomendasi yang dipersonalisasi untuk pengguna lama dengan akurasi prediksi tinggi (RMSE < 0.25).
3. Hasil evaluasi menunjukkan bahwa kedua pendekatan saling melengkapi dalam skenario penggunaan yang berbeda.
4. Sistem ini tidak hanya menjawab kebutuhan wisatawan untuk menentukan destinasi yang relevan, tetapi juga memberikan dasar untuk pengembangan lebih lanjut dalam mendukung pariwisata Kota Bandung secara digital.
5. Dengan peningkatan fitur dan evaluasi berkelanjutan, sistem ini dapat menjadi komponen penting dalam layanan rekomendasi wisata berbasis data di tingkat kota.

---

> *Catatan gambar yang dapat ditambahkan:*
>
> * Gambar 1: Histogram distribusi rating wisatawan
> * Gambar 3: Grafik training/validation loss model CF
> * Gambar 4: Contoh hasil rekomendasi sistem CBF
> * Gambar 5: Contoh hasil rekomendasi sistem CF (user-based)
