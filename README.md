# Laporan Proyek Machine Learning – Rekomendasi Wisata Kota Bandung

![Ikon Kota Bandung](assets/iconic_bandung.jpeg)

## Domain Proyek

Bandung, ibu kota Provinsi Jawa Barat, dikenal sebagai salah satu kota wisata favorit di Indonesia. Dikenal dengan julukan “Paris van Java”, Bandung memiliki sejarah panjang sebagai kota tujuan wisata sejak zaman kolonial Belanda. Kombinasi antara iklim sejuk pegunungan, keragaman budaya Sunda, serta kemajuan urban menjadikannya tempat yang unik dan menarik. Berdasarkan data dari Dinas Kebudayaan dan Pariwisata Kota Bandung, kota ini menerima lebih dari 7 juta kunjungan wisatawan domestik dan internasional setiap tahunnya.

Bandung menawarkan beragam destinasi: dari wisata alam seperti Tebing Keraton dan Tangkuban Perahu, wisata edukasi dan sejarah seperti Museum Geologi dan Gedung Sate, hingga taman hiburan dan tempat belanja modern seperti Trans Studio Bandung dan kawasan Dago.

Namun, dengan lebih dari 400 lokasi wisata yang tersebar di seluruh kota dan sekitarnya, wisatawan sering mengalami kesulitan dalam memilih tujuan yang paling relevan dengan minat dan kebutuhannya. Oleh karena itu, pengembangan sistem rekomendasi wisata yang cerdas dan adaptif sangat penting.

Proyek ini mengembangkan sistem rekomendasi wisata menggunakan dua pendekatan utama: Content-Based Filtering (CBF) dan Collaborative Filtering (CF). Pendekatan ini mirip dengan bagaimana Machine Learning digunakan dalam bidang lain seperti prediksi risiko medis, namun disesuaikan untuk kebutuhan pariwisata. Rekomendasi diberikan berdasarkan kemiripan konten antar tempat wisata dan pola perilaku pengguna lain yang serupa, dengan tujuan meningkatkan pengalaman pengguna dan efisiensi dalam menentukan destinasi.

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

### Solution Approach

* **CBF:** Menggunakan cosine similarity antar TF-IDF deskripsi/kategori wisata.
* **CF:** Menggunakan deep learning embedding dan matrix factorization, serta confidence factor (jumlah tetangga, rerata rating pengguna dan tempat).

---



## Data Understanding

### Dataset:

* `tourism_with_id.csv`: informasi lengkap tempat wisata.
* `tourism_rating.csv`: data interaksi rating pengguna.
* `user.csv`: data demografi pengguna.

**Dataset diperoleh dari**: [Kaggle - Indonesia Tourism Destination Dataset](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination)

### Struktur Data:

#### Fitur pada `tourism_with_id.csv`:

* `Place ID`: ID tempat wisata
* `Place Name`: Nama tempat wisata
* `Category`: Kategori tempat (alam, budaya, dll)
* `City`: Kategori Kota 
* `Price Category`: Kategori harga
* `Description`: Deskripsi singkat tempat wisata
* `Time_Minutes`, `Unnamed: 11`, `Unnamed: 12`: kolom dihapus karena terlalu banyak missing

#### Fitur pada `tourism_rating.csv`:

* `User ID`: ID pengguna
* `Place ID`: ID tempat wisata yang dirating
* `Rating`: Skor penilaian 1–5

#### Fitur pada `user.csv`:

* `User ID`: ID pengguna
* `Username`: Nama pengguna 
* `Location`: Lokasi pengguna
* `Age`: Usia pengguna

### Penanganan Missing:

* Kolom dengan missing value tinggi seperti  `Time_Minutes`, `Unnamed: 11`, dan `Unnamed: 12` dihapus.
* Fokus fitur pada: `Place Name`, `Category`, `Description`, dan rating pengguna.
* Data numerik dan kategorikal penting dibersihkan untuk proses modeling.

### Insight Awal:

* Banyaknya tempat wisata tanpa kota menunjukkan kurangnya metadata geografis.
* Distribusi rating relatif merata di seluruh tempat.
* Umur pengguna memiliki distribusi dominan di usia 20–40 tahun.


### Visualisasi:

* ![Distribusi Rating](assets/rating_distribution.png)

  *Gambar 1. Histogram distribusi rating wisatawan.*


---

## Data Preparation

B

Berikut adalah tahapan lengkap dalam proses **persiapan data** sebelum digunakan dalam model Content-Based Filtering (CBF) dan Collaborative Filtering (CF):

| No | Langkah                  | Deskripsi                                                                                                                                |
| -- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | Data Cleaning            | Menghapus kolom `Time_Minutes`, `Unnamed: 11`, `Unnamed: 12` yang banyak kosong pada  |
| 2  | Filter Lokasi: Bandung   | Memfilter hanya tempat wisata yang berada di Kota Bandung berdasarkan kolom `City` dari `tourism_with_id.csv`.                           |
| 3  | Merge Dataset            | Menggabungkan: <br> - `tourism_rating.csv` dengan `tourism_with_id.csv` melalui `Place_Id` <br> - dan `user.csv` melalui `User_Id`.      |
| 4  | TF-IDF Vectorization     | Menggunakan `TfidfVectorizer()` pada kolom `Category` untuk menghasilkan vektor fitur tempat wisata.                                     |
| 5  | Cosine Similarity Matrix | Menghitung kemiripan antar tempat wisata menggunakan `cosine_similarity()` dari hasil TF-IDF.                                            |
| 6  | Persiapan Data untuk CF  | Menyiapkan kolom `User_Id`, `Place_Id`, dan `Rating` sebagai input ke dalam model Collaborative Filtering.                               |

---

### Kode dan Penjelasan Kode

#### 1. Data Cleaning

```python
place_df.drop(['Time_Minutes', 'Unnamed: 11', 'Unnamed: 12'], axis=1, inplace=True)
user_df = user_df.dropna(subset=['Age', 'HomeTown'])
```

Baris ini menghapus kolom dengan banyak missing value dari `place_df`, serta menghapus baris yang memiliki NA pada kolom penting di `user_df`.

#### 2. Filter Lokasi Bandung

```python
place_df = place_df[place_df['City'].str.contains("Bandung", na=False)]
```

Memfilter data agar hanya menyisakan tempat wisata yang berada di Bandung.

#### 3. Merge Dataset

```python
df_rating = pd.merge(df_rating, place_df[['Place_Id']], how='right', on='Place_Id')
df_user = pd.merge(user_df, df_rating[['User_Id']], how='right', on='User_Id').drop_duplicates().sort_values('User_Id')
```

Menggabungkan data rating dengan data tempat wisata dan data user untuk menyusun data lengkap yang siap dianalisis.

#### 4. TF-IDF Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(place_df['Category'])
```

Mengubah data kategori tempat wisata ke bentuk numerik vektor menggunakan TF-IDF.

#### 5. Cosine Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix)
```

Mengukur kesamaan antar tempat wisata berdasarkan hasil TF-IDF menggunakan cosine similarity.

#### 6. Split untuk CF

```python
from sklearn.model_selection import train_test_split

X = df_rating[['User_Id', 'Place_Id']].values
y = df_rating['Rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Membagi data rating menjadi data pelatihan dan data uji untuk digunakan dalam model Collaborative Filtering.

---

### Catatan Penting

* Proses penggabungan (merge) dilakukan untuk memastikan hanya tempat wisata yang memiliki data lengkap dan relevan yang digunakan.
* Proses TF-IDF dan cosine similarity digunakan untuk membangun model Content-Based Filtering.
* Data rating yang sudah disiapkan akan digunakan sebagai input untuk model Collaborative Filtering.

---


## Modeling

### 1. Content-Based Filtering (CBF)

Content-Based Filtering (CBF) merekomendasikan item (dalam hal ini tempat wisata) berdasarkan kemiripan kontennya. Dalam proyek ini, pendekatan CBF dilakukan dengan:

* Menggunakan **TF-IDF** untuk mengubah data kategori tempat wisata menjadi representasi vektor numerik.
* Menggunakan **Cosine Similarity** untuk menghitung kemiripan antar tempat wisata berdasarkan vektor hasil TF-IDF.

#### Rumus TF-IDF:

$$
\text{TF-IDF}(t, d) = \text{tf}(t, d) \times \log\left(\frac{N}{df(t)}\right)
$$

Keterangan:

* $tf(t, d)$: frekuensi kemunculan term $t$ pada dokumen $d$
* $df(t)$: jumlah dokumen yang mengandung term $t$
* $N$: total jumlah dokumen

#### Rumus Cosine Similarity:

$$
\text{cosine}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}
$$

Keterangan:

* $A, B$: vektor representasi dua tempat wisata
* $A \cdot B$: dot product antara A dan B
* $\|A\|$: panjang (norma) vektor A

#### Cuplikan kode dari notebook:

```python
tfidf_vectorizer_model = TfidfVectorizer(max_features=5000)
```

**Penjelasan**:

* Menginisialisasi objek `TfidfVectorizer` dengan maksimal 5000 fitur kata.
* Objek ini digunakan untuk mengubah teks kategori tempat wisata menjadi matriks TF-IDF.

```python
cosine_sim = cosine_similarity(tfidf_matrix)
```

**Penjelasan**:

* Menghitung skor kemiripan antar tempat wisata dari matriks TF-IDF.
* Output-nya adalah matriks 2D yang menunjukkan tingkat kemiripan antara semua kombinasi tempat wisata.

---

### 2. Collaborative Filtering (CF)

Collaborative Filtering (CF) memanfaatkan data interaksi antara pengguna dan item untuk melakukan rekomendasi. Teknik ini tidak memperhatikan konten item, tapi mengandalkan kesamaan perilaku pengguna.

Dalam proyek ini digunakan pendekatan berbasis deep learning dengan model **RecommenderNet** yang dibangun menggunakan TensorFlow.

#### Struktur Model:

```python
class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_places, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = layers.Embedding(num_users, embedding_size, embeddings_initializer="he_normal")
        self.user_bias = layers.Embedding(num_users, 1)
        self.place_embedding = layers.Embedding(num_places, embedding_size, embeddings_initializer="he_normal")
        self.place_bias = layers.Embedding(num_places, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        place_vector = self.place_embedding(inputs[:, 1])
        place_bias = self.place_bias(inputs[:, 1])

        dot_user_place = tf.tensordot(user_vector, place_vector, 2)

        return dot_user_place + user_bias + place_bias
```

#### Penjelasan kode:

* `Embedding`: memetakan ID pengguna dan ID tempat menjadi representasi vektor berdimensi tetap (`embedding_size`).
* `dot_user_place`: operasi dot product antara embedding pengguna dan tempat, menghasilkan prediksi rating.
* `bias`: bias tambahan untuk memperhalus prediksi rating.

#### Rumus estimasi rating dalam CF:

$$
\hat{r}_{u, i} = \mathbf{p}_u \cdot \mathbf{q}_i + b_u + b_i
$$

Keterangan:

* $\mathbf{p}_u$: vektor embedding pengguna $u$
* $\mathbf{q}_i$: vektor embedding tempat $i$
* $b_u$: bias pengguna
* $b_i$: bias tempat

---

### Confidence Scoring (Margaris et al., 2025)

Confidence score digunakan untuk mengukur kepercayaan terhadap rekomendasi, berdasarkan:

* Jumlah tetangga (pengguna yang memberikan rating)
* Rata-rata rating pengguna
* Rata-rata rating tempat

Tujuan utama dari confidence scoring adalah memberikan bobot tambahan pada prediksi model agar lebih realistis dan relevan.

---

### Evaluasi Model

Untuk mengevaluasi performa model, digunakan metrik **Root Mean Squared Error (RMSE)**:

$$
RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
$$

Keterangan:

* $y_i$: rating aktual
* $\hat{y}_i$: rating hasil prediksi
* $n$: jumlah sampel

---


## Evaluation

### Evaluasi Content-Based Filtering (CBF)

Untuk metode CBF atau Content-Based Filtering, sistem memberikan rekomendasi berdasarkan kemiripan konten dari tempat wisata. Ini termasuk kategori dan deskripsi teks dari setiap tempat. Karena sistem ini tidak melibatkan data interaksi pengguna seperti rating secara eksplisit, maka evaluasi dilakukan secara kualitatif.

#### Evaluasi Kualitatif

* Relevansi dievaluasi dengan melihat apakah tempat wisata yang direkomendasikan memiliki kategori atau deskripsi yang mirip.
* Sistem dievaluasi dengan cara manual: apakah rekomendasinya logis jika dilihat dari sudut pandang wisatawan.
* Misalnya, jika pengguna memilih Trans Studio Bandung, maka rekomendasi ideal adalah taman hiburan atau pusat hiburan serupa, bukan tempat yang sama sekali berbeda jenis.

#### Kesimpulan CBF

Metode ini sangat cocok untuk pengguna baru yang belum memiliki histori penilaian. Hal ini dikenal sebagai solusi untuk masalah cold start user.

---

### Evaluasi Collaborative Filtering (CF)

Pada metode Collaborative Filtering, sistem menggunakan pendekatan berbasis neural network, yaitu model RecommenderNet. Model ini dibuat untuk mempelajari pola rating dari pengguna terhadap tempat wisata dan memprediksi rating pada tempat yang belum pernah dikunjungi oleh pengguna tersebut.

#### Evaluasi Kuantitatif dengan RMSE

Root Mean Squared Error atau RMSE adalah salah satu metode evaluasi paling umum untuk model prediksi. RMSE mengukur seberapa jauh nilai prediksi dari nilai sebenarnya. Semakin kecil nilai RMSE, maka semakin baik performa model.

Rumus RMSE menghitung rata-rata dari kuadrat selisih antara rating sebenarnya dengan rating hasil prediksi. Setelah itu diambil akar dari hasil tersebut.

Pada proyek ini, model dihentikan secara otomatis jika nilai RMSE pada data validasi sudah mencapai kurang dari nol koma dua lima. Ini dilakukan dengan callback atau fungsi penghentian otomatis.

#### Callback Kode Evaluasi

Berikut adalah kode program yang digunakan untuk menghentikan pelatihan model jika sudah mencapai target RMSE:

```python
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_root_mean_squared_error') < 0.25:
            print('Lapor! Metrik validasi sudah sesuai harapan')
            self.model.stop_training = True
```

Penjelasan:

* Baris pertama membuat kelas callback dari library TensorFlow.
* Fungsi on\_epoch\_end dijalankan setiap kali model selesai satu kali pelatihan penuh atau epoch.
* Jika nilai RMSE pada validasi kurang dari nol koma dua lima, pelatihan akan dihentikan dan model dianggap cukup baik.

#### Visualisasi Grafik Evaluasi

Model juga dievaluasi melalui grafik, yang menunjukkan nilai RMSE pada data pelatihan dan data validasi.

```python
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('Model Evaluation')
plt.ylabel('Root Mean Squared Error')
plt.xlabel('Epoch')
plt.ylim(ymin=0, ymax=0.4)
plt.legend(['Train', 'Validation'], loc='center left')
plt.show()
```

Penjelasan:

* Dua garis digambar, masing-masing untuk data pelatihan dan data validasi.
* Grafik ini membantu melihat apakah model mengalami overfitting atau tidak.
* Jika kedua garis berada dalam tren menurun dan saling berdekatan, maka model dianggap stabil.

---

### Hasil Akhir Evaluasi

Model Collaborative Filtering menunjukkan hasil yang baik dengan nilai RMSE di bawah nol koma dua lima. Artinya, model memiliki kemampuan prediksi yang baik dan dapat merekomendasikan tempat yang relevan dengan preferensi pengguna berdasarkan histori rating mereka.


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

* Tercapai. Evaluasi menunjukkan performa memuaskan dan sesuai ekspektasi.

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

1. Sistem rekomendasi wisata berbasis ML berhasil dibangun menggunakan dua pendekatan utama: Content-Based Filtering dan Collaborative Filtering.
2. Evaluasi model menunjukkan:

   * **CBF** efektif untuk pengguna baru (cold-start), memberikan rekomendasi berdasarkan konten yang relevan.
   * **CF** unggul untuk pengguna lama dengan personalisasi tinggi dan akurasi prediksi yang baik (RMSE < 0.25).
   
3. Proyek ini telah berhasil menjawab permasalahan bisnis dalam memberikan rekomendasi wisata yang lebih akurat dan relevan di Kota Bandung.
4. Sistem dapat ditingkatkan lebih lanjut dengan pendekatan hybrid, penguatan metadata, dan validasi langsung dari pengguna nyata.
5. Rekomendasi ini berpotensi besar dalam mendukung pertumbuhan pariwisata lokal melalui sistem cerdas yang adaptif, kontekstual, dan berbasis data.

---
