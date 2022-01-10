# %% [markdown]
# 
# # Praktikum VIII
# 
# Selamat datang pada sesi ke-8 praktikum unggulan Universitas Gunadarma. Melanjutkan kegiatan praktikum sebelumnya, pada minggu ini Anda akan melanjutkan tahapan pengolahan data yaitu data training dan model generation. 
# 
# ### Dataset
# Dataset yang Anda gunakan yaitu dataset Beton (concrete) yang berisi informasi mengenai komposisi beton, pengaruh waktu dan hasil kekuatan beton yang berisi 1030 row data dengan 9 fitur. 
# 
# ### Studi Kasus
# Dengan menggunakan dataset tersebut Anda akan diminta menyelesaikan permasalahan klasifikasi, dimana model akan diminta untuk menentukan kekuatan beton berdasarkan kombinasi fitur yang dimiliki. 
# 
# ### Algoritma
# Algoritma yang digunakan adalah K-Nearest Neighbour.

# %% [markdown]
# # Tugas 
# 
# Buatlah laporan akhir yang menjelaskan langkah-langkah serta penjelasan tahapan yang Anda lakukan pada praktikum ini, yang didalamnya menjawab beberapa pertanyaan sebagai berikut
# 1. Rubahlah fitur cement menjadi fitur target class yang ingin Anda prediksi, dengan nilai fitur > 281 merupakan kelas 1 dan < 281 menjadi nilai 0. Berapa akurasi model dan nilai k untuk skenario ini?
# 2. Pada hasil diagram heatmap, terlihat banyak konfigurasi warna dan nilai yang ditampilkan.  Jelaskan arti kolom berwarna merah pada hasil diagram heatmap yang dihasilkan! Pasangan variabel apa saja yang bernilai merah?
# 3. Apa yang terjadi jika, nilai test set diganti menjadi 0.3 kemudian 0.2 dan 0,1 ? 
# 4. Buatlah tabel yang berisi hasil pengujian Anda dan sertakan berapa nilai k dan nilai akurasi untuk masing-masing skenario uji.
# 5. Apakah terjadi perbedaan nilai akurasi model? Jika Ya, jelaskan pendapat Anda mengapa hal tersebut bisa terjadi.
# 6. Berapa nilai presisi dan recall dari masing-masing model yang Anda coba serta jelaskan arti dari nilai tersebut.

# %% [markdown]
# # Import Dataset
# 

# %%
# Import library
import pandas as pd
import numpy as np


# %%
#### import dataset 
data = pd.read_csv('https://gitlab.com/andreass.bayu/file-directory/-/raw/main/new_concreate.csv')

# Lihat 5 data awal
data.head(5)

# %% [markdown]
# # Review Dataset

# %%
# Lihat deskripsi dari tiap kolom
data.describe()

# %%
# Lihat tipe data dari tiap kolom
data.dtypes

# %%
# mlihat jumlah atribut dan data / dimensi data
data.shape

# %%
# hitung dan melihat jumlah data per label kelas
for col in data.columns:
    print('Attribute name:',col)
    print('-------------------')
    print(data[col].value_counts())
    print('-------------------')

# %%
#import library seaborn untuk visualisasi
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Plot figure untuk menentukan distribusi kelas
plt.figure(figsize=(8,5))

# menghitung baris setiap kelas
sns.countplot(x="coarseagg", data=data)

# %% [markdown]
# #### *Dataset memiliki distribusi nilai untuk kelas coarseagg yang beragam.*

# %% [markdown]
# # Dataset preparation

# %%
# Buat salinan dataframe
df = data.copy(deep = True)

# mengubah/convert nilai "?" nilai ke bentuk Na / NaN untuk diproses lebih lanjut
for col in data.columns:
  df[[col]] = data[[col]].replace('?',np.NaN)


# %%

# seleksi kolom fitur/feature columns dari dataset
null_data = df.iloc[:,:-4]

# temukan nilai null untuk semua atribut dan jumlahkan total nilai null
null_data.isnull().sum()

# %%

# jatuhkan/drop semua baris yang memiliki nilai null
df = df.dropna()

# pilih kolom fitur/feature columns dari dataset
null_data = df.iloc[:,:-4]

# cek ulang nilai null
null_data.isnull().sum()


# %% [markdown]
# **StandardScaler** *adalah class dari sklearn untuk melakukan normalisasi data agar data yang digunakan tidak memiliki penyimpangan yang besar.*

# %%
# Import library standard scaler 

from sklearn.preprocessing import StandardScaler

# Buat dataframe dengan tipe data int64
colname= []
for col in df.columns:
  if df[col].dtype == "int64":
      colname.append(col)

# Buat salinan dataset untuk keperluan persiapan data / data preparation
df_copy = df.copy(deep = True)
df_fe = df.copy()


# Buat kerangka data untuk fitur kategoris / categorical features
df_fe.drop('coarseagg',axis='columns', inplace=True)
df_fe.drop(colname,axis='columns', inplace=True)


# buat dataframe untuk kelas target / target class
df_cl = df.copy()
df_cl.drop(df_copy.iloc[:,:-4],axis='columns', inplace=True)



# membuat objek scaler / scaler object
std_scaler = StandardScaler()
std_scaler


# Normalisasikan atribut numerik dan tetapkan ke dalam dataframe baru
df_norm = pd.DataFrame(std_scaler.fit_transform(df_copy[colname]), columns=colname)

# %% [markdown]
# Mengapa Menskalakan data untuk KNN?
# Ketika Anda menggunakan algoritma berbasis jarak seperti KNN, proses penyesuaian skala data sangatlah penting karena akan mempengaruhi nilai dari proses perhitungan jarak dilakukan dengan skala yang sama.
# 
# Sebagai ilustrasi perhitungan jarak menggunakan dua fitur yang besaran/rentangnya sangat bervariasi dengan menggunakan perhitungan jarak eucledian.
# Jarak Euclidean = [(100000–80000)^2 + (30–25)^2]^(1/2)
# 
# fitur dengan jangkauan yang lebih besar akan mendominasi atau mengecilkan fitur yang lebih kecil sepenuhnya dan ini akan berdampak pada kinerja semua model berbasis jarak karena akan memberikan bobot yang lebih tinggi pada variabel yang memiliki nilai yang lebih tinggi.¶

# %%
# import library Ordinal Encoder dari package library sklearn.preprocessing 
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()

# enconde fitur kategoris/categorical features menjadi fitur numerik/numerical features   
for col in df_fe.columns[:]:
  if df_fe[col].dtype == "object":
    df_fe[col] = ord_enc.fit_transform(df_fe[[col]])
    

# %% [markdown]
# #### 1. Nilai fitur > 281 merupakan kelas 1 dan < 281 menjadi nilai 0 dari target class coarseagg
# 

# %%
# Melakukan proses encoding untuk mengubah fitur coarseagg menjadi nilai biner. 
df_cl["coarseagg"] = np.where(df_cl["coarseagg"]<281, 0, 1)    

# %%
# Masukkan kolom id ke datasets yang berbeda
df_norm.insert(0, 'id', range(0, 0 + len(df_norm)))
df_fe.insert(0, 'id', range(0, 0 + len(df_fe)))
df_cl.insert(0, 'id', range(0, 0 + len(df_cl)))

# Lihat shapes datasets yang telah di proses 
print(df_norm.shape)
print(df_fe.shape)
print(df_cl.shape)

# %%
# Gabungkan semua datasets
df_feature = pd.merge(df_norm,df_fe, on=["id"])
df_final = pd.merge(df_feature,df_cl, on=["id"])

# Drop kolom id dari gabungan dataset
df_final.drop('id',axis='columns', inplace=True)

# Lihat 5 data awal dari gabungan dataset
df_final.head(5)

# %% [markdown]
# # Visualization

# %%
p = df_final.hist(figsize = (20,20))

# %% [markdown]
# **Scatter matrix plot adalah** plot yang digunakan untuk membuat sekumpulan scatter plot dari beberapa pasang variabel. Hal ini sangat bermanfaat terutama ketika ingin menganalisis bagaimana bentuk hubungan antar variabel. Plot ini sangat bermanfaat untuk digunakan untuk data yang ukurannya tidak terlalu besar. Untuk menggunakan scatter matrix kita harus memanggil fungsi *scatter_matrix* dari *pandas.plotting*

# %%
from pandas.plotting import scatter_matrix

p=scatter_matrix(df_final,figsize=(25, 25))

# %%
# Buat visualisasi korelasi data dengan heatmap
import seaborn as sns
import matplotlib.pyplot as plt

 # plot heatmap
plt.figure(figsize=(12,10))
p=sns.heatmap(df_final.corr(), annot=True,cmap ='RdYlGn')

# %% [markdown]
# #### 2. Koefisien Korelasi Pearson: 
# 
# keterhubungan ter-erat pada heatmap
# * slag dan cement mencapai nilai -0.62
# * fineagg dan water mencapai nilai -0.51

# %% [markdown]
# # Proses Modelling dengan KNN 

# %%
## menampilkan 5 data awal dari dataset yang akan digunakan
df_final.head(5)

# %%
## memisahkan data fitur dengan label yang akan di pelajari
train_data = df_final.drop("coarseagg",axis = 1)
train_data.head()

# %%
## memisahkan data label dengan label yang akan di pelajari
df_final.coarseagg = df_final.coarseagg.astype(np.int64)
label_data = df_final.coarseagg
label_data.head()

# %%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train_data,label_data,test_size=0.4,random_state=42)

# %% [markdown]
# # Proses Training 
# 
# ##### Train Test Split 
# Train dan test split dimana kita akan memisahkan data yang akan digunakan oleh algoritma dalam proses training dan testing. Proses ini akan membantu kita melakukan evaluasi terhadap model machine learning yang dibangun.
# 
# 
# 
# ##### Cross Validation
# Ketika model dipecah menjadi pelatihan dan pengujian, ada kemungkinan bahwa jenis titik data tertentu dapat sepenuhnya menjadi bagian pelatihan (training) atau pengujian (testing). Hal ini akan menyebabkan model berkinerja buruk. Oleh karena itu masalah over-fitting dan underfitting dapat dihindari dengan baik dengan teknik validasi silang (cross validation)
# 
# 
# 
# ##### Stratify 
# Parameter stratify membuat pembagian sehingga proporsi nilai dalam sampel yang dihasilkan akan sama dengan proporsi nilai yang diberikan untuk stratifikasi parameter.
# 
# Misalnya, jika variabel y adalah variabel kategori biner dengan nilai 0 dan 1 dan ada 25% dari nol dan 75% dari satu, stratify=y akan memastikan bahwa pembagian acak Anda memiliki 25% dari 0 dan 75% dari 1.
# 
# For Reference : https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

# %%
# import library model KNN dengan alias/as 'KNeighborsClassifier'
from sklearn.neighbors import KNeighborsClassifier 
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

# buat variabel kosong untuk menyimpan metrik KNN/KNN metrics
scores=[]
# Kita coba nilai k yang berbeda untuk KNN (dari k=1 sampai k=26)
lrange=list(range(1,25))
# loop proses KNN
for k in lrange:
  # masukkan nilai k dan ukuran 'jarak'
  knn=KNeighborsClassifier(n_neighbors=k)
  # masukan data train/ data latih untuk melatih KNN
  knn.fit(X_train,y_train.ravel())
  # lihat prediksi KNN dengan memasukkan data uji/data test
  y_pred=knn.predict(X_test)
  # tambahkan performance metric akurasi
  scores.append(metrics.accuracy_score(y_test,y_pred))
plt.figure(2,figsize=(15,5))


optimal_k = lrange[scores.index(max(scores))]
print("Nilai k KNN yang optimal adalah %d" % optimal_k)
print("Skor optimalnya adalah %.2f" % max(scores))


# plot hasilnya
plt.plot(lrange, scores,ls='dashed')
plt.xlabel('Nilai dari k untuk KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores untuk nilai k dari k-Nearest-Neighbors')
plt.show()

# %%
knn = KNeighborsClassifier(13)

knn.fit(X_train,y_train)
knn.score(X_test,y_test)

# %% [markdown]
# # Evaluasi Hasil Matriks
# 
# 
# 

# %%
y_pred = knn.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# %%
#import classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

# %% [markdown]
# #### 3. Nilai test 0.3, 0.2 dan 0.1
# 
# * Nilai test 0.3 pada kelas 0 mendapat nilai support sebanyak 68 sample
# * Nilai test 0.2 pada kelas 0 mendapat nilai support sebanyak 45 sample
# * Nilai test 0.1 pada kelas 0 mendapat nilai support sebanyak 23 sample
# 
# tetapi karena nilai fiturnya 281, nilai precision, recall, f1-score akan tetap sama di 1.00

# %% [markdown]
# #### 4. Nilai k dan akurasi

# %%
print("Nilai k KNN yang optimal adalah %d" % optimal_k)
print("Skor optimalnya adalah %.2f" % max(scores))

# %% [markdown]
# #### 5. Perbedaan akurasi model
# 
# * Terjadi perbedaan pada data test 0.4, 0.3, 0.2 dan 0.1
# 
# * Test Set 0.4 dimana nilai ini merepresentasikan berapa perbandingan nilai Test set dan Training Set yang digunakan, lalu membagi/split 60% dari data menjadi data Training Set, dan 40% menjadi Data Test
# 
# * jika lebih sedikit Data Training, estimasi parameter memiliki varians yang lebih besar. jika lebih sedikit Testing Data, statistik kinerja Anda akan memiliki varians yang lebih besar.

# %% [markdown]
# #### 6. Nilai precision dan recall
# 
# <h5 align="center">40% Data Test</h5> 
# 
# |  | precision | recall | f1-score | suppport |
# |------|------|------|------|------|
# | 0 | 1.00 | 1.00 | 1.00 | 90 |
# | accuracy |  |  | 1.00 | 90 |
# | macro avg | 1.00 | 1.00 | 1.00 | 90 |
# | weigted avg | 1.00 | 1.00 | 1.00 | 90 |
# 
# 
# <h5 align="center">30% Data Test</h5> 
# 
# |  | precision | recall | f1-score | suppport |
# |------|------|------|------|------|
# | 0 | 1.00 | 1.00 | 1.00 | 68 |
# | accuracy |  |  | 1.00 | 68 |
# | macro avg | 1.00 | 1.00 | 1.00 | 68 |
# | weigted avg | 1.00 | 1.00 | 1.00 | 68 |
# 
# 
# <h5 align="center">20% Data Test</h5> 
# 
# |  | precision | recall | f1-score | suppport |
# |------|------|------|------|------|
# | 0 | 1.00 | 1.00 | 1.00 | 45 |
# | accuracy |  |  | 1.00 | 45 |
# | macro avg | 1.00 | 1.00 | 1.00 | 45 |
# | weigted avg | 1.00 | 1.00 | 1.00 | 45 |
# 
# 
# <h5 align="center">10% Data Test</h5> 
# 
# |  | precision | recall | f1-score | suppport |
# |------|------|------|------|------|
# | 0 | 1.00 | 1.00 | 1.00 | 23 |
# | accuracy |  |  | 1.00 | 23 |
# | macro avg | 1.00 | 1.00 | 1.00 | 23 |
# | weigted avg | 1.00 | 1.00 | 1.00 | 23 |

# %% [markdown]
# * Nilai Precision dan Recall pada setiap model tetap sama di nilai 1.00 karena nilai fiturnya 281
#  
# * Nilai Precision adalah rasio pengamatan positif yang diprediksi dengan benar dengan total pengamatan positif yang diprediksi. 
#  
# * Nilai Recall adalah rasio pengamatan positif yang diprediksi dengan benar dengan semua pengamatan di kelas yang sebenarnya. 

# %%



