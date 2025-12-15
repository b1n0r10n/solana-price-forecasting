# Laporan Proyek Machine Learning - Bintang Akalla Junjunan

## ***Domain Project***

Cryptocurrency telah menjadi salah satu instrumen investasi yang populer dalam beberapa tahun terakhir. Salah satu cryptocurrency yang menunjukkan pertumbuhan signifikan adalah Solana (SOL). Solana dikenal dengan kecepatan transaksi yang tinggi dan biaya yang rendah, membuatnya menarik bagi pengembang dan investor. Namun, seperti kebanyakan cryptocurrency, harga Solana sangat volatil, sehingga sulit bagi investor untuk memprediksi pergerakan harganya.

Volatilitas harga Solana disebabkan oleh berbagai faktor, termasuk sentimen pasar, perkembangan teknologi, regulasi, dan aktivitas perdagangan. Untuk meminimalkan risiko dan memaksimalkan keuntungan, investor memerlukan alat prediksi yang andal. Machine learning dan deep learning menawarkan pendekatan yang potensial dalam memodelkan dan memprediksi harga aset finansial dengan menganalisis data historis dan mengenali pola yang kompleks.

Menurut penelitian oleh Zhang et al. (2020), metode deep learning seperti Long Short-Term Memory (LSTM) telah menunjukkan performa yang baik dalam memprediksi data time series finansial. Hal ini menjadikan LSTM sebagai kandidat yang menjanjikan untuk prediksi harga cryptocurrency seperti Solana.

Proyek ini bertujuan untuk mengembangkan model prediksi harga harian Solana menggunakan pendekatan machine learning dan deep learning. Dengan menganalisis data historis dan mengenali pola kompleks, diharapkan model ini dapat membantu investor dalam membuat keputusan investasi yang lebih baik, meminimalkan risiko, dan memaksimalkan keuntungan.

Referensi:

* Zhang, Y., Li, P., Wang, S., & Shen, D. (2020). Financial time series forecasting with deep learning: A systematic literature review. IEEE Access, 8, 181447-181468.

## ***Business Understanding***

Dalam proyek ini, tujuan utama adalah menerapkan metode machine learning untuk memprediksi harga penutupan harian Solana, sehingga dapat membantu investor dalam pengambilan keputusan yang lebih baik.

***Problem Statements***

Dalam konteks ini, kami mengidentifikasi beberapa pertanyaan kunci yang perlu dijawab untuk mencapai tujuan proyek:

* Bagaimana membangun model machine learning yang dapat memprediksi harga harian Solana?

* Algoritma machine learning apa yang paling efektif untuk memprediksi harga Solana berdasarkan data historis?

* Seberapa akurat model prediktif dalam memprediksi harga Solana dibandingkan dengan model baseline?

***Goals***

Berdasarkan problem statements di atas, tujuan utama proyek ini adalah sebagai berikut:

* Membangun model prediktif yang mampu memprediksi harga harian Solana.

* Mengidentifikasi algoritma machine learning yang paling efektif untuk prediksi harga Solana.

* Mengevaluasi model menggunakan MAE dan RMSE

***Solution Statements***

Untuk mencapai tujuan tersebut, kami merumuskan beberapa solusi berikut:

* Menggunakan algoritma Long Short-Term Memory (LSTM) untuk memodelkan data harga historis Solana. LSTM dipilih karena kemampuannya dalam menangani data time series dan mengenali pola jangka panjang.

* Menggunakan Regresi Linear sebagai model baseline untuk membandingkan efektivitas model LSTM.

* Mengukur performa model menggunakan metrik evaluasi seperti Mean Absolute Error (MAE) dan Root Mean Squared Error (RMSE), yang sesuai untuk mengukur kesalahan prediksi dalam konteks data kontinu seperti harga.

## ***Data Understanding***

Data yang digunakan dalam submission ini adalah data historis harga Solana (SOL) yang diperoleh dari [investing.com](https://www.investing.com/crypto/solana/historical-data). Data mencakup periode dari 13 Juli 20240 hingga 5 November 2024

Informasi Dataset:
* Jumlah data lebih dari 1500 sampel
* Beberapa fitur dengan format yang masih perlu dibersihkan

Dataset terdiri dari fitur-fitur berikut:
* Date : Tanggal pencatatan
* Price : Harga pada hari tersebut
* Open : Harga pembukaan pada hari tersebut
* High : Harga tertinggi pada hari tersebut
* Low : Harga terendah pada hari tersebut
* Volume : Volume perdagangan pada hari tersebut

***Load Data***

Pada langkah ini, kita melakukan analisis ekplorasi data untuk memahami struktur, pola, dan anomali dalam dataset. Dari `df.info()` menunjukkan informasi umum dari dataset, dimana beberapa kolom ada yang bertipe object dan float64, serta menunjukkan jumlah baris dalam kolom

***Exploratory Data Analysis (EDA)***

Pada langkah ini, kita melakukan mengeksplorasi dan mencoba memahami dataset yang digunakan

* ***Checking Missing Value:***

| RangeIndex: 1570 entries, 0 to 1569 |          |                |         |
|-------------------------------------|----------|----------------|---------|
| Data columns (total 7 columns):     |          |                |         |
| #                                   | Column   | Non-Null Count | Dtype   |
| ---                                 | ------   | -------------- | -----   |
| 0                                   | Date     | 1570 non-null  | object  |
| 1                                   | Price    | 1570 non-null  | float64 |
| 2                                   | Open     | 1570 non-null  | float64 |
| 3                                   | High     | 1570 non-null  | float64 |
| 4                                   | Low      | 1570 non-null  | float64 |
| 5                                   | Vol.     | 1172 non-null  | object  |
| 6                                   | Change % | 1570 non-null  | object  |
| dtypes: float64(4), object(3)       |          |                |         |
| memory usage: 86.0+ KB              |          |                |         |

Setelah melihat informasi umum di atas, dapat dilihat bahwa kolom Vol. itu memiliki jumlah data 1172, yang  dimana ini berbeda dari kolom-kolom lainnya, oleh karena itu kita akan melakukan cek apakah terdapat missing value atau tidak. Hasil dari pengecekan menunjukkan bahwa kolom Vol. itu memiliki 398 missing value, oleh karena itu ini akan di proses lebih lanjut lagi 

| Date         | 0   |
|--------------|-----|
| Price        | 0   |
| Open         | 0   |
| High         | 0   |
| Low          | 0   |
| Vol.         | 398 |
| Change %     | 0   |
| dtype: int64 |     |

Selanjutnya kita melakukan konversi kolom Vol. dan Change% ke dalam bentuk numerik, hal ini dilakukan disini karena pada tahap-tahap selanjutnya mengharuskan dua kolom ini berbentuk numerik. Kolom Vol. dikonversi menjadi string terlebih dahulu agar dapat mengubah 'M' dan 'K' dengan 'e6' dan 'e3' untuk notasi ilmiah, setelah ini menghapus koma pada kolom Vol.(jika ada). Kolom Vol. dikonversi menjadi nilai numerik(float). Kolom 'Change%' sama seperti kolom 'Vol.' dimana akan dikonversi ke string terlebih dahulu untuk menghilangkan karakter '%' sebelum dikonversi ke dalam bentuk numerik.

* ***Distribusi Volume Perdagangan:***

    Disini kita melakukan visualisasi untuk distribusi volume perdagangan Solana serta menambahkan informasi statistik deskriptif, disini dapat diketahui bahwa volume paling besar dari perdagangan Solana sebensar 3.977200e+08 dan volume pergadangan Solana paling kecil adalah 8.500000e+04. Frekuensi volume perdagangan Solana paling banyak berada di kisaran 4840000 yang ditunjukkan pada keterangan pada hasil visualisasinya.

![f78785b4-063a-44fd-8ad4-319ccf073dc6](https://github.com/user-attachments/assets/e0e4c873-18b9-42df-abf7-4f2cdf9b1572)

 | Statistik deskriptif untuk 'Vol.': |              |
|------------------------------------|--------------|
| count                              | 1.172000e+03 |
| mean                               | 1.417380e+07 |
| std                                | 3.923494e+07 |
| min                                | 8.500000e+04 |
| 25%                                | 3.537500e+06 |
| 50%                                | 5.290000e+06 |
| 75%                                | 8.462500e+06 |
| max                                | 3.977200e+08 |
| Name: Vol., dtype: float64         |              |
| Mode: 4840000.0                    |              |

* ***Checking Outliers:*** 

    Pada tahap ini kita akan melakukan pengecekan outliers menggunakan boxplot dan menggunakan perhitungan, dari hasil visualisasi ditunjukkan bahwa pada data numerik itu tidak ada outliers, hanya saja distribusinya yang tidak seimbang.

    ![69f74347-c187-4209-9091-95d823b5b05a](https://github.com/user-attachments/assets/f8f29dfa-edb3-4439-b547-cb1dd71c09e7)

    Untuk memastikan lebih lanjut pengecekan outlier digunakan perhitungan seperti berikut, dan hasilnya menunjukkan data numerik ini tidak ada outliers.

***Univariate Analysis***

* ***Visualisasi Distribusi Harga Solana***

    Langkah pertama dalam memahami data harga Solana adalah dengan melihat distribusinya, histogram ini memberikan gambaran yang jelas tentang seberapa sering harga-harga tertentu muncul dalam dataset yang kita miliki.

    ![1d575a2b-c2fc-4eab-b41a-4fe0a2fac4d3](https://github.com/user-attachments/assets/13a83817-d4e7-48d5-83d6-b8a9d06d716a)

    Dari histogram, kita dapat melihat bahwa harga Solana cenderung terkonsentrasi pada lebih lebih rendah yaitu dengan nilai modus 1.79, dengan beberapa puncak di sepanjang spektrum harga.Frekuensi yang tinggi di harga rendah menunjukkan bahwa harga ini sering muncul dalam riwayat data Solana. Solana pernah menyentuh harga 258.477.

| Statistik deskriptif untuk 'Price': |             |
|-------------------------------------|-------------|
| count                               | 1570.000000 |
| mean                                | 67.556690   |
| std                                 | 62.992891   |
| min                                 | 1.004000    |
| 25%                                 | 20.007000   |
| 50%                                 | 35.518000   |
| 75%                                 | 126.452000  |
| max                                 | 258.477000  |
| Name: Price, dtype: float64         |             |
| Mode:                               | 1.787       |

* ***Visualisasi Distribusi  Persentase Perubahan Harga Harian Solana***

    Selanjutnya kita akan melihat visualisasi dari distribusi persentase perubahan harga harian Solana. Histogram ini memberikan wawasan tentang volatilitas harga harian Solana dari hari ke hari.

    Distribusi perubahan harga harian Solana terlihat simetris dan cenderung membentuk distribusi normal, dengan puncak di -3.13 persen yang ditunjukkan dari perhitungan dibawah gambar visualisasi. Ini menunjukkan bahwa sebagain besar perubahan harga harian berkisar di sekitar nilai stabil. Solana pernah mengalami kenaikan yang cukup signifikan dalam sehari yaitu sebesar 64.83000. Ekor yang lebih panjang di sisi negatif dan positif menunjukkan adanya hari-hari tertentu yang mengalam perubahan harga yang lebih ekstrem.

    ![84fe2c43-6c5b-454d-ab44-146ca4c8777c](https://github.com/user-attachments/assets/0ed7f5a7-f70d-4962-a7cf-e6d808a33782)

| Statistik deskriptif untuk 'Change %': |             |
|----------------------------------------|-------------|
| count                                  | 1570.000000 |
| mean                                   | 0.564102    |
| std                                    | 7.010851    |
| min                                    | -42.350000  |
| 25%                                    | -3.120000   |
| 50%                                    | 0.050000    |
| 75%                                    | 3.540000    |
| max                                    | 64.830000   |
| Name: Change %, dtype: float64         |             |
| Mode:                                  | -3.13       |

***Multivariate Analysis***

* ***Visualisasi Correlation Heatmap***

    ![image](https://github.com/user-attachments/assets/1998998d-9d39-4012-861e-bdb20ce99afe)

    Pada tahap ini kita menggunakan visualisasi heatmap untuk mempelajari hubungan antar variabel utama seperti `Price`, `Open`, `High`, `Low`, `Vol.`, dan `Change %.`, ini memberikan gambaran tentang seberapa kuat hubungan antaraa dua variabel dalam dataset.

    terlihat bahwa variabel-variabel harga memiliki korelasi yang hampir sempura, mendekati 1. Hal ini umum dalam data harga keuangank karena harga pembukaan, tertinggi, terendah, dan penutup biasanya bergerak bersamaan dalam periode waktu yang sama.

    Korelasi antara `Vol.` dan variabel-variabel harga lainnya sangat rendah, menunjukkan bahwa volume perdagangan tidak memiliki hubungan linear yang kuat dengan perubahan harga harian. Variabel `Change%` juga memiliki korelasi terbalik, ini terjadi karena volatilitas di harga yang rendah dan lebih stabil di harga yang tinggi.

* ***Visualisasi Pairplot***

    ![66702bb8-ce59-4808-b882-ac85f271995a](https://github.com/user-attachments/assets/aadcb1b4-eb0d-464a-955a-fdc8c49a8be8)

    Selanjutnya ktia melakukan visualisasi menggunakan Pairplot yang memberikan pandangan lebih lanjut dengan menunjukkan distribusi setiap variabel di sepanjang diagonal dan hubungan antar variabel dalam bentuk scatter plot di luar diagonal.

    Pada scatter plot antar variabel harga terlihat bahwa hubugnan di antara variabel-variabel ini sangat linear dan berkorelasi kuat. Misalkan jika `Open` naik, variabel harga lainnya juga cenderung naik. Hal ini umum dalam data keuangan dimana harga pembuka, tertinggi, terendah, dan penutupan biasanya berfluktuasi bersama-sama dalam satu hari.

    Distribusi `Change %` terlihat mendekati distribusi normal dengan puncak di sekitar nol. Ini berarti perubahan harga harian sebagian besar berada di sekitar titik nol, yang menunjukkan volatilitas harian yang relatif rendah dengan beberapa outlier di kedua sisi (positif dan negatif). Sedangkan distribusi `Vol.` menunjukkan beberapa puncak, yang mungkin menunjukkan lonjakan volume pada waktu-waktu tertentu.

***Time Series Analysis***

Pada langkah ini data tanggal dikonversi ke format datetime dan diurutkan secara kronologis, ini penting agar data dapat divisualisasikan dalam urutan yang benar. Grafik harga Solana sepanjang waktu dari dataset yang dimiliki menunjukkan kenaikan signifikan ke puncak harga tertinggi yang diikuti oleh penurunan tajam dan fluktuasi di sekitar harga yang lebih rendah.

Puncak harga tinggi ini bisa mengindikasikan fase "bull run" atau masa ketertarikan pasar yang tinggi, sementara penurunan tajam setelahnya mungkin terkait dengan koreksi pasar atau penjualan besar-besaran oleh investor. Fluktiasi setelah penurunan ini menunjukkan adanya stabilisasi di level harga baru, yang mungkin merefleksikan level support yang dipertahankan oleh pasar.

![7174a686-244d-43d4-ac67-e0ed57fdbc82](https://github.com/user-attachments/assets/c50c62c6-8297-47bc-94e8-b6b027947324)

* ***Visualisasi Menggunakan Candlestick***

    ![image](https://github.com/user-attachments/assets/7864e237-d7c0-4f4e-b8fd-430c19451f2d)

    Selanjutnya kita melakukan visualisasi menggunakan candlestick, digunakan untuk memberikan infrmasi harga pembuka, tertinggi, terendah, dan penutup setiap hari. Visualisasi ini sangat membantu dalam melihat pola-pola harian yang bisa menunjukkan apakah pasar berada dalam tren naik atau turun.

    Kita melakukan ini karena pola candlestick ini memungkinkan kita untuk mengidentifikasi titik pembalikan, momentum harga, atau pola konsolidasi.

* ***Visualisasi  Trend, Seaonal, dan Resid***

    ![image](https://github.com/user-attachments/assets/923c01a1-3d43-41ae-9c0a-2806323d5037)

    Dekomposisi komponen time series, data harga dipecah menjadi `Trend`, `Seasonal`, dan `Residual`. Komponen `Trend` menunjukkan pola kangka panjang dari data harga, `Seasonal` menyoroti pola musiman atau berulang dalam periode tertentu, dan `Residual` mengungkapkan variasi yang tidak dijelaskan oleh dua komponen sebelumnya.

    Trend terlihat jelas puncak tren di sekitar aktu kenaikan harga maksumum, diikuti oleh tren menurun dan kemudian stabilisasi di level yang lebih rendah, menunjukan pergerakan umum harga Solana dalam jangka panjang.

    Pada pola `Seasonal` menandakan bahwa ada pola yang teratur dan konsistem dalam harga yang mungkin disebabkan oleh aktivitas pasar yang membentuk siklus. Sedangkan `Residual` adalah fluktuasi yang tidak dapat dijelaskan oleh tren atau pola musiman, yang sering kali mencerminkan kejadian yang tidak terduga atau volatilitas mendadak di pasar.

* ***Visualisasi ACF dan PACF***

    ![image](https://github.com/user-attachments/assets/7d55726c-976a-4d3d-91ab-d97c47d53a62)
    
    ![image](https://github.com/user-attachments/assets/9300f98d-e171-4593-a44f-b745f0601a19)

    Selanjutnya menggunakan ACF dan PACF yang memberikan informasi tentang korelasi harga saat ini dengan harga pada periode sebelumnya (lag). ACF menunjukkan bahwa ada kubungankuat antar harga untuk beberapa lag pertama, sementara PACF memperlihatkan bahwa korelasi signifikan turama pada lag awal.

    Pada ACF menunjukkan orelasi yang tinggi pada lag awal menunjukkan bahwa harga saat ini masih sangat dipengaruhi oleh harga-harga sebelumnya. Ini menandakan adanya keterkaitan antar waktu yang kuat dalam data, yang bisa dimanfaatkan untuk prediksi berbasis model time series. Dan nilai PACF yang signifikan pada lag awal menunjukkan bahwa harga terpengaruh langsung oleh harga pada satu atau dua periode sebelumnya, penting untuk mempertimbangkan lag ini dalam model prediksi. 

* ***Visualisasi Volatilitas***

    ![image](https://github.com/user-attachments/assets/37d4b48d-4839-4832-8b5f-2297a6bea106)

    Volatilitas dihitung sebagai standar deviasi dari return harian dalam rolling window 7 hari. Grafk ini menyoroti fluktuasi volatilitas Solana dalam dalam jangka pendek. Puncak volatilitas terlihat di beberapa periode tertentu, terutama di sekitar waktu puncak harga dan penurunan tajam berikutnya. Ini menandakan masa ketidakpastian dan pergerakan besar di pasar, sering kali terkait dengan peristiwa eksternal atau perubahan sentimen investor.

* ***Visualisasi Volume Perdagangan Solana***

    ![image](https://github.com/user-attachments/assets/8960489d-054c-47c0-8a45-2ce1a09eb92f)

    Disini kita melakukan visualisasi volume perdagangan Solana menunjukkan jumlah perdagangan yang terjadi setiap hari. Ada beberapa lonjakan signifikan dalam volume yang tampaknya bertepatan dengan puncak harga atau periode volatilitas tinggi.

***Technical Indicators***

Pada tahap ini, kita menambahkan moving averages (MA), exponential moving average (EMA), dan moving average convergence divergence (MACD).

* ***Visualisasi Harga Solana dengan Moving Averages***

    ![image](https://github.com/user-attachments/assets/3707499d-ee9d-4ef7-8934-bcd204b03bf1)

    Grafik menunjukkan garis harga Solana bersamaan dengan MA7 dan MA21. Perpotongan antara MA7 dan MA21 memberikan sinyal perubahan tren. Ketika MA7 melintasi MA21 dari bawah ke atas, itu bisa menjadi sinyal tren naik (bullish crossover), menunjukkan potensi pembelian. Sebaliknya, ketika MA7 melintasi MA21 dari atas ke bawah, ini adalah sinyal tren turun (bearish crossover), menandakan potensi penjualan.

Selanjutnya kita menghitung Exponential Moving Avarage dan Moving Average Convergence Divergence. EMA12 dan EMA26 digunakan dalam perhitungan MACD. EMA memberikan bobot lebih besar pada harga terbaru, sehingga lebih responsif terhadap perubahan harga. MACD dihitung sebagai selisih antara EMA12 dan EMA26, sedangkan garis sinyal (Signal Line) merupakan EMA dari MACD itu sendiri. 

* ***Visualisasi MACD dan Signal Line***

    ![image](https://github.com/user-attachments/assets/23effe1c-0b23-4e71-a907-f1ecce9def29)

    Grafik menunjukkan pergerakan MACD dan Signal Line. Ketika MACD melintasi Signal Line dari bawah ke atas, ini menandakan sinyal beli (bullish), sedangkan ketika MACD melintasi Signal Line dari atas ke bawah, itu adalah sinyal jual (bearish). MACD juga dapat menunjukkan momentum pasar; semakin jauh MACD dari garis nol, semakin kuat momentum tersebut.

* ***Visualisasi Boxplot***

    ![image](https://github.com/user-attachments/assets/df39753d-88f0-4a65-a210-f81f97d3821e)

    Visualisasi Boxplot menunjukkan distribusi harga Solana setiap bulan sepanjang tahun, dengan indikator nilai minimum, maksimum, median, serta adanya outlier.

    Bulan-bulan seperti November menunjukkan rentang harga yang lebih luas dan median yang lebih tinggi, mengindikasikan adanya volatilitas yang tinggi. Sementara bulan lain, seperti Juli dan Juni, menunjukkan harga yang lebih stabil dengan rentang yang lebih sempit. Outlier yang muncul pada beberapa bulan menandakan adanya fluktuasi harga yang ekstrim dalam waktu singkat, sering kali disebabkan oleh peristiwa eksternal atau peningkatan volume perdagangan mendadak.


## ***Data Preparation***

Tahap ini melibatkan pembersihan dan transformasi data agar siap digunakan dalam model

* ***Handling Missing Value*** : 

    Jumlah missing values pada 'Vol.': 0

    Langkah pertama dalam menangani missing values pada kolom Vol. adalah dengan menghitung median dari kolom tersebut. Median dipilih karena lebih stabil terhadap outlier dibandingkan rata-rata (mean), sehingga memberikan estimasi yang lebih andal dalam data perdagangan yang bisa sangat fluktuatif.

    Setelah menghitung median, kita menggunakan nilai tersebut untuk menggantikan missing values pada kolom Vol.. Proses ini memastikan bahwa data lengkap dan tidak ada nilai yang hilang, sehingga analisis selanjutnya tidak terganggu.

* ***Feature Extraction*** :

* ***Konversi kolom Date***

    Untuk melakukan analisis time series, kolom Date dikonversi ke dalam format datetime. Ini memungkinkan kita untuk memanfaatkan fungsi-fungsi analisis waktu pada tahap selanjutnya.

* ***Pengukutan Data***

    Setelah konversi, data diurutkan berdasarkan tanggal untuk memastikan urutan kronologis yang benar. Mengonversi dan mengurutkan data berdasarkan tanggal memberikan alur kronologis yang jelas, yang sangat penting dalam analisis time series. Ini memungkinkan kita untuk menganalisis data harga dari waktu ke waktu dengan lebih akurat.

* ***Menambahkan Moving Avarages, Daily Return, dan Date as Ordinal***

    Selanjutnya menambahkan Moving Averages, Daily Return, dan Date as Ordinal. Moving average ini berguna untuk menghaluskan fluktuasi jangka pendek dan membantu dalam melihat tren jangka panjang dan menengah dalam pergerakan harga. Daily return dihitung sebagai persentase perubahan harga harian. Ini adalah metrik penting yang menunjukkan volatilitas harian dan potensi keuntungan atau kerugian dalam jangka pendek. Untuk keperluan pemodelan regresi dan LSTM, kolom Date dikonversi ke bentuk ordinal, yang menyajikan tanggal dalam format numerik. Hal ini diperlukan karena model tidak dapat bekerja langsung dengan data dalam format datetime.

* ***Data Splitting dan Normalisasi***

    Pada tahap ini kita melakukan split data dan melakukan normalisasi data pada kolom Price. Data dipecah menjadi bagian pelatihan dan pengujian dengan proporsi 80:20, selanjutnya melakukan normalisasi data menggunakan MinMaxScaler dengan rentang 0 hingga 1 untuk digunakan pada model LSTM. Selalah melakukan split data untuk model LSTM, kita melakukan split data untuk model regresi linear.

## ***Modeling***

Tahap ini melibatkan pembangunan dan pelatihan model untuk memprediksi harga Solana

* ***Regresi Linear:***

    Pada tahap ini model regresi linear dilatih menggunakan data pelatihan dan mencoba untuk menemukan hubungan linear antara fitur(tanggal ordinal) dan target variabel(harga).

    Regresi Linear digunakan sebagai model baseline karena kesederhanaannya dan kemudahan dalam interpretasi. Meskipun model ini tidak diharapkan memberikan hasil terbaik pada data yang sangat volatil seperti harga cryptocurrency, keberadaannya penting untuk membandingkan efektivitas model yang lebih kompleks.

* ***LSTM:***

    Setelah membuat model regresi linear, kita akan membuat model LSTM, alasan dipilihnya karena kemampuannya dalam menangani data time series dan mengenali pola temporal jangka panjang. LSTM mampu menangkap hubungan non-linear dalam data yang tidak bisa ditangkap oleh model linear sederhana, menjadikannya lebih cocok untuk memprediksi harga yang sangat volatil seperti Solana.

    Selanjutnya kita menentukan Window size, disini kita ditetapkan menjadi 60, yang artinya model akan menggunakan 60 data sebelumnya untuk memprediksi harga berikutnya. Pemilihan window size yang tepat sangat penting karena memengaruhi konteks data yang dilihat oleh model.

    Fungsi `create_dataset` dibuat untuk membentuk dataset yang akan diproses oleh LSTM. Dataset ini terdiri dari sekuens data dengan panjang window size sebagai input dan harga aktual berikutnya sebagai output. Fungsi ini memungkinkan data time series diubah menjadi format yang dapat dimengerti oleh LSTM, di mana setiap titik prediksi bergantung pada urutan data sebelumnya.

    Setelah membuat fungsi `create_dataset`, selanjutnya kita menyiapkan data test dan data train yang dibentuk dengan memeprtimbangkan window size, untu memastikan setiap sekuens memiliki panjang yang sama untuk diproses oleh model LSTM.

    Setelah proses pengubahhan data, kita lanjutkan untuk membuat model LSTM yang disertai lapisan dropout untuk mencegah overfitting. Model kemudian dilatih menggunakan optimizer `adam` dan fungsi los `mean_squared_error` selama 50 epoch.
    * Layer pertama LSTM dengan 256 unit dan `return_sequences=True` untuk output sequential
    * Dropout Layer dengan tingkat 20% untuk mencegah overfitting
    * Layer kedua LSTM dengan 128 unit
    * Dense layer dengan 1 unit untuk prediksi harga
    * Epochs: 50
    * Batch Size: 32
    * Optimizer: Adam
    * Loss Function: Mean Squared Error (MSE)

    kita mencoba berbagai konfigurasi jumlah neuron dan lapisan untuk menemukan arsitektur yang memberikan hasil terbaik.

    Untuk mencegah overfitting, dropout rate sebesar 20% ditambahkan setelah setiap lapisan LSTM.



* ***Kelebihan dan Kekurangan Model:***
    * ***Regresi Linear:*** Memiliki kelebihan yaitu simpel dan mudah diinterpretasikan, sedangkan kekurangannya tidak mampu menangkap pola non-linear yang komplkes dalam data cryptocurrency yang volatil.
    * ***LSTM:*** Memiliki kelebihan yaitu mampu menangkap hubungan temporal dan pola non-linear dalam data time series, sedangkan kekurangannya membutuhkan waktu pelatihan yang lama dan lebih kompleks dalam tuning hyperparameter (bergantung dataset)

## ***Evaluasi***

Setelah melatih model, tentunya kita harus melakukan evaluasi pada model yang telah kita buat. Kita mulai dengan mengevaluasi regresi linear menggunakan dua metrik yaitu `Mean Absolute Error (MAE)` dan `Root Mean Squared Error (RMSE)`. MAE menghitung rata-rata dari kesalahan absolut antara nilai prediksi dan nilai aktual, yang memberi kita ukuran rata-rata kesalahan tanpa mempertimbangkan arah kesalahan. Di sisi lain, RMSE memberikan gambaran lebih sensitif terhadap kesalahan yang lebih besar, karena setiap selisih dipangkatkan dua, memperbesar dampak dari outlier.

Evaluasi model dilakukan dengan menggunakan metrik MAE, yatu mengukur rata-rata absolut dari selisih antara nilai aktual dan prediksi, memberikan indikasi seberapa besar kesalahan prediksi secara umum. RMSE mengukur akar kuadrat dari rata-rata kesalahan kuadrat, lebih sensitif terhadap outlier dibandingkan MAE.

* ***Regresi Linear:***

    Output cell: 
    Linear Regression MAE :
    96.33515759814426
    Linear Regression RMSE :
    100.02367955459133

    Hasil evaluasi dari model regresi linear yaitu MAE sebesar 96.335 mengindikasikan bahwa rata-rata prediksi model menyimpang sekitar 96.335 unit dari nilai aktual, dan RMSE sebesar 100.02 menandakan ada beberapa kesalahan prediksi yang cukup besar, menunjukkan model regresi linear mungkin tidak ideal untuk data yang sangat volatil seperti harga crypto.

* ***LSTM:***

    Selanjutnya, kita beralih ke model LSTM dan menggunakan metrik evaluasi yang sama untuk melihat performanya. Pada model LSTM, data prediksi dan nilai aktual harus di-inverse-transform karena data telah dinormalisasi sebelumnya. Setelah inversi, kita dapat menghitung MAE dan RMSE pada skala asli untuk membandingkan kinerja LSTM dengan Linear Regression secara langsung.

    Hasil dari evaluasi model LSTM yaitu MAE  sebesar 4.987 menunjukkan bahwa model LSTM memiliki rata-rata kesalahan yang jauh lebih kecil dibandingkan Linear Regression, yang berarti prediksinya lebih mendekati harga aktual. Pada RMSE sebesar 6.456 memperlihatkan bahwa LSTM mampu mengatasi fluktuasi data kripto dengan lebih baik, meskipun masih terdapat beberapa outlier atau fluktuasi yang sulit diprediksi. 

    Output cell:
    LSTM MAE: 
    4.987166164519681
    LSTM RMSE: 
    6.456942636757551

***Analisis Hasil Evaluasi***

LSTM memiliki MAE dan RMSE yang jauh lebih rendah dibandingkan Regresi Linear. Ini menunjukkan bahwa LSTM lebih akurat dalam memprediksi harga Solana. Untuk sensitivias terhadap volatilitas, regresi Linear gagal menangkap pola non-linear dan volatilitas tinggi dalam data, menghasilkan kesalahan prediksi yang besar, sedangkan LSTM berhasil menangkap pola temporal dan fluktuasi harga, menghasilkan prediksi yang lebih akurat.

***Visualisasi Hasil Prediksi***
Selanjutnya kita coba visualisasikan grafik loss training dan validation selama proses pelatihan LSTM, ini memberikan wawasan tentang performa model seiring waktu. Di awal pelatihan, nilai loss sangat tinggi, namun seiring bertambahnya epoch, nilai loss menurun dan mencapai stabilitas. Ini menunjukkan bahwa model berhasil belajar dari data tanpa terlalu banyak overfitting pada data training, yang dapat diamati dari jarak yang relatif konsisten antara training dan validation loss.

![image](https://github.com/user-attachments/assets/de62d6a4-7b1d-432b-93dc-b16822edfcd6)

Pada langkah ini, model LSTM yang telah kita latih digunakan untuk membuat prediksi pada data uji (`X_test_lstm`). Karena data input telah dinormalisasi selama preprocessing, hasil prediksi juga berada dalam skala yang sama. Oleh karena itu, dilakukan inversi skala menggunakan scaler.inverse_transform agar hasil prediksi kembali ke skala asli, sehingga dapat dibandingkan langsung dengan data harga aktual.

Tahap akhir adalah memvisualisasikan hasil repdiksi dan membandingkannya dengan harga aktual dalam rentang waktu yang sama, pada grafiik ini garis biru menunjukkan harga aktual dari Solana, sedangkan garis merah menunjukkan harga prediksi yang dihasilkan oleh model LSTM.

Grafik ini menunjukkan bahwa prediksi dari model LSTM mengikuti tren harga aktual dengan cukup baik, meskipun ada beberapa fluktuasi yang tidak dapat sepenuhnya diprediksi oleh model. Dekatnya pergerakan antara garis merah dan biru mengindikasikan bahwa model LSTM mampu menangkap pola harga Solana dalam rentang waktu yang diuji.

![image](https://github.com/user-attachments/assets/17af4086-0413-44dc-8219-417d31d8fb59)

***Menghubungkan Hasil dengan Bussiness Understanding***

Model LSTM berhasil membangun model prediksi yang akurat untuk harga harian Solana, menjawab pertanyaan tentang bagaimana membangun model machine learning yang efektif, tujuan untuk mengidentifikasi algoritma yang efektif tercapai dengan LSTM yang menunjukkan performa terbaik dibandingkan dengan model baseline.

***Kesimpulan***

Dalam proyek ini, kami telah berhasil membangun dan mengevaluasi dua model prediksi harga Solana menggunakan Regresi Linear dan Long Short-Term Memory (LSTM). Hasil evaluasi menunjukkan bahwa model LSTM memiliki performa yang jauh lebih baik dengan MAE sebesar 4.987 dan RMSE sebesar 6.456, dibandingkan dengan Regresi Linear yang memiliki MAE sebesar 96.335 dan RMSE sebesar 100.02.



