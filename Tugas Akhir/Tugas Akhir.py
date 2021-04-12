import pandas
from sklearn import linear_model

print("Kami ingin mengetahui faktor-faktor yang mempengaruhi jumlah tahanan dan napi "
      + "di Indonesia. Kami mencoba menggunakan dua faktor, yaitu pertumbuhan ekonomi "
      + "di Indonesia dan jumlah impor barang modal di Indonesia. Di sini, kami mencari "
      + "hubungan antara pertumbuhan ekonomi di Indonesia dan jumlah impor barang modal "
      + "di Indonesia, terhadap jumlah tahanan dan napi di Indonesia. Kami menggunakan "
      + "regresi linier berganda untuk prediksi kali ini")

dataFrame = pandas.read_csv("Data Tugas Akhir.csv")
print("\n============================================================================"
      + " Data Set X1, X2, dan Y ===================================================="
      + "===============\n")
print("Keterangan : \n- X1 = Pertumbuhan ekonomi di Indonesia. \n- X2 = Jumlah impor "
      + "barang modal di Indonesia. \n- Y = Jumlah tahanan dan napi di Indonesia.\n")
print(dataFrame)

x = dataFrame[['X1', 'X2']]
y = dataFrame['Y']

regresi = linear_model.LinearRegression()
regresi.fit(x, y)

dataYangDiprediksi = pandas.read_csv("Data yang akan Diprediksi.csv")
print("\n============================================================================"
      + " Data Yang Akan Diprediksi ================================================="
      + "===============\n")
print(dataYangDiprediksi)
print("\n============================================================================"
      + "============================================================================"
      + "===============\n")

# Mencari korelasi pada setiap variabel
print("\n============================================================================"
      + " Korelasi =================================================================="
      + "===============\n")
print("Korelasi dapat mengukur sejauh mana dua variabel saling terkait. Korelasi "
      + "digunakan untuk mengukur kekuatan dan arah dari hubungan linier antara dua "
      + "variabel. Nilai dari korelasi ada di antara -1 hingga 1. Jika mendekati -1, "
      + "maka korelasi negatifnya kuat. Jika mendekati 1, maka korelasi positifnya "
      + "kuat.\n")
print(dataFrame[['X1', 'X2', 'Y']].corr())
#

# Memprediksi jumlah tahanan dan napi pada tahun 2018 dengan pertumbuhan ekonomi
# sebesar 5,06% dan jumlah impor barang modal sebesar 4004,4
x1 = {}
x2 = {}
prediksiJumlahTahananNapi = {}
x1[0] = 5.06
x2[0] = 4004.4
prediksiJumlahTahananNapi[0] = regresi.predict([[x1[0], x2[0]]])
print("\n============================================================================="
      + "= Hasil Prediksi =========================================================="
      + "===============")
print("============================================================================="
      + "= Tahun 2018 =============================================================="
      + "===============\n")
print("Untuk tahun 2018, kami beri X1-nya adalah " + str(x1[0]) + " dan X2-nya adalah "
      + str(x2[0]) + ".")
print("Jika pertumbuhan ekonomi (X1) sebesar " + str(x1[0]) + " dan jumlah impor barang "
      + "modal (X2) adalah " + str(x2[0]) + ", maka kemungkinan jumlah tahanan dan napi "
      + "(Y) adalah " + str(prediksiJumlahTahananNapi[0][0]))
print("\n============================================================================"
      + "============================================================================"
      + "===============\n")
#

# Mencari koefisien dari pertumbuhan ekonomi (X1) dan jumlah impor barang modal (X2)
koefisienX1 = regresi.coef_[0]
koefisienX2 = regresi.coef_[1]
print("============================================================================="
      + "= Koefisien ==============================================================="
      + "===============\n")
print("Koefisien adalah faktor yang mendeskripsikan hubungan dengan unknown variable. "
      + "Dalam kasus kami, yang dicari adalah koefisien dari X1 terhadap Y dan "
      + "koefisien X2 terhadap Y. Hasil koefisien menunjukkan kami apa yang terjadi, "
      + "jika kami menaikkan atau menurunkan salah satu nilai independen yaitu X1 atau "
      + "X2.\n")
print("Koefisien milik X1 : " + str(koefisienX1))
print("Koefisien milik X2 : " + str(koefisienX2))
print("\nDengan ini, maka setiap nilai pertumbuhan ekonomi naik 1 (persen), "
      + "jumlah tahanan dan napi turun sebesar " + str((koefisienX1 * (-1))) + ". Dan "
      + "jika jumlah impor barang modal naik sebesar 1, maka jumlah tahanan "
      + "dan napi juga naik sebesar " + str(koefisienX2))
print("\n============================================================================"
      + "============================================================================"
      + "===============\n")
#

# Memprediksi jumlah tahanan dan napi pada tahun 2019, dengan pertumbuhan ekonomi
# dinaikkan menjadi 6,12% dan jumlah impor barang modal sama seperti tahun lalu.
x1[1] = 6.12
x2[1] = 4004.4
prediksiJumlahTahananNapi[1] = regresi.predict([[x1[1], x2[1]]])
print("============================================================================="
      + "= Tahun 2019 =============================================================="
      + "===============\n")
print("Sekarang, kita coba menaikkan nilai pertumbuhan ekonomi untuk tahun "
      + "berikutnya, yaitu tahun 2019.")
print("Jika pertumbuhan ekonomi(X1) sebesar " + str(x1[1]) + " dan jumlah impor barang "
      + "modal (X2) adalah " + str(x2[1]) + ", maka kemungkinan jumlah tahanan dan napi "
      + "(Y) adalah " + str(prediksiJumlahTahananNapi[1][0]))
print("\nPrediksi jumlah tahanan dan napi tahun sebelumnya adalah "
      + str(prediksiJumlahTahananNapi[0][0]) + ". Kita cek apakah koefisiennya sudah "
      + "benar.\n")

konstanta = regresi.intercept_
hasilCekKoefisien = konstanta + (x1[1] * koefisienX1) + (x2[1] * koefisienX2)
print("Rumus hasil pengecekan = Konstanta + (X1 * koefisienX1) + (X2 * koefisienX2)")
print("Hasil pengecekan = " + str(konstanta) + " + ("
      + str(x1[1]) + " * " + str(koefisienX1) + ") + (" + str(x2[1]) + " * "
      + str(koefisienX2) + ") = " +str(hasilCekKoefisien))
print("\nDi sini dapat terlihat, bahwa koefisien X1 yang bernilai " + str(koefisienX1)
      + " sudah benar. Itu karena, di antara hasil pengecekan dengan prediksi jumlah "
      + "tahanan dan napi, memiliki hasil yang sama.")
print("\n============================================================================"
      + "============================================================================"
      + "===============\n")
#

# Memprediksi jumlah tahanan dan napi pada tahun 2020, dengan jumlah impor barang 
# modal dinaikkan menjadi 4791,4 dan pertumbuhan ekonominya naik menjadi 6,25%.
x1[2] = 6.25
x2[2] = 4791.4
prediksiJumlahTahananNapi[2] = regresi.predict([[x1[2], x2[2]]])
print("============================================================================="
      + "= Tahun 2020 =============================================================="
      + "===============\n")
print("Sekarang, kita coba menaikkan kedua nilai X sekaligus untuk tahun "
      + "berikutnya, yaitu tahun 2020.")
print("Jika pertumbuhan ekonomi (X1) sebesar " + str(x1[2]) + " dan jumlah impor barang "
      + "modal (X2) adalah " + str(x2[2]) + ", maka kemungkinan jumlah tahanan dan napi "
      + "(Y) adalah " + str(prediksiJumlahTahananNapi[2][0]))
print("\nPrediksi jumlah tahanan dan napi tahun sebelumnya adalah "
      + str(prediksiJumlahTahananNapi[1][0]) + ". Kita cek apakah koefisiennya sudah "
      + "benar.\n")

konstanta = regresi.intercept_
hasilCekKoefisien = konstanta + (x1[2] * koefisienX1) + (x2[2] * koefisienX2)
print("Rumus hasil pengecekan = Konstanta + (X1 * koefisienX1) + (X2 * koefisienX2)")
print("Hasil pengecekan = " + str(konstanta) + " + ("
      + str(x1[2]) + " * " + str(koefisienX1) + ") + (" + str(x2[2]) + " * "
      + str(koefisienX2) + ") = " +str(hasilCekKoefisien))
print("Di sini dapat terlihat, bahwa koefisien X2 yang bernilai " + str(koefisienX2)
      + " sudah benar. Itu karena, di antara hasil pengecekan dengan prediksi jumlah "
      + "tahanan dan napi, memiliki hasil yang sama.")
print("\n============================================================================"
      + "============================================================================"
      + "===============\n")
#

# Menampilkan selisih antara data aktual dengan data hasil prediksi
print("=====================================================================Selisih "
      + "Hasil Prediksi dengan Data Aktual ========================================="
      + "===============\n")
dataHasilPrediksi = {'Tahun' : ['2018', '2019', '2020'],
                     'Y' : [prediksiJumlahTahananNapi[0][0],
                            prediksiJumlahTahananNapi[1][0],
                            prediksiJumlahTahananNapi[2][0]]}
dataFrameHasilPrediksi = pandas.DataFrame(dataHasilPrediksi)
print(dataFrameHasilPrediksi)
print("\nData aktual Y tahun 2018 : 248,659\n")

selisihY = abs(248659 - prediksiJumlahTahananNapi[0])
print("Rumus selisih = ABS(Y aktual - Y hasil prediksi)") 
print("Selisih Y dari hasil prediksi dengan data aktual tahun 2018 : ABS(248659 - "
      + str(prediksiJumlahTahananNapi[0][0]) + ") = " + str(selisihY[0]) + ".")
print("\nJadi, selisih Y aktual dengan Y hasil prediksi tahun 2018 adalah " 
      + str(selisihY[0]) + "\nDapat terlihat bahwa prediksi hampir mendekati data "
      + "aslinya. \nSelisih hanya kami tampilkan pada tahun 2018 saja, karena data "
      + "aktual X1 dan X2 tidak tersedia pada websitenya.")
print("\n============================================================================"
      + "============================================================================"
      + "===============\n")

