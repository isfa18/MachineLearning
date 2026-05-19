from flask import Flask, request, jsonify
import numpy as np
import mysql.connector

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================
# FLASK
# =====================
app = Flask(__name__)

# =====================
# DB CONNECTION
# =====================
conn = mysql.connector.connect(
    host="195.88.211.226",
    user="langgen1_lj_db",
    password="~ao-S%9UGMrU,^bP",
    database="langgen1_lj_db"
)

# =====================
# BERSIHKAN BERAT
# =====================
def bersihkan_berat(nilai):
    try:
        nilai = str(nilai)
        nilai = nilai.replace("Kg", "")
        nilai = nilai.replace("KG", "")
        nilai = nilai.replace("kg", "")
        nilai = nilai.replace(",", ".")
        nilai = nilai.strip()
        return float(nilai)
    except:
        return None

# =====================
# AMBIL DATA DB
# =====================
def ambil_data_db(barang):
    cursor = conn.cursor()

    query = """
    SELECT bk.berat 
    FROM barang_keluar bk
    JOIN kategori_barang kb ON bk.id_kategoriBarang = kb.id
    WHERE LOWER(kb.nama_kategori) LIKE %s
    """

    cursor.execute(query, ('%' + barang.lower() + '%',))
    rows = cursor.fetchall()

    data = []

    for r in rows:
        berat = bersihkan_berat(r[0])
        if berat is not None:
            data.append(berat)

    cursor.close()
    return data

# =====================
# AMBIL DATA FULL
# =====================
def ambil_data(barang):
    return ambil_data_db(barang)

# =====================
# BUILD DATASET MACHINE LEARNING
# =====================
def build_training_data(barang_list):

    X = []
    y = []

    for barang in barang_list:

        data = ambil_data(barang)

        print("====================")
        print("Barang:", barang)
        print("Data:", data)
        print("Jumlah data:", len(data))

        if len(data) < 2:
            print("Data kurang dari 2, dilewati")
            continue

        avg = np.mean(data)
        mx = np.max(data)
        mn = np.min(data)
        std = np.std(data)

        # Membuat variasi stok simulasi untuk training
        # Tidak random, supaya hasil stabil
        stok_simulasi_list = np.linspace(
            avg * 0.1,
            avg * 5.0,
            100
        )

        for stok_simulasi in stok_simulasi_list:

            # Label:
            # 1 = JUAL
            # 0 = TUNGGU
            label = 1 if stok_simulasi >= avg else 0

            X.append([
                avg,
                mx,
                mn,
                std,
                stok_simulasi
            ])

            y.append(label)

    return np.array(X), np.array(y)

# =====================
# TRAIN MODEL
# =====================
barang_list = ["besi", "dus", "atum"]

X, y = build_training_data(barang_list)

model = None
akurasi_model = None

if len(X) > 0 and len(np.unique(y)) > 1:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    akurasi_model = accuracy_score(y_test, y_pred)

    print("====================")
    print("Model berhasil dilatih")
    print("Algoritma: Decision Tree Classifier")
    print("Jumlah data training:", len(X))
    print("Class:", np.unique(y, return_counts=True))
    print("Akurasi:", akurasi_model)
    print("====================")

else:
    print("====================")
    print("Model tidak dilatih")
    print("Penyebab: data tidak cukup atau label hanya 1 class")
    print("====================")

# =====================
# PREDIKSI
# =====================
def prediksi_jual(barang, stok):

    data = ambil_data(barang)

    if len(data) < 2:
        return {
            "rekomendasi": "TUNGGU",
            "keterangan": "Data pengeluaran barang kurang dari 2"
        }

    avg = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    std = np.std(data)

    features = np.array([[
        avg,
        mx,
        mn,
        std,
        stok
    ]])

    if model is not None:
        pred = model.predict(features)[0]
    else:
        pred = 0

    # Validasi akhir sesuai logika bisnis
    # Kalau stok >= rata-rata pengeluaran, maka JUAL
    if stok >= avg:
        hasil = "JUAL"
    else:
        hasil = "TUNGGU"

    return {
        "rekomendasi": hasil,
        "prediksi_model": "JUAL" if pred == 1 else "TUNGGU",
        "stok": float(stok),
        "rata_rata_pengeluaran": round(float(avg), 2),
        "pengeluaran_maksimum": round(float(mx), 2),
        "pengeluaran_minimum": round(float(mn), 2),
        "standar_deviasi": round(float(std), 2),
        "akurasi_model": round(float(akurasi_model), 4) if akurasi_model is not None else None,
        "algoritma": "Decision Tree Classifier",
        "keterangan": "JUAL jika stok lebih besar atau sama dengan rata-rata pengeluaran"
    }

# =====================
# API PREDIKSI
# =====================
@app.route('/prediksi', methods=['POST'])
def prediksi():

    try:
        req = request.json

        barang = req.get("barang")
        stok = float(req.get("stok"))

        hasil = prediksi_jual(barang, stok)

        return jsonify({
            "barang": barang,
            "stok": stok,
            "rekomendasi": hasil["rekomendasi"],
            "detail": hasil
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "rekomendasi": "TUNGGU"
        })

# =====================
# API CEK DATA
# =====================
@app.route('/cek-data', methods=['GET'])
def cek_data():

    try:
        barang = request.args.get("barang")

        if not barang:
            return jsonify({
                "error": "Parameter barang wajib diisi. Contoh: /cek-data?barang=besi"
            })

        data = ambil_data(barang)

        if len(data) == 0:
            return jsonify({
                "barang": barang,
                "jumlah_data": 0,
                "data": [],
                "keterangan": "Data tidak ditemukan"
            })

        avg = np.mean(data)

        return jsonify({
            "barang": barang,
            "jumlah_data": len(data),
            "data": data,
            "rata_rata_pengeluaran": round(float(avg), 2),
            "pengeluaran_maksimum": round(float(np.max(data)), 2),
            "pengeluaran_minimum": round(float(np.min(data)), 2),
            "standar_deviasi": round(float(np.std(data)), 2),
            "rekomendasi_jika_stok_sama_atau_diatas": round(float(avg), 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

# =====================
# API HOME
# =====================
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "API Prediksi Penjualan Barang Berjalan",
        "algoritma": "Decision Tree Classifier",
        "status_model": "aktif" if model is not None else "tidak aktif",
        "akurasi_model": round(float(akurasi_model), 4) if akurasi_model is not None else None,
        "endpoint_prediksi": "/prediksi",
        "endpoint_cek_data": "/cek-data?barang=besi"
    })

# =====================
# RUN
# =====================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
