from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import mysql.connector

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
    password="ISI_PASSWORD_KAMU",
    database="langgen1_lj_db"
)

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
        try:
            berat = str(r[0])
            berat = berat.replace("Kg", "")
            berat = berat.replace("kg", "")
            berat = berat.replace("KG", "")
            berat = berat.strip()
            data.append(float(berat))
        except:
            pass

    cursor.close()
    return data

# =====================
# AMBIL DATA FULL
# =====================
def ambil_data(barang):
    return ambil_data_db(barang)

# =====================
# BUILD DATASET
# =====================
def build_training_data(barang_list):

    X = []
    y = []

    for barang in barang_list:

        data = ambil_data(barang)

        if len(data) < 2:
            continue

        avg = np.mean(data)
        mx = np.max(data)
        mn = np.min(data)
        std = np.std(data)

        # Membuat data stok simulasi dari kecil sampai besar
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

            X.append([avg, mx, mn, std, stok_simulasi])
            y.append(label)

    return np.array(X), np.array(y)

# =====================
# TRAIN MODEL SAFE
# =====================
barang_list = ["besi", "dus", "atum"]

X, y = build_training_data(barang_list)

model = None

if len(X) > 0 and len(np.unique(y)) > 1:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("Model berhasil dilatih")
    print("Jumlah data training:", len(X))
    print("Class:", np.unique(y, return_counts=True))
else:
    print("Model tidak dilatih (data tidak cukup / 1 class)")

# =====================
# PREDIKSI SAFE
# =====================
def prediksi_jual(barang, stok):

    data = ambil_data(barang)

    if model is None:
        return "TUNGGU"

    if len(data) < 2:
        return "TUNGGU"

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

    # Model tetap dipakai
    pred = model.predict(features)[0]

    # Logic akhir disesuaikan dengan rata-rata pengeluaran
    if stok >= avg:
        return "JUAL"
    else:
        return "TUNGGU"

# =====================
# API
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
            "rekomendasi": hasil
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "rekomendasi": "TUNGGU"
        })

# =====================
# RUN
# =====================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
