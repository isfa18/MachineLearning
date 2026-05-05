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
    password="~ao-S%9UGMrU,^bP",
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
            data.append(float(str(r[0]).replace("Kg","").strip()))
        except:
            pass

    return data

# =====================
# AMBIL DATA FULL
# =====================
def ambil_data(barang):
    return ambil_data_db(barang)

# =====================
# BUILD DATASET (SAFE)
# =====================
def build_training_data(barang_list):

    X = []
    y = []

    for barang in barang_list:

        data = ambil_data(barang)

        if len(data) < 5:
            continue

        avg = np.mean(data)
        mx = np.max(data)
        mn = np.min(data)
        std = np.std(data)

        # =====================
        # FIX: bikin variasi label biar tidak 1 class
        # =====================
        for i in range(len(data)):

            stok_simulasi = avg * np.random.uniform(0.7, 1.3)

            label = 1 if stok_simulasi > avg * 1.05 else 0

            X.append([avg, mx, mn, std, stok_simulasi])
            y.append(label)

    return np.array(X), np.array(y)

# =====================
# TRAIN MODEL SAFE
# =====================
barang_list = ["besi", "dus"]

X, y = build_training_data(barang_list)

model = None

if len(X) > 0 and len(np.unique(y)) > 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("Model berhasil dilatih")
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

    features = np.array([[
        np.mean(data),
        np.max(data),
        np.min(data),
        np.std(data),
        stok
    ]])

    pred = model.predict(features)[0]

    return "JUAL" if pred == 1 else "TUNGGU"

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
