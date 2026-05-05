from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import mysql.connector
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# =====================
# FLASK APP
# =====================
app = Flask(__name__)

# =====================
# DATABASE CONNECTION
# =====================
conn = mysql.connector.connect(
    host="195.88.211.226",
    user="langgen1_lj_db",
    password="~ao-S%9UGMrU,^bP",
    database="langgen1_lj_db"
)

# =====================
# LOAD CSV
# =====================
try:
    data_csv = pd.read_csv("dataset_penjualan_rongsok_update_v2.csv", sep=";")
    data_csv["barang"] = data_csv["barang"].str.lower()
except:
    data_csv = pd.DataFrame()

# =====================
# AMBIL DATA DATABASE
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
            data.append(float(str(r[0]).replace("Kg", "").strip()))
        except:
            pass

    return data

# =====================
# GABUNG CSV + DB
# =====================
def ambil_data(barang):

    # CSV
    if not data_csv.empty:
        data_csv_filtered = data_csv[
            data_csv["barang"] == barang.lower()
        ]["berat_keluar_kg"].tolist()
    else:
        data_csv_filtered = []

    # DB
    data_db = ambil_data_db(barang)

    # GABUNG
    data = data_csv_filtered + data_db

    return data

# =====================
# BUILD DATASET TRAINING
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

        # simulasi stok training (proxy label)
        stok = avg * np.random.uniform(0.8, 1.2)

        label = 1 if stok >= avg else 0

        X.append([avg, mx, mn, std, stok])
        y.append(label)

    return np.array(X), np.array(y)

# =====================
# TRAIN MODEL ML
# =====================
barang_list = ["besi", "dus"]

X, y = build_training_data(barang_list)

model = LogisticRegression()

if len(X) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)

# =====================
# PREDIKSI FUNCTION
# =====================
def prediksi_jual(barang, stok):

    data = ambil_data(barang)

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
# API ENDPOINT
# =====================
@app.route('/prediksi', methods=['POST'])
def prediksi():

    req = request.json

    if not req:
        return jsonify({"error": "Request kosong"}), 400

    barang = req.get("barang")
    stok = req.get("stok")

    if barang is None or stok is None:
        return jsonify({"error": "Input tidak lengkap"}), 400

    try:
        stok = float(stok)
    except:
        return jsonify({"error": "Stok harus angka"}), 400

    hasil = prediksi_jual(barang, stok)

    return jsonify({
        "barang": barang,
        "stok": stok,
        "rekomendasi": hasil
    })

# =====================
# RUN APP
# =====================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
