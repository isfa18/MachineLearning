from flask import Flask, request, jsonify
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import os

app = Flask(__name__)

# =====================
# LOAD DATA
# =====================
try:
    data = pd.read_csv("dataset_penjualan_rongsok_update_v2.csv", sep=";")
except Exception as e:
    print("Error load dataset:", e)
    data = pd.DataFrame()

model = {}

# =====================
# TRAIN MODEL KMEANS
# =====================
if not data.empty:
    for barang in data["barang"].unique():
        subset = data[data["barang"] == barang]["berat_keluar_kg"]

        # cek data cukup
        if len(subset) < 2:
            continue

        subset = subset.values.reshape(-1, 1)

        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(subset)

        centers = sorted(kmeans.cluster_centers_.flatten())

        model[barang.lower()] = {
            "centers": centers
        }

# =====================
# FUNGSI PREDIKSI
# =====================
def prediksi_jual(barang, stok):
    if barang not in model:
        return "DATA TIDAK ADA"

    centers = model[barang]["centers"]

    if len(centers) < 2:
        return "DATA TIDAK CUKUP"

    batas_bawah = centers[0]
    batas_atas = centers[1]

    # LOGIC BARU
    if stok >= batas_atas:
        return "JUAL"
    else:
        return "TUNGGU"

# =====================
# API ENDPOINT
# =====================
@app.route('/prediksi', methods=['POST'])
def prediksi():
    try:
        req = request.json

        if not req:
            return jsonify({"error": "Request kosong"}), 400

        barang = req.get('barang')
        stok = req.get('stok')

        if not barang or stok is None:
            return jsonify({"error": "Input tidak lengkap"}), 400

        barang = str(barang).lower()

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

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# =====================
# RUN APP
# =====================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
