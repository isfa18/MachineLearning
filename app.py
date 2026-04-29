from flask import Flask, request, jsonify
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import os

app = Flask(__name__)

# =====================
# LOAD DATA
# =====================
data = pd.read_csv("dataset_penjualan_rongsok_update_v2.csv", sep=";")

model = {}

for barang in data["barang"].unique():
    subset = data[data["barang"] == barang]["berat_keluar_kg"].values.reshape(-1,1)

    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(subset)

    centers = sorted(kmeans.cluster_centers_.flatten())

    model[barang] = {
        "kmeans": kmeans,
        "centers": centers
    }

# =====================
# PREDIKSI BARU
# =====================
def prediksi_jual(barang, stok):
    if barang not in model:
        return "DATA TIDAK ADA"

    centers = model[barang]["centers"]

    batas_bawah = centers[0]
    batas_atas = centers[1]

    if stok >= batas_atas:
        return "JUAL"
    elif stok >= batas_bawah:
        return "PERTIMBANGKAN"
    else:
        return "TUNGGU"

# =====================
# API
# =====================
@app.route('/prediksi', methods=['POST'])
def prediksi():
    req = request.json

    barang = req.get('barang')
    stok = req.get('stok')

    hasil = prediksi_jual(barang, stok)

    return jsonify({
        "barang": barang,
        "stok": stok,
        "rekomendasi": hasil
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
