from flask import Flask, request, jsonify
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import os
import mysql.connector

app = Flask(__name__)

# =====================
# KONEKSI DATABASE
# =====================
conn = mysql.connector.connect(
    host="195.88.211.226",
    user="langgen1_lj_db",
    password="~ao-S%9UGMrU,^bP",
    database="langgen1_lj_db"
)

# =====================
# AMBIL DATA HYBRID (CSV + DB)
# =====================
def ambil_data(barang):
    try:
        barang = barang.lower()

        # ===== CSV =====
        data_csv = pd.read_csv("dataset_penjualan_rongsok_update_v2.csv", sep=";")
        data_csv = data_csv[data_csv["barang"] == barang]["berat_keluar_kg"]

        # ===== DATABASE =====
        query = """
        SELECT berat_kg 
        FROM barang_keluar 
        WHERE LOWER(nama_barang) = %s
        ORDER BY tanggal DESC
        LIMIT 50
        """
        data_db = pd.read_sql(query, conn, params=[barang])

        # ===== VALIDASI =====
        data_db_series = data_db["berat_kg"] if not data_db.empty else pd.Series()

        # ===== HYBRID (DB LEBIH DOMINAN) =====
        data_db_weighted = pd.concat([data_db_series] * 3)  # DB diperkuat
        semua_data = pd.concat([data_csv, data_db_weighted])

        return semua_data.dropna()

    except Exception as e:
        print("Error ambil data:", e)
        return pd.Series()

# =====================
# PREDIKSI
# =====================
def prediksi_jual(barang, stok):

    data = ambil_data(barang)

    if len(data) < 2:
        return "DATA TIDAK CUKUP"

    subset = data.values.reshape(-1,1)

    # KMeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(subset)

    centers = sorted(kmeans.cluster_centers_.flatten())

    batas_bawah = centers[0]
    batas_atas = centers[1]

    # 🔥 threshold lebih realistis
    batas_jual = (batas_bawah + batas_atas) / 2

    # DEBUG (opsional, bisa dihapus nanti)
    print("Centers:", centers)
    print("Batas jual:", batas_jual)
    print("Stok:", stok)

    if stok >= batas_jual:
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
