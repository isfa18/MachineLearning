from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import mysql.connector
import os

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
# LOAD CSV SEKALI SAJA (BIAR CEPAT)
# =====================
try:
    data_csv_global = pd.read_csv("dataset_penjualan_rongsok_update_v2.csv", sep=";")
except:
    data_csv_global = pd.DataFrame()

# =====================
# AMBIL DATA HYBRID (ADAPTIF)
# =====================
def ambil_data(barang):
    try:
        barang = barang.lower()

        # ===== CSV =====
        if not data_csv_global.empty:
            data_csv = data_csv_global[
                data_csv_global["barang"] == barang
            ]["berat_keluar_kg"].tolist()
        else:
            data_csv = []

        # ===== DATABASE =====
        cursor = conn.cursor()
        query = """
        SELECT bk.berat 
        FROM barang_keluar bk
        JOIN kategori_barang kb ON bk.id_kategoriBarang = kb.id
        WHERE LOWER(kb.nama_kategori) = %s
        ORDER BY bk.tanggal_keluar DESC
        """
        cursor.execute(query, (barang,))
        rows = cursor.fetchall()

        data_db = [row[0] for row in rows if row[0] is not None]

        n_db = len(data_db)

        # ===== HYBRID ADAPTIF =====
        if n_db == 0:
            data = data_csv

        elif n_db < 5:
            data = data_csv + data_db

        elif n_db < 20:
            data = data_csv + (data_db * 2)

        elif n_db < 50:
            data = data_csv + (data_db * 3)

        else:
            data = data_db

        return data

    except Exception as e:
        print("Error ambil data:", e)
        return []

# =====================
# PREDIKSI
# =====================
def prediksi_jual(barang, stok):

    data = ambil_data(barang)

    if len(data) < 2:
        return "DATA TIDAK CUKUP"

    try:
        subset = np.array(data).reshape(-1,1)

        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(subset)

        centers = sorted(kmeans.cluster_centers_.flatten())

        # 🔥 threshold tengah (WAJIB)
        batas_jual = (centers[0] + centers[1]) / 2

        # debug (opsional)
        print("Centers:", centers)
        print("Batas jual:", batas_jual)
        print("Stok:", stok)

        if stok >= batas_jual:
            return "JUAL"
        else:
            return "TUNGGU"

    except Exception as e:
        return f"ERROR KMEANS: {str(e)}"

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
# RUN
# =====================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
