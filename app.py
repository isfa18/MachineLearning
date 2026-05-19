from flask import Flask, request, jsonify
import os
import numpy as np
import mysql.connector

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =====================
# FLASK
# =====================
app = Flask(__name__)


# =====================
# DB CONFIG
# =====================
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "195.88.211.226"),
    "user": os.getenv("DB_USER", "ISI_USER_DATABASE"),
    "password": os.getenv("DB_PASSWORD", "ISI_PASSWORD_DATABASE"),
    "database": os.getenv("DB_NAME", "ISI_NAMA_DATABASE")
}


# =====================
# DB CONNECTION
# =====================
def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


# =====================
# BERSIHKAN ANGKA BERAT
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
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        query = """
        SELECT bk.berat
        FROM barang_keluar bk
        JOIN kategori_barang kb ON bk.id_kategoriBarang = kb.id
        WHERE LOWER(kb.nama_kategori) LIKE %s
        """

        cursor.execute(query, ("%" + barang.lower() + "%",))
        rows = cursor.fetchall()

        data = []

        for row in rows:
            berat = bersihkan_berat(row[0])
            if berat is not None:
                data.append(berat)

        return data

    except Exception as e:
        print("ERROR ambil_data_db:", e)
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# =====================
# AMBIL DATA
# =====================
def ambil_data(barang):
    return ambil_data_db(barang)


# =====================
# BUILD DATASET ML
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

        if len(data) < 5:
            print("Data kurang dari 5, dilewati")
            continue

        rata_rata = np.mean(data)
        maksimum = np.max(data)
        minimum = np.min(data)
        standar_deviasi = np.std(data)

        # Membuat data stok simulasi untuk training ML
        stok_simulasi_list = np.linspace(
            rata_rata * 0.3,
            rata_rata * 4.0,
            150
        )

        for stok_simulasi in stok_simulasi_list:
            batas_jual = rata_rata * 1.2

            # Label supervised learning
            # 1 = JUAL
            # 0 = TUNGGU
            label = 1 if stok_simulasi >= batas_jual else 0

            X.append([
                rata_rata,
                maksimum,
                minimum,
                standar_deviasi,
                stok_simulasi
            ])

            y.append(label)

    return np.array(X), np.array(y)


# =====================
# TRAIN MODEL
# =====================
barang_list = ["besi", "dus"]

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

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    akurasi_model = accuracy_score(y_test, y_pred)

    print("====================")
    print("MODEL BERHASIL DILATIH")
    print("Jumlah data training:", len(X))
    print("Class:", np.unique(y, return_counts=True))
    print("Akurasi:", akurasi_model)
    print("====================")

else:
    print("====================")
    print("MODEL TIDAK DILATIH")
    print("Cek data barang_keluar dan kategori_barang")
    print("====================")


# =====================
# PREDIKSI JUAL
# =====================
def prediksi_jual(barang, stok):
    data = ambil_data(barang)

    if model is None:
        return {
            "rekomendasi": "TUNGGU",
            "keterangan": "Model belum berhasil dilatih"
        }

    if len(data) < 2:
        return {
            "rekomendasi": "TUNGGU",
            "keterangan": "Data pengeluaran barang kurang"
        }

    rata_rata = np.mean(data)
    maksimum = np.max(data)
    minimum = np.min(data)
    standar_deviasi = np.std(data)
    batas_jual = rata_rata * 1.2

    features = np.array([[
        rata_rata,
        maksimum,
        minimum,
        standar_deviasi,
        stok
    ]])

    prediksi = model.predict(features)[0]

    hasil = "JUAL" if prediksi == 1 else "TUNGGU"

    return {
        "rekomendasi": hasil,
        "stok": round(float(stok), 2),
        "rata_rata_pengeluaran": round(float(rata_rata), 2),
        "pengeluaran_maksimum": round(float(maksimum), 2),
        "pengeluaran_minimum": round(float(minimum), 2),
        "standar_deviasi": round(float(standar_deviasi), 2),
        "batas_jual": round(float(batas_jual), 2),
        "akurasi_model": round(float(akurasi_model), 4) if akurasi_model is not None else None,
        "keterangan": "Rekomendasi dihasilkan menggunakan Logistic Regression"
    }


# =====================
# API HOME
# =====================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "API Prediksi Penjualan Barang Berjalan",
        "status_model": "aktif" if model is not None else "tidak aktif",
        "akurasi_model": round(float(akurasi_model), 4) if akurasi_model is not None else None,
        "endpoint_prediksi": "/prediksi",
        "endpoint_cek_data": "/cek-data?barang=besi"
    })


# =====================
# API CEK DATA
# =====================
@app.route("/cek-data", methods=["GET"])
def cek_data():
    try:
        barang = request.args.get("barang")

        if not barang:
            return jsonify({
                "error": "Parameter barang wajib diisi. Contoh: /cek-data?barang=besi"
            }), 400

        data = ambil_data(barang)

        if len(data) == 0:
            return jsonify({
                "barang": barang,
                "jumlah_data": 0,
                "data": [],
                "keterangan": "Data tidak ditemukan di database"
            })

        rata_rata = np.mean(data)

        return jsonify({
            "barang": barang,
            "jumlah_data": len(data),
            "data": data,
            "rata_rata_pengeluaran": round(float(rata_rata), 2),
            "pengeluaran_maksimum": round(float(np.max(data)), 2),
            "pengeluaran_minimum": round(float(np.min(data)), 2),
            "standar_deviasi": round(float(np.std(data)), 2),
            "batas_jual": round(float(rata_rata * 1.2), 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# =====================
# API PREDIKSI
# =====================
@app.route("/prediksi", methods=["POST"])
def prediksi():
    try:
        req = request.get_json()

        if req is None:
            return jsonify({
                "error": "Body JSON tidak boleh kosong",
                "rekomendasi": "TUNGGU"
            }), 400

        barang = req.get("barang")
        stok = req.get("stok")

        if not barang or stok is None:
            return jsonify({
                "error": "Field barang dan stok wajib diisi",
                "contoh": {
                    "barang": "besi",
                    "stok": 9100
                },
                "rekomendasi": "TUNGGU"
            }), 400

        stok = float(stok)

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
        }), 500


# =====================
# RUN UNTUK DEPLOY
# =====================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
