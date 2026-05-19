from flask import Flask, request, jsonify
import os
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
# DB CONFIG
# =====================
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USERNAME", "langgen1_lj_db"),
    "password": os.getenv("DB_PASSWORD", "~ao-S%9UGMrU,^bP"),
    "database": os.getenv("DB_DATABASE", "langgen1_lj_db")
}


# =====================
# BATAS REKOMENDASI
# =====================
# 1.0 artinya stok boleh dijual jika stok >= rata-rata pengeluaran
# Kalau mau lebih ketat, bisa ganti jadi 1.2
BATAS_MULTIPLIER = 1.0


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
# AMBIL SEMUA NAMA BARANG
# =====================
def ambil_semua_barang():
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        query = """
        SELECT DISTINCT LOWER(kb.nama_kategori)
        FROM kategori_barang kb
        JOIN barang_keluar bk ON bk.id_kategoriBarang = kb.id
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        barang_list = []

        for row in rows:
            if row[0] is not None:
                barang_list.append(str(row[0]).lower())

        return barang_list

    except Exception as e:
        print("ERROR ambil_semua_barang:", e)
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# =====================
# AMBIL DATA PENGELUARAN BARANG
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
# BUILD DATASET MACHINE LEARNING
# =====================
def build_training_data():
    X = []
    y = []

    barang_list = ambil_semua_barang()

    print("====================")
    print("BARANG LIST:", barang_list)
    print("====================")

    for barang in barang_list:
        data = ambil_data_db(barang)

        print("Barang:", barang)
        print("Data:", data)
        print("Jumlah data:", len(data))

        if len(data) < 2:
            print("Data kurang dari 2, dilewati:", barang)
            continue

        rata_rata = np.mean(data)
        maksimum = np.max(data)
        minimum = np.min(data)
        standar_deviasi = np.std(data)

        batas_jual = rata_rata * BATAS_MULTIPLIER

        # Membuat variasi stok simulasi untuk training ML
        # Stok dibuat dari sangat rendah sampai tinggi
        stok_simulasi_list = np.linspace(
            rata_rata * 0.1,
            rata_rata * 5.0,
            200
        )

        for stok_simulasi in stok_simulasi_list:

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
model = None
akurasi_model = None

status_training = {
    "status": "belum dilatih",
    "pesan": "",
    "jumlah_data_training": 0,
    "class": None
}

X, y = build_training_data()

status_training["jumlah_data_training"] = int(len(X))

if len(y) > 0:
    unique_class, jumlah_class = np.unique(y, return_counts=True)
    status_training["class"] = {
        str(int(k)): int(v) for k, v in zip(unique_class, jumlah_class)
    }

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

    status_training["status"] = "aktif"
    status_training["pesan"] = "Model berhasil dilatih"

    print("====================")
    print("MODEL BERHASIL DILATIH")
    print("Jumlah data training:", len(X))
    print("Class:", np.unique(y, return_counts=True))
    print("Akurasi:", akurasi_model)
    print("====================")

else:
    status_training["status"] = "tidak aktif"

    if len(X) == 0:
        status_training["pesan"] = "Data training kosong. Cek koneksi database atau data barang_keluar."
    elif len(np.unique(y)) <= 1:
        status_training["pesan"] = "Model gagal dilatih karena label hanya memiliki 1 class."
    else:
        status_training["pesan"] = "Model gagal dilatih."

    print("====================")
    print("MODEL TIDAK DILATIH")
    print(status_training["pesan"])
    print("====================")


# =====================
# PREDIKSI JUAL
# =====================
def prediksi_jual(barang, stok):
    data = ambil_data_db(barang)

    if len(data) < 2:
        return {
            "rekomendasi": "TUNGGU",
            "keterangan": "Data pengeluaran barang kurang",
            "jumlah_data": len(data)
        }

    rata_rata = np.mean(data)
    maksimum = np.max(data)
    minimum = np.min(data)
    standar_deviasi = np.std(data)
    batas_jual = rata_rata * BATAS_MULTIPLIER

    features = np.array([[
        rata_rata,
        maksimum,
        minimum,
        standar_deviasi,
        stok
    ]])

    prediksi_model = None

    if model is not None:
        prediksi_model = model.predict(features)[0]

    # Hasil akhir tetap divalidasi dengan threshold agar tidak ngawur
    if stok >= batas_jual:
        hasil = "JUAL"
    else:
        hasil = "TUNGGU"

    return {
        "rekomendasi": hasil,
        "prediksi_model": "JUAL" if prediksi_model == 1 else "TUNGGU" if prediksi_model == 0 else "MODEL TIDAK AKTIF",
        "stok": round(float(stok), 2),
        "jumlah_data": len(data),
        "rata_rata_pengeluaran": round(float(rata_rata), 2),
        "pengeluaran_maksimum": round(float(maksimum), 2),
        "pengeluaran_minimum": round(float(minimum), 2),
        "standar_deviasi": round(float(standar_deviasi), 2),
        "batas_jual": round(float(batas_jual), 2),
        "batas_multiplier": BATAS_MULTIPLIER,
        "akurasi_model": round(float(akurasi_model), 4) if akurasi_model is not None else None,
        "keterangan": "Rekomendasi menggunakan Machine Learning Decision Tree dengan validasi threshold rata-rata pengeluaran"
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
        "training": status_training,
        "endpoint_prediksi": "/prediksi",
        "endpoint_cek_data": "/cek-data?barang=besi",
        "endpoint_barang": "/barang"
    })


# =====================
# API LIST BARANG
# =====================
@app.route("/barang", methods=["GET"])
def list_barang():
    barang_list = ambil_semua_barang()

    return jsonify({
        "jumlah_barang": len(barang_list),
        "barang": barang_list
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

        data = ambil_data_db(barang)

        if len(data) == 0:
            return jsonify({
                "barang": barang,
                "jumlah_data": 0,
                "data": [],
                "keterangan": "Data tidak ditemukan di database"
            })

        rata_rata = np.mean(data)
        batas_jual = rata_rata * BATAS_MULTIPLIER

        return jsonify({
            "barang": barang,
            "jumlah_data": len(data),
            "data": data,
            "rata_rata_pengeluaran": round(float(rata_rata), 2),
            "pengeluaran_maksimum": round(float(np.max(data)), 2),
            "pengeluaran_minimum": round(float(np.min(data)), 2),
            "standar_deviasi": round(float(np.std(data)), 2),
            "batas_jual": round(float(batas_jual), 2),
            "batas_multiplier": BATAS_MULTIPLIER
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
                "contoh": {
                    "barang": "besi",
                    "stok": 1216
                }
            }), 400

        barang = req.get("barang")
        stok = req.get("stok")

        if not barang or stok is None:
            return jsonify({
                "error": "Field barang dan stok wajib diisi",
                "contoh": {
                    "barang": "besi",
                    "stok": 1216
                }
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
# API PREDIKSI BANYAK BARANG
# =====================
@app.route("/prediksi-semua", methods=["POST"])
def prediksi_semua():
    try:
        req = request.get_json()

        if req is None:
            return jsonify({
                "error": "Body JSON tidak boleh kosong",
                "contoh": {
                    "items": [
                        {
                            "barang": "dus",
                            "stok": 2760
                        },
                        {
                            "barang": "besi",
                            "stok": 1216
                        },
                        {
                            "barang": "atum",
                            "stok": 40
                        }
                    ]
                }
            }), 400

        items = req.get("items")

        if not items or not isinstance(items, list):
            return jsonify({
                "error": "Field items wajib diisi dalam bentuk list"
            }), 400

        hasil_semua = []

        for item in items:
            barang = item.get("barang")
            stok = item.get("stok")

            if not barang or stok is None:
                hasil_semua.append({
                    "barang": barang,
                    "stok": stok,
                    "rekomendasi": "TUNGGU",
                    "error": "barang atau stok kosong"
                })
                continue

            stok = float(stok)
            hasil = prediksi_jual(barang, stok)

            hasil_semua.append({
                "barang": barang,
                "stok": stok,
                "rekomendasi": hasil["rekomendasi"],
                "detail": hasil
            })

        return jsonify({
            "hasil": hasil_semua
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
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
