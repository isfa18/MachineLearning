from flask import Flask, request, jsonify
import numpy as np
import mysql.connector

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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
            berat = berat.strip()

            data.append(float(berat))
        except:
            pass

    cursor.close()
    return data


# =====================
# AMBIL DATA
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

        print("Barang:", barang)
        print("Data dari DB:", data)
        print("Jumlah data:", len(data))

        if len(data) < 5:
            print("Data kurang dari 5, dilewati:", barang)
            continue

        rata_rata = np.mean(data)
        maksimum = np.max(data)
        minimum = np.min(data)
        standar_deviasi = np.std(data)

        # Membuat variasi stok untuk data training
        # Dari stok kecil sampai stok besar
        stok_simulasi_list = np.linspace(
            rata_rata * 0.3,
            rata_rata * 3.5,
            100
        )

        for stok_simulasi in stok_simulasi_list:

            # Aturan label
            batas_jual = rata_rata * 1.2

            if stok_simulasi >= batas_jual:
                label = 1   # JUAL
            else:
                label = 0   # TUNGGU

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

    print("=====================")
    print("MODEL BERHASIL DILATIH")
    print("=====================")
    print("Jumlah data training:", len(X))
    print("Jumlah class:", np.unique(y, return_counts=True))
    print("Akurasi:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

else:
    print("=====================")
    print("MODEL TIDAK DILATIH")
    print("=====================")
    print("Penyebab kemungkinan:")
    print("1. Data dari database kosong")
    print("2. Data kurang dari 5")
    print("3. Label hanya memiliki 1 class")


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

    if prediksi == 1:
        hasil = "JUAL"
    else:
        hasil = "TUNGGU"

    return {
        "rekomendasi": hasil,
        "rata_rata_pengeluaran": round(float(rata_rata), 2),
        "pengeluaran_maksimum": round(float(maksimum), 2),
        "pengeluaran_minimum": round(float(minimum), 2),
        "standar_deviasi": round(float(standar_deviasi), 2),
        "batas_jual": round(float(batas_jual), 2),
        "stok": round(float(stok), 2),
        "keterangan": "JUAL jika stok lebih besar atau sama dengan batas jual"
    }


# =====================
# API PREDIKSI
# =====================
@app.route('/prediksi', methods=['POST'])
def prediksi():
    try:
        req = request.json

        barang = req.get("barang")
        stok = req.get("stok")

        if barang is None or stok is None:
            return jsonify({
                "error": "barang dan stok wajib diisi",
                "rekomendasi": "TUNGGU"
            })

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
        })


# =====================
# API CEK DATA
# =====================
@app.route('/cek-data', methods=['GET'])
def cek_data():
    try:
        barang = request.args.get("barang")

        if barang is None:
            return jsonify({
                "error": "Parameter barang wajib diisi. Contoh: /cek-data?barang=besi"
            })

        data = ambil_data(barang)

        if len(data) == 0:
            return jsonify({
                "barang": barang,
                "jumlah_data": 0,
                "data": [],
                "keterangan": "Data tidak ditemukan di database"
            })

        return jsonify({
            "barang": barang,
            "jumlah_data": len(data),
            "data": data,
            "rata_rata": round(float(np.mean(data)), 2),
            "maksimum": round(float(np.max(data)), 2),
            "minimum": round(float(np.min(data)), 2),
            "standar_deviasi": round(float(np.std(data)), 2),
            "batas_jual": round(float(np.mean(data) * 1.2), 2)
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
        "endpoint_prediksi": "/prediksi",
        "endpoint_cek_data": "/cek-data?barang=besi"
    })


# =====================
# RUN
# =====================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
