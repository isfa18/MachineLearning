from flask import Flask, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import os

app = Flask(__name__)

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "dataset_penjualan_rongsok_update_v2.csv")

data = pd.read_csv(file_path, sep=";")

# rapikan data
data["barang"] = data["barang"].str.lower()

model = {}

for barang in data["barang"].unique():
    subset = data[data["barang"] == barang]

    # fitur (X) → stok & penjualan
    X = subset[["stok", "berat_keluar_kg"]]

    # label (y) → buat sendiri
    y = np.where(subset["berat_keluar_kg"] > subset["berat_keluar_kg"].mean(), "JUAL", "TUNGGU")

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    model[barang] = clf


def prediksi_jual(barang, stok):
    barang = barang.lower()

    if barang not in model:
        return "DATA TIDAK ADA"

    subset = data[data["barang"] == barang]

    rata_keluar = subset["berat_keluar_kg"].mean()

    # prediksi pakai model
    clf = model[barang]
    pred = clf.predict([[stok, rata_keluar]])[0]

    return pred


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
