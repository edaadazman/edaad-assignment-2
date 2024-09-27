from flask import Flask, render_template, request, jsonify
import numpy as np
from kmeans import KMeans
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for rendering
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from PIL import Image
import sklearn.datasets as datasets

app = Flask(__name__)

# Global variable to store dataset
X, _ = datasets.make_blobs(
    n_samples=300,
    centers=[[0, 0], [2, 2], [-3, 2], [2, -4]],
    cluster_std=1,
    random_state=0,
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate_new_dataset", methods=["POST"])
def generate_new_dataset():
    global X
    centers = [[0, 0], [2, 2], [-3, 2], [2, -4]]
    X, _ = datasets.make_blobs(
        n_samples=300, centers=centers, cluster_std=1, random_state=None
    )
    return jsonify({"message": "New dataset generated successfully"})


@app.route("/run_kmeans", methods=["POST"])
def run_kmeans():
    k = int(request.json["k"])

    # Run KMeans algorithm
    kmeans = KMeans(X, k)
    kmeans.lloyds()

    # Convert the GIF and final image to base64 to send to the frontend
    images = kmeans.snaps
    img_data = []
    for img in images:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_data.append(img_str)

    # Create GIF
    gif_file = BytesIO()
    images[0].save(
        gif_file,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=500,
    )
    gif_base64 = base64.b64encode(gif_file.getvalue()).decode("utf-8")

    return jsonify({"images": img_data, "gif": gif_base64})


if __name__ == "__main__":
    app.run(debug=True)
