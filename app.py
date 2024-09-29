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
    n_samples = 500  # The number of points to generate
    num_features = 2  # Dimensionality of the dataset (2D points)

    # Generate completely random points between -5 and 5
    X = np.random.uniform(low=-5, high=5, size=(n_samples, num_features))

    return jsonify(
        {"message": "New dataset generated successfully", "points": X.tolist()}
    )


@app.route("/run_kmeans", methods=["POST"])
def run_kmeans():
    k = int(request.json["k"])
    init_method = request.json.get(
        "init_method", "random"
    )  # Get init method, default to 'random'

    # Run KMeans algorithm with the selected initialization method
    kmeans = KMeans(X, k, init_method=init_method)
    kmeans.lloyds()

    # Send all the snapshots (data points, assignments, centroids) to the frontend
    history_data = []
    for i in range(len(kmeans.assignment_history)):
        step_data = {
            "assignments": kmeans.assignment_history[i],
            "centers": kmeans.centers_history[i].tolist(),
        }
        history_data.append(step_data)

    return jsonify({"history": history_data, "points": X.tolist()})


if __name__ == "__main__":
    app.run(debug=True)
