import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
from io import BytesIO


class KMeans:
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.assignment = [-1 for _ in range(len(data))]
        self.snaps = []

    def snap(self, centers):
        # Instead of saving to a temp file, we save directly to memory using BytesIO
        buf = BytesIO()
        fig, ax = plt.subplots()
        ax.scatter(self.data[:, 0], self.data[:, 1], c=self.assignment)
        ax.scatter(centers[:, 0], centers[:, 1], c="r")
        fig.savefig(buf, format="png")  # Save directly to a BytesIO buffer
        plt.close()

        # Load the image directly into Pillow from the buffer
        buf.seek(0)
        self.snaps.append(im.open(buf))

    def isunassigned(self, i):
        return self.assignment[i] == -1

    def initialize(self):
        return self.data[np.random.choice(len(self.data), size=self.k, replace=False)]

    def make_clusters(self, centers):
        for i in range(len(self.assignment)):
            for j in range(self.k):
                if self.isunassigned(i):
                    self.assignment[i] = j
                    dist = self.dist(centers[j], self.data[i])
                else:
                    new_dist = self.dist(centers[j], self.data[i])
                    if new_dist < dist:
                        self.assignment[i] = j
                        dist = new_dist

    def compute_centers(self):
        centers = []
        for i in range(self.k):
            cluster = [
                self.data[j]
                for j in range(len(self.assignment))
                if self.assignment[j] == i
            ]
            centers.append(np.mean(np.array(cluster), axis=0))

        return np.array(centers)

    def unassign(self):
        self.assignment = [-1 for _ in range(len(self.data))]

    def are_diff(self, centers, new_centers):
        return any(self.dist(centers[i], new_centers[i]) != 0 for i in range(self.k))

    def dist(self, x, y):
        return np.linalg.norm(x - y)

    def lloyds(self):
        centers = self.initialize()
        self.make_clusters(centers)
        new_centers = self.compute_centers()
        self.snap(new_centers)
        while self.are_diff(centers, new_centers):
            self.unassign()
            centers = new_centers
            self.make_clusters(centers)
            new_centers = self.compute_centers()
            self.snap(new_centers)
        return
