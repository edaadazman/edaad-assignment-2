import numpy as np


class KMeans:
    def __init__(self, data, k, init_method="random"):
        self.data = data
        self.k = k
        self.assignment = [-1 for _ in range(len(data))]
        self.centers_history = []  
        self.assignment_history = []  
        self.init_method = init_method

    def snap(self, centers):
        self.centers_history.append(centers.copy())
        self.assignment_history.append(self.assignment.copy())

    def isunassigned(self, i):
        return self.assignment[i] == -1

    def initialize(self):
        if self.init_method == "farthest_first":
            return self.initialize_farthest_first()
        elif self.init_method == "kmeans++":
            return self.initialize_kmeans_pp()
        else:  
            return self.initialize_random()

    def initialize_random(self):
        return self.data[np.random.choice(len(self.data), size=self.k, replace=False)]

    def initialize_kmeans_pp(self):
        centroids = []
        first_centroid_idx = np.random.randint(self.data.shape[0])
        first_centroid = self.data[first_centroid_idx]
        centroids.append(first_centroid)

        for _ in range(1, self.k):
            distances = []

            for point in self.data:
                min_dist = np.min(
                    [self.euclidean_dist(point, c) ** 2 for c in centroids]
                )
                distances.append(min_dist)

            probabilities = np.array(distances) / np.sum(distances)
            next_centroid_idx = np.random.choice(self.data.shape[0], p=probabilities)
            centroids.append(self.data[next_centroid_idx])

        return np.array(centroids)

    def initialize_farthest_first(self):
        centroids = []
        initial_centroid_idx = np.random.randint(self.data.shape[0])
        initial_centroid = self.data[initial_centroid_idx]
        centroids.append(initial_centroid)

        for _ in range(1, self.k):
            distances = []
            for point in self.data:
                dists = np.array([self.euclidean_dist(point, c) for c in centroids])
                min_dist = np.min(dists)
                distances.append(min_dist)

            max_idx = np.argmax(distances)
            centroids.append(self.data[max_idx])

        return np.array(centroids)

    def euclidean_dist(self, point, centroid):
        return np.sqrt(np.sum((point - centroid) ** 2))

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
            if cluster:
                centers.append(np.mean(np.array(cluster), axis=0))
            else:
                new_centroid = self.data[np.random.randint(self.data.shape[0])]
                centers.append(new_centroid)
        return np.array(centers)

    def unassign(self):
        self.assignment = [-1 for _ in range(len(self.data))]

    def are_diff(self, centers, new_centers):
        return any(self.dist(centers[i], new_centers[i]) != 0 for i in range(self.k))

    def dist(self, x, y):
        return np.linalg.norm(x - y)

    def lloyds(self, manual_centroids=None):
        if manual_centroids is not None and len(manual_centroids) == self.k:
            centers = np.array(manual_centroids)
        else:
            centers = self.initialize()

        self.snap(centers)

        self.make_clusters(centers)
        new_centers = self.compute_centers()
        self.snap(new_centers)  

        while self.are_diff(centers, new_centers):
            self.unassign()
            centers = new_centers
            self.make_clusters(centers)
            new_centers = self.compute_centers()
            self.snap(new_centers)
