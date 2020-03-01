import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):


    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly (use Euclidean distance to calculate other centers)
    # Loop n_cluster-1 times:
    #   compute distance squared from each point to the nearest cluster
    #   select a point with the largest probability (squared_distance/sum(squared_distance))
    def find_nearest_point(points, centroid_idx):
        
        #k = np.zeros((len(points), 2))
        #k[:] = x[int(centroid)]
        #print('k[:] = x[int(centroid)]: \t', k)
        print(points[0],centroid_idx)
        distances = (x[points] - x[centroid_idx]) ** 2
        distances = np.sum(distances, axis=1)
        p = np.argmin(distances)
        # points = np.append(points[0: p], points(p + 1: len(x)), axis=0)
        return points[p]  # points

    def find_next_center(points, nearest_point_idx):
        #p = np.zeros((len(points), 2))
        #p[:] = x[nearest_point_idx]
        distances = (x[points] - x[nearest_point_idx]) ** 2
        distances = np.sum(distances, axis=1)
        k = np.argmax(distances)
        center_idx=points[k]
        points = np.append(points[0: k], points[(k + 1): n])

        return center_idx, points

    centers = np.zeros(n_cluster).astype(int)  # x index of each centroid
    idx = generator.randint(0, n)  # first cluster index
    centers[0] = int(idx)
    points = np.append(np.arange(0, idx), np.arange(idx + 1, len(x)), axis=0)
    for i in np.arange(0, n_cluster - 1):
        print(centers)
        nearest_idx = int(find_nearest_point(points, centers[i]))
        centers[i + 1], points = find_next_center(points, nearest_idx)


    centers=list(centers)
    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers


def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)


class KMeans():


    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"

        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE


        def assign_points(self, x, centers):
            N = x.shape[0]
            distances = np.zeros((N, self.n_cluster))

            for i in range(self.n_cluster):
                distances[:, i] = np.sum((x - centers[i]) ** 2, axis=1)

            y = np.argmin(distances, axis=1)
            return y

        def calculate_centroids(self, x, y):
            d = x.shape[1]
            centroids = np.zeros((self.n_cluster, d))

            for i in range(self.n_cluster):
                membership = (y == i).reshape([len(y),1]) #reshape([-1, 1])
                if np.sum(membership)==0:
                    continue
                centroids[i] = np.sum(membership * x, axis=0) / np.sum(membership)
                #centers[i] = np.sum(membership * x, axis=0) / (np.sum(membership) + 1e-10)

            return centroids

        def measure_distortion(self, x, centers):
            N = x.shape[0]
            distance = np.zeros((N, self.n_cluster))

            for i in range(self.n_cluster):
                distance[:, i] = np.sum((x - centroids[i]) ** 2, axis=1)
            return np.sum(np.min(distance, axis=1))

        idx = self.generator.choice(N, size=self.n_cluster)
        centroids = x[idx]
        J = 1e10
        for i in range(self.max_iter):
            y = assign_points(self, x, centroids)
            #distance = np.zeros((N, self.n_cluster))
            #for i in range(self.n_cluster):
            #    distance[:, i] = np.sum((x - centers[i]) ** 2, axis=1)
            #j=np.sum(np.min(distance, axis=1))
            j = measure_distortion(self, x, centroids)
            if (np.abs(J - j) / N < self.e):
                return centroids, y, i + 1

            centroids = calculate_centroids(self, x, y)
            J = j
        
        #self.y=y
        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, self.max_iter


class KMeansClassifier():


    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, y, centroid_func=get_lloyd_k_means):


        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centers
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE

        k_means = KMeans(self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, k, max_iterations = k_means.fit(x)

        centroid_labels = []
        for i in range(self.n_cluster):
            yi = y[(k == i)]
            if (yi.size > 0):
                _, idx, counts = np.unique(
                    yi, return_index=True, return_counts=True)

                index = idx[np.argmax(counts)]
                centroid_labels.append(yi[index])
            else:
                centroid_labels.append(0)
        centroid_labels = np.array(centroid_labels)
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centers = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centers.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        n_cluster = self.centers.shape[0]
        N = x.shape[0]
        #distances = np.zeros((N, self.n_cluster))
        distances = np.zeros((N, n_cluster))

        #for i in range(self.n_cluster)
        for i in range(n_cluster):

            distances[:, i] = np.sum((x - self.centers[i]) ** 2, axis=1)
        centroid_idx = np.argmin(distances, axis=1)
        labels = []
        for i in centroid_idx:
            labels.append(self.centroid_labels[i])

        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)


def transform_image(image, code_vectors):


    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE

    #quantizer=KMeansClassifier(code_vectors.shape[0])
    #quantizer.centers=code_vectors

    #data = image.shape[:2]

    # convert to RGB array
    N=image.shape[0]
    M=image.shape[1]
    data = image.reshape(N * M, 3)

    coded_index=[np.argmin(np.sum((point - code_vectors)**2, axis=1), axis=0) for point in data]

    new_im=[code_vectors[y] for y in coded_index]
    new_im=np.array(new_im).reshape((N,M,3))

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im
