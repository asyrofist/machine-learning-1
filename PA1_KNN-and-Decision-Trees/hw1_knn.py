from __future__ import division, print_function

from typing import List

import numpy as np
import scipy

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

from collections import Counter
class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    #TODO: save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        self.features = features
        self.labels = labels
        return

    #TODO: predict labels of a list of points
    def predict(self, features: List[List[float]]) -> List[int]:
        # features: List[List[float]] a list of points
        # return: List[int] a list of predicted labels
        predictions=[]
        for test_case in features:
            neighbors = self.get_k_neighbors(test_case)
            predictions.append(Counter(neighbors).most_common(1)[0][0])
        return predictions

    #TODO: find KNN of one point
    def get_k_neighbors(self, point: List[float]) -> List[int]:
        # point: List[float] one example
        # return: List[int] labels of K nearest neighbor
        
        distances = []
        nn_labels=[]
        for i in np.arange(len(self.features)):
            distance = self.distance_function(point, self.features[i])
            distances.append([distance, i])
        distances= sorted(distances)
        #print (distances)
        for i in np.arange(self.k):
            dist_index=distances[i][1]
            nn_labels.append(self.labels[dist_index])

        self.neighbors= dist_index
        return nn_labels

    #TODO: Do the classification 
    def test_classify(model):
        from data import test_processing
        model.train(Xtrain, ytrain)
        predictions=model.predict(Xtest)
        print('Classification f1_score:', f1_score(ytest,predictions))
        return
if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
