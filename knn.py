from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import numpy as np
import math
import re

from time import sleep

def parsing(r_file):
    file_data = np.loadtxt(r_file, delimiter=',')

    label = file_data[0:, 0]
    features = file_data[0:, 1:]

    return features, label


class knnClasifier():
    train_data = np.empty(1)
    train_label = np.empty(1)
    n_neighbor = 5
    distance = "Euclidean"


    def __init__(self,n_neighbor=5, distance="Euclidean"):
        self.n_neighbor = n_neighbor
        self.distance = distance


    def fit(self,X, y):
        self.train_data = X
        self.train_label =y

        return self


    def get_distance(self,X):
        result_dist = list()
        result_indices = list()

        for i in range(len(X)):
            if self.distance == 'Euclidean':
                target_dist = self.euclidean(X[i],self.train_data)
            elif self.distance == 'Manhattan':
                target_dist = self.manhattan(X[i],self.train_data)
            elif self.distance == 'L':
                target_dist = self.L(X[i],self.train_data)
            np_indices = np.argsort(target_dist)[:5]
            np_dist = np.take(target_dist, np_indices)
            result_dist.append(np_dist)
            result_indices.append(np_indices)

        result_dist = np.array(result_dist)
        result_indices = np.array(result_indices)
        return result_dist, result_indices

    def predict(self,X):
        dist_list, index_list = self.get_distance(X)
        labeling = np.take(self.train_label, index_list)
        result = list()
        for i in labeling:
            counting = Counter(i)
            result.append(counting.most_common(1)[0])

        return np.array(result, dtype=int)


    def manhattan(self,X,y):
        dist = np.sum(abs(X-y), axis=1)
        return dist.T
    def L(self,X,y):
        dist = np.amax(abs(y-X), axis=1)
        return dist
    def euclidean(self,X,y):
        dist = np.sqrt(np.sum(np.power(abs(X-y),2),axis=1))
        return dist.T