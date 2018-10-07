from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import numpy as np
import math
import re

from time import sleep

def parsing(r_file):
        label = list()
        sample = list()
        while True:
            line = r_file.readline()
            if not line: break
            #data = line.split(",")
            data = re.findall(r"[\w]+",line)
            label.append(data[0])
            temp_sample = list()
            for i in range(1, len(data)):
                temp_sample.append(data[i])
            #print temp_sample
            sample.append(temp_sample)

        return sample, label

class knnClasifier():
    train_data = list()
    train_label = list()
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
        test = list()
        for i in range(len(X)):
            min_dist = list()
            for j in range(len(self.train_data)):
                if self.distance == 'Euclidean':
                    target_dist = self.euclidean(X[i],self.train_data[j])
                elif self.distance == 'Manhattan':
                    target_dist = self.manhattan(X[i],self.train_data[j])
                elif self.distance == 'L':
                    target_dist = self.L(X[i],self.train_data[j])

                min_dist.append([target_dist, j])
            #print min_dist

            #test.append(sorted(min_dist, key = lambda data:data[0])[:self.n_neighbor])
            result_dist.append(map(lambda x:x[0], sorted(min_dist, key = lambda data:data[0])[:self.n_neighbor]))
            result_indices.append(map(lambda x:x[1], sorted(min_dist, key = lambda data:data[0])[:self.n_neighbor]))
        #test.append(sorted(min_dist, key = lambda data:data[0])[:self.n_neighbor])
        #print test
        #sleep(10)
        return result_dist, result_indices

    def predict(self,X):
        result = list()
        dist_list, index_list = self.get_distance(X)
        for index in index_list:
            label_list = list()
            for i in index:
               label_list.append(self.train_label[i])

            label_ctr = Counter(label_list)
            result.append(label_ctr.most_common(1)[0][0])

        return result


    def manhattan(self,X,Y):
        dist =0
        for x,y in zip(X,Y):
            dist += abs(float(x)-float(y))

        return dist
    def L(self,X,Y):
        dist = 0
        for x,y in zip(X,Y):
            dist = max(dist, abs(float(x)-float(y)))

        return dist
    def euclidean(self,X,Y):
        dist = 0
        for x,y in zip(X,Y):
            dist += (float(x)-float(y))**2

        return math.sqrt(dist)