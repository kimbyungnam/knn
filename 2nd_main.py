import knn
from sklearn.neighbors import KNeighborsClassifier

train_file = "E:/2018-2/data_mining/homework1/hw1_data/digits_train.csv"
test_file = "E:/2018-2/data_mining/homework1/hw1_data/digits_test.csv"

sample, label = knn.parsing(train_file)
t_sample, t_label = knn.parsing(test_file)
K = knn.knnClasifier(distance='Euclidean').fit(sample, label)
nbrs = KNeighborsClassifier(n_neighbors=5, p=2).fit(sample, label)
mine_dist, mine_index = K.get_distance(t_sample)
knn_dist, knn_index = nbrs.kneighbors(t_sample)
mine_pred = K.predict(t_sample)
knn_pred = nbrs.predict(t_sample)
#print mine_pred == knn_pred
#print t_sample[0],"\n"
print knn_pred,"\n",t_label
#print "\n", knn_pred,"\n"