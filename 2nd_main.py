import knn
from sklearn.neighbors import KNeighborsClassifier

train_file = open("E:/2018-2/data_mining/homework1/hw1_data/digits_train.csv", "r")
test_file = open("E:/2018-2/data_mining/homework1/hw1_data/digits_test.csv", "r")

sample, label = knn.parsing(train_file)
t_sample, t_label = knn.parsing(test_file)
K = knn.knnClasifier(distance='L').fit(sample, label)
nbrs = KNeighborsClassifier(n_neighbors=5, p=float("inf")).fit(sample, label)
mine_dist, mine_index = K.get_distance(t_sample)
knn_dist, knn_index = nbrs.kneighbors(t_sample)
mine_pred = K.predict(t_sample)
knn_pred = nbrs.predict(t_sample)
print mine_pred == knn_pred
#for i in range(len(knn_pred)):
#    if not knn_pred[i] == mine_pred[i]:
#        print "dist : ",knn_dist[i],mine_dist[i],"\n","index : ",knn_index[i],mine_index[i],"\n"
# for x,y in zip(mine_dist,knn_dist):
#     for i in range(len(x)):
#         if not x[i] == y[i]:
#             print "dist : ",x, y,"\n"
# for x,y in zip(mine_index, knn_index):
#     for i in range(len(x)):
#         if not x[i] == y[i]:
#             print "index : ",x, y,"\n"
#print K.predict(sample) == nbrs.predict(sample)