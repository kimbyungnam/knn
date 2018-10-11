import knn
import numpy as np

train_file = "E:/2018-2/data_mining/homework1/hw1_data/digits_train.csv"
test_file = "E:/2018-2/data_mining/homework1/hw1_data/digits_test.csv"

sample, label = knn.parsing(train_file)
t_sample, t_label = knn.parsing(test_file)

cnt = 0

    #knn classifier
train_file = "E:/2018-2/data_mining/homework1/hw1_data/digits_train.csv"
test_file = "E:/2018-2/data_mining/homework1/hw1_data/digits_test.csv"

K = knn.knnClasifier(distance='Euclidean').fit(sample, label)
mine_dist, mine_index = K.get_distance(t_sample)
mine_pred = K.predict(t_sample)

cv = knn.validation(mine_pred,t_label, 10)
print cv.precision(),"\n",cv.recall(),"\n",cv.F1_measure(),"\n",cv.accuracy(),"\n"

    #for k fold cross validation
# for x,y in knn.k_fold(sample, k=5):
#     cnt += 1
#     #print np.take(sample,x)
#
#     k_train_data = sample[x]
#     k_train_label = label[x]
#     k_test_data = sample[y]
#     k_test_label = label[y]
#
#     K = knn.knnClasifier(n_neighbor=5,distance='Euclidean').fit(k_train_data, k_train_label)
#     mine_dist, mine_index = K.get_distance(k_test_data)
#     mine_pred = K.predict(k_test_data)
#
#     cv = knn.validation(mine_pred,k_test_label, 10)
#     print "#%d\nprecision : "%cnt,cv.precision(),"\nrecall : ",cv.recall(),"\nf1-measure : ",cv.F1_measure(),"\naccuracy : ",cv.accuracy(),"\n"