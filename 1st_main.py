import knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

#data files
train_file = open("E:/2018-2/data_mining/homework1/hw1_data/cancer_train.csv", "r")
test_file = open("E:/2018-2/data_mining/homework1/hw1_data/cancer_test.csv", "r")

#parsing data
sample_data, sample_label = knn.parsing(train_file)
test_data, test_label = knn.parsing(test_file)

#KNN
nbrs = KNeighborsClassifier(n_neighbors=5).fit(sample_data, sample_label)
#for x,y in zip(test_data, test_label):
 #   print nbrs.predict(x)
print nbrs.predict(test_data)
#print "KNN",nbrs.score(test_data, test_label),

#Decision tree
# dt = DecisionTreeClassifier().fit(sample_data, sample_label)
# print "Decision tree",dt.score(test_data, test_label),
#
# #SVM
# clf = svm.SVC().fit(sample_data,sample_label)
# print "SVM", clf.score(test_data, test_label)
