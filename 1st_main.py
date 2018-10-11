import numpy as np
import pandas as pd
import knn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

#data files
train_file = open("E:/2018-2/data_mining/homework1/hw1_data/cancer_train.csv", "r")
test_file = open("E:/2018-2/data_mining/homework1/hw1_data/cancer_test.csv", "r")

#parsing data
sample_data, sample_label = knn.parsing(train_file)
test_data, test_label = knn.parsing(test_file)

#KNN
nbrs = KNeighborsClassifier(n_neighbors=5).fit(sample_data, sample_label)
# print nbrs.score(sample_data, sample_label)
#Decision tree
dt = DecisionTreeClassifier().fit(sample_data, sample_label)
# print dt.score(test_data,test_label)
#SVM
clf = svm.SVC().fit(sample_data,sample_label)
#print clf.score(sample_data, sample_label)
#print clf.score(test_data,test_label)

#hyper_parameter_tunning
    #knn
k_list = [i+1 for i in range(20)]
grid_knn_graph = GridSearchCV(nbrs, {'n_neighbors' : k_list},cv=8, scoring='accuracy', return_train_score=True)
grid_knn = GridSearchCV(nbrs, {'n_neighbors' : k_list, 'metric' : ['minkowski','euclidean','manhattan'], 'p':[2,1,float('inf')],
                               'weights' : ['uniform','distance']}, cv=8, scoring='accuracy', return_train_score=True)
grid_knn_graph.fit(test_data,test_label)
grid_knn.fit(test_data,test_label)
pd.DataFrame(grid_knn.cv_results_).to_csv("./knn_hyper_parameter_tunning.csv")
plt.figure("knn")
plt.plot(k_list, grid_knn_graph.cv_results_['mean_test_score'])
plt.xlabel('K')
plt.ylabel('accuracy')
print "KNN\nBest : ",grid_knn.best_score_,"\tparam : ",grid_knn.best_params_,"\testimator : ",grid_knn.best_estimator_

    #Decision tree
len_depth = [i+3 for i in range(30)]
grid_dt = GridSearchCV(dt, {'max_depth':len_depth}, cv=8, scoring='accuracy', return_train_score = True)
grid_dt.fit(test_data,test_label)
plt.figure("decision tree")
plt.plot(len_depth, grid_dt.cv_results_['mean_test_score'])
plt.xlabel('depth')
plt.ylabel('accuracy')
pd.DataFrame(grid_dt.cv_results_).to_csv("./dt_hyper_parameter_tunning.csv")
print "DT\nBest : ",grid_dt.best_score_,"\tparam : ",grid_dt.best_params_,"\testimator : ",grid_dt.best_estimator_

    #svm
gamma = [0.1,0.01,0.001,0.0001]
grid_svm = GridSearchCV(clf, {'gamma':gamma}, cv= 8, return_train_score=True)
grid_svm.fit(test_data,test_label)
plt.figure("svm")
plt.plot(gamma, grid_svm.cv_results_['mean_test_score'])
plt.xlabel('gamma')
plt.ylabel('accuracy')
print "SVM\nBest : ",grid_svm.best_score_,"\tparam : ",grid_svm.best_params_,"\testimator : ",grid_svm.best_estimator_

#confusion metrix
    #knn
final_nbrs = KNeighborsClassifier(n_neighbors=3,p=2,metric='minkowski',weights='uniform').fit(sample_data,sample_label)
final_pred_knn = final_nbrs.predict(test_data)
print "knn\n",confusion_matrix(test_label, final_pred_knn)
print classification_report(test_label, final_pred_knn, target_names=['P','N'])
    #decision tree
final_dt = DecisionTreeClassifier(max_depth=6,criterion='gini',min_samples_split=2).fit(sample_data,sample_label)
final_pred_dt = final_dt.predict(test_data)
print "decision tree\n",confusion_matrix(test_label, final_pred_dt)
print classification_report(test_label, final_pred_dt, target_names=['P','N'])
    #svm
final_clf = svm.SVC(gamma=0.01).fit(sample_data,sample_label)
final_pred_clf = final_clf.predict(test_data)
print "svm\n",confusion_matrix(test_label,final_pred_clf)
print classification_report(test_label, final_pred_clf, target_names=['P','N'])
plt.show()