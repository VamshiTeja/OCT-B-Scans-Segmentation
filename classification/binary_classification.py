# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-04-25 12:43:38
# @Last Modified by:   vamshi
# @Last Modified time: 2019-01-24 00:11:07

import os,sys
import numpy as np
import cv2
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import itertools

import pickle

from itertools import cycle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import xgboost
from xgboost import XGBClassifier
from sklearn.preprocessing import label_binarize
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import classification_report

import sklearn.metrics as metrics

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

random.seed(42)

train_dir = "./train/"
train_labels_dir = "./train_labels/"

images  = []
class_labels = []

print "loading data"

for file in os.listdir(train_dir):
    img = cv2.imread(os.path.join(train_dir,file),0)
    img = img.flatten()
    img_label = cv2.imread(os.path.join(train_labels_dir,file[:-4]+".png"),1)
    images.append(img)
    b,g,r = cv2.split(img_label)

    if(np.sum(r)==0 and np.sum(g)==0):
        class_labels.append(0)
    else:
        class_labels.append(1)

print("Done")
print "Splitting data"

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def Classify(data_list, target_list, kfold, clf,clf_name="Random Forest Classifier"):
    conf_matrices = []
    Y = target_list
    X = data_list
    cvscores, conf_matrices, accuracy = [], [], []
    precisions = []
    recalls = []
    f1scores = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    max_acc = 0.0
    for train,test in kfold.split(X,Y):
        clf.fit(X[train], Y[train])
        pred_labels = clf.predict(X[test])
        true_labels = Y[test]

        print("Accuracy =", clf.score(X[test], Y[test]))

        (true_positive, true_negative, false_positive, false_negative) = (0.0, 0.0, 0.0, 0.0)
        for i in range(0, len(pred_labels)):
            if (pred_labels[i], true_labels[i]) == (1, 1):
                true_positive = true_positive + 1
            if (pred_labels[i], true_labels[i]) == (0, 0):
                true_negative = true_negative + 1
            if (pred_labels[i], true_labels[i]) == (1, 0):
                false_positive = false_positive + 1
            if (pred_labels[i], true_labels[i]) == (0, 1):
                false_negative = false_negative + 1
        detection_rate = true_positive / (true_positive + false_negative)
        false_alarm = false_positive / (true_negative + false_positive)
        acc = (true_positive + true_negative) / (true_positive + false_negative + true_negative + false_positive)

        precision = true_positive/(true_positive+false_positive)
        recall =  true_positive/(true_positive+false_negative)
        f1score = 2*precision*recall/(precision+recall)

        precisions.append(precision)
        recalls.append(recall)
        f1scores.append(f1score)

        conf_matrix = np.array([[detection_rate, 1 - detection_rate], [false_alarm, 1 - false_alarm]])
        accuracy.append(acc)
        conf_matrices.append(conf_matrix)
        score = clf.score(X[test], Y[test])
        cvscores.append(score)
        preds = clf.predict_proba(X[test])[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(Y[test], preds)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        if isinstance(clf, RandomForestClassifier):
            name = "rfc"
        else:
            name = "xgb"
        if acc>max_acc:
            max_acc = acc
            pickle.dump(clf, open(name+"_binary.pickle", "wb"))
        #save model file

    fig = plt.figure()
    #plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    print len(mean_tpr)
    print mean_tpr
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    print("Mean AUC: %.2f"%mean_auc)
    std_auc = np.std(aucs)
    print ("STD AUC: %.2f"%std_auc)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    # plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f)' % (mean_auc), lw=2,
    #          alpha=.8)
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='black', alpha=.2,
    #                  label=r'$\pm$ %0.2f std. dev.'%(std_auc))
    #
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([0.8, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristics')
    # plt.legend(loc="lower right")
    # plt.savefig(clf_name+".eps", format="eps", dpi=fig.dpi)
    # plt.show()

    print("Mean Test Accuracy: %.5f, STD of Accuracy: %.5f"%(np.mean(accuracy),np.std(accuracy)))
    print("Mean Test Precision: %.5f, Mean Test Recall: %.5f "%(np.mean(precisions), np.std(precisions)))
    print("Mean Test Recall: %.5f, STD of Recall: %.5f "%(np.mean(recalls), np.std(recalls)))
    print("Mean Test F1 score: %.5f, STD of F1 score: %.5f"%(np.mean(f1scores), np.std(f1scores)))

    conf_matrices = np.asarray(conf_matrices)
    accuracy_avg = np.mean(accuracy)
    accuracy_var = np.var(accuracy)
    conf_mat_avg = np.mean(conf_matrices, axis=0)
    conf_mat_var = np.var(conf_matrices, axis=0)
    #plt.figure()
    class_names = np.asarray(['Gradable', 'Non-Gradable'])
    # plot_confusion_matrix(conf_mat_avg, classes=class_names,
    #                       title='Average Confusion matrix,accuracy= %0.2f ' % accuracy_avg)
    return conf_mat_avg, conf_mat_var, accuracy_avg, accuracy_var, mean_fpr, mean_tpr, tprs_lower, tprs_upper, mean_auc, std_auc


X_train, y_train = np.array(images), np.array(class_labels)


############################ RFC ############################ 
clf = RandomForestClassifier(max_depth=8, n_estimators=500, random_state=42)
kfold = StratifiedKFold(n_splits=5, random_state=8)

conf_mat_avg, conf_mat_var, accuracy_avg, accuracy_var, mean_fpr1, mean_tpr1, tprs_lower1, tprs_upper1, mean_auc1, std_auc1 = Classify(X_train, y_train, kfold, clf, "RFC")

rfc_results = [conf_mat_avg, conf_mat_var, accuracy_avg, accuracy_var, mean_fpr1, mean_tpr1, tprs_lower1, tprs_upper1, mean_auc1, std_auc1]
rfc_dict = {"tpr": mean_tpr1, "fpr": mean_fpr1, "auc":mean_auc1, "auc_std":std_auc1 }

np.save("rfc", rfc_dict)
scores = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1':'f1'}
results = cross_validate(clf, X_train, y_train, scoring=scores, cv=kfold)

test_accuracies = results['test_accuracy']
test_f1 = results['test_f1']
test_recall = results['test_recall']
test_precision = results['test_precision']

print("Mean Test Accuracy: %.5f, STD of Accuracy: %.5f"%(np.mean(test_accuracies),np.std(test_accuracies)))
print("Mean Test Precision: %.5f, Mean Test Recall: %.5f "%(np.mean(test_precision), np.std(test_precision)))
print("Mean Test Recall: %.5f, STD of Recall: %.5f "%(np.mean(test_recall), np.std(test_recall)))
print("Mean Test F1 score: %.5f, STD of F1 score: %.5f"%(np.mean(test_f1), np.std(test_f1)))

X_train,X_test,y_train, y_test = train_test_split(np.array(images),np.array(class_labels),test_size=0.8,random_state=42)

clf.fit(X_train,y_train)
pickle.dump(clf, open("rfc.pickle.dat", "wb"))
scores = clf.score(X_test,y_test)
y_pred = clf.predict(X_test)

classification_report = classification_report(y_test, y_pred)

print("Classification Report")
print(classification_report)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(conf_matrix)

y_pred_prob = clf.predict_proba(X_test)
fpr1, tpr1, threshold = roc_curve(y_test, y_pred_prob[:,1], pos_label=1)
roc_auc1 = auc(fpr1, tpr1)


######################### XGBoost ###########################
# X_train, y_train = np.array(images), np.array(class_labels)
#
# model = XGBClassifier(max_depth=8,seed=7)
# kfold = StratifiedKFold(n_splits=5, random_state=42)
#
# Finalconf_mat_avg, conf_mat_var, accuracy_avg, accuracy_var, mean_fpr2, mean_tpr2, tprs_lower2, tprs_upper2, mean_auc2, std_auc2 = Classify(X_train, y_train, kfold, model, "XGBoost")
#
# xgboost_results = [Finalconf_mat_avg, conf_mat_var, accuracy_avg, accuracy_var, mean_fpr2, mean_tpr2, tprs_lower2, tprs_upper2, mean_auc2, std_auc2]
#
# xgboost_dict = {"tpr": mean_tpr2, "fpr": mean_fpr2, "auc":mean_auc2, "auc_std":std_auc2 }
#
# np.save("xgboost_dict", xgboost_dict)
# scores = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1':'f1'}
# results = cross_validate(model, X_train, y_train, scoring=scores, cv=kfold)
#
# test_accuracies = results['test_accuracy']
# test_f1 = results['test_f1']
# test_recall = results['test_recall']
# test_precision = results['test_precision']
#
# print("Mean Test Accuracy: %.5f, STD of Accuracy: %.5f"%(np.mean(test_accuracies),np.std(test_accuracies)))
# print("Mean Test Precision: %.5f, Mean Test Recall: %.5f "%(np.mean(test_precision), np.std(test_precision)))
# print("Mean Test Recall: %.5f, STD of Recall: %.5f "%(np.mean(test_recall), np.std(test_recall)))
# print("Mean Test F1 score: %.5f, STD of F1 score: %.5f"%(np.mean(test_f1), np.std(test_f1)))
#
# X_train,X_test,y_train, y_test = train_test_split(np.array(images),np.array(class_labels),test_size=0.8,random_state=42)
#
# model.fit(np.array(X_train), np.array(y_train))
# y_pred = model.predict(np.array(X_test))
#
# predictions = [round(value) for value in y_pred]
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# #classification_report = classification_report(y_test, predictions)
# #print("Classification Report")
# conf_matrix = confusion_matrix(y_test, predictions)
# print("Confusion Matrix")
# print(conf_matrix)
#
# y_pred_prob = model.predict_proba(X_test)

import matplotlib
matplotlib.rcParams['font.family'] = ['serif']
matplotlib.rcParams['font.serif'] = ['Times New Roman']


mean_fpr = np.linspace(0, 1, 100)
mean_tpr1 = np.ones(shape=(100,1))
mean_tpr1[0]=0.

mean_tpr2 = np.array([0.,    0.984, 0.984, 0.984, 0.984, 0.984, 0.984, 0.984, 0.984, 0.984, 0.988, 0.988,
 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 1.,    1.,    1.,    1.,
 1.,    1.,    1.,    1.,    1. ,   1.,    1.,    1.,    1.,    1.,    1.,    1.,
 1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,
 1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,
 1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,    1. ,   1.,
 1.,    1.,    1.,    1. ,   1.,    1.,    1.,    1.,    1.,    1.,    1.,    1.,
 1.,    1.,    1.,    1.,    1. ,   1.  ,  1. ,   1.,    1.,    1.,    1.,    1.,
 1.,    1.,    1.,    1.   ])

plt.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "Times New Roman"
csfont = {'fontname':'Times New Roman'}
fig = plt.figure()
plt.grid()
markers_on = np.array([0.02, 0.05])

plt.plot(mean_fpr, mean_tpr1, '-g', label = 'RFC' ,linewidth=2,color='green')
plt.plot(mean_fpr, mean_tpr2, '-g', label = 'XGB' ,linewidth=2,color='blue')
plt.axvline(x=0.02, linestyle='--', color='red', linewidth=2)
plt.legend(loc = 'lower right')
# plt.tight_layout()
plt.xlim([0, 0.25])
plt.ylim([0.95, 1.002])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
fig.savefig('bin_roc.eps', format='eps')
# fig.set_size_inches(width, height)
plt.show()

#filename = 'finalized_model.sav'
#joblib.dump(model, filename)

# some time later...
 
# load the model from disk
#filename = 'finalized_model.sav'
#loaded_model = joblib.load(filename)
#result = loaded_model.score(X_test, y_test)
#print(result)

