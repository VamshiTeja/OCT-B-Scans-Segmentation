# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-04-25 12:43:38
# @Last Modified by:   vamshi
# @Last Modified time: 2019-01-23 00:55:59

import os,sys
import numpy as np
import cv2
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

from itertools import cycle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from sklearn.preprocessing import label_binarize
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import classification_report

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
import sklearn.metrics as metrics

random.seed(0)

train_dir = "./train/"
train_labels_dir = "./train_labels/"

images  = []
class_labels = []

print "loading data"

for file in os.listdir(train_dir):
	#print(file)
	img = cv2.imread(os.path.join(train_dir,file),0)
	#print(img)
	img = img.flatten()
	#print(img)
	img_label = cv2.imread(os.path.join(train_labels_dir,file[:-4]+".png"),1)

	images.append(img)
	b,g,r = cv2.split(img_label)
	#print(np.sum(b))

	if(np.sum(r)!=0 and np.sum(g)!=0):
		#print 2
		class_labels.append(2)
	elif(np.sum(r)!=0 and np.sum(g)==0 ):
		#print 0
		class_labels.append(0)
	elif(np.sum(r)==0 and np.sum(g)!=0):
		#print 1
		class_labels.append(1)
	elif(np.sum(r)==0 and np.sum(g)==0):
		#print 3
		class_labels.append(3)


print("Done")
print "Splitting data"

X_train,X_test,y_train, y_test = train_test_split(images,class_labels,test_size=0.2,random_state=41)
print np.array(X_train).shape

'''
Images = './dataset.pickle'

try:
    f = open(Images,'wb')
    save = {
    'X_train'  : X_train,
    'X_test'   : X_test,
    'y_train'   : y_train,
    'y_test'    : y_test,
    }
    pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
    f.close() 
except Exception as e:
    print('Unable to save data to ',Images,':',e)
    raise

statinfo = os.stat(Images)
print('Compressed pickle size:',statinfo.st_size)
'''

######################## RFC #########################

clf = RandomForestClassifier(max_depth=7, n_estimators=500,random_state=0)
kfold = StratifiedKFold(n_splits=5, random_state=8)
scores = {'accuracy': 'accuracy',  'f1':'f1_macro'}
results = cross_validate(clf, X_train, y_train, scoring=scores, cv=kfold)

test_accuracies = results['test_accuracy']
test_f1 = results['test_f1']
# test_recall = results['test_recall']
# test_precision = results['test_precision']

print("Mean Test Accuracy: %.5f, STD of Accuracy: %.5f"%(np.mean(test_accuracies),np.std(test_accuracies)))
# print("Mean Test Precision: %.5f, Mean Test Recall: %.5f "%(np.mean(test_precision), np.std(test_precision)))
# print("Mean Test Recall: %.5f, STD of Recall: %.5f "%(np.mean(test_recall), np.std(test_recall)))
print("Mean Test F1 score: %.5f, STD of F1 score: %.5f"%(np.mean(test_f1), np.std(test_f1)))


clf.fit(X_train,y_train)
pickle.dump(clf, open("rfc.pickle.dat", "wb"))
scores = clf.score(X_test,y_test)
print "cross validation -- 5"
scores = cross_val_score(clf, images,class_labels,cv=5)
print scores
print("mean: %.5f, var: %.5f"%(np.mean(scores),np.var(scores)))

y_pred = clf.predict(X_test)

y_pred_proba = clf.predict_proba(X_test)
classification_report = classification_report(y_test, y_pred)
print classification_report
conf_matrix = confusion_matrix(y_test, y_pred)
print conf_matrix


y_test = label_binarize(y_test, classes=[0, 1, 2, 3])
y_pred = label_binarize(y_pred, classes=[0, 1, 2, 3])
n_classes = y_test.shape[1]

fpr1 = dict()
tpr1 = dict()
roc_auc1 = dict()
for i in range(n_classes):
    fpr1[i], tpr1[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
    roc_auc1[i] = auc(fpr1[i], tpr1[i])

# Compute micro-average ROC curve and macro-average ROC
fpr1["micro"], tpr1["micro"], _ = roc_curve(y_test.ravel(), y_pred_proba.ravel())
roc_auc1["micro"] = auc(fpr1["micro"], tpr1["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr1[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr1[i], tpr1[i])
mean_tpr /= n_classes



#colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#for i, color in zip(range(n_classes), colors):
#    plt.plot(fpr1[i], tpr1[i], color=color, lw=lw,
#             label='ROC curve of class {0} (area = {1:0.2f})'
#             ''.format(i, roc_auc1[i]))

########################## XGBoost ###############################


model = XGBClassifier()

kfold = StratifiedKFold(n_splits=5, random_state=42)
scores = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1':'f1'}
results = cross_validate(model, X_train, y_train, scoring=scores, cv=kfold)

test_accuracies = results['test_accuracy']
test_f1 = results['test_f1']
test_recall = results['test_recall']
test_precision = results['test_precision']

print("Mean Test Accuracy: %.5f, STD of Accuracy: %.5f"%(np.mean(test_accuracies),np.std(test_accuracies)))
print("Mean Test Precision: %.5f, Mean Test Recall: %.5f "%(np.mean(test_precision), np.std(test_precision)))
print("Mean Test Recall: %.5f, STD of Recall: %.5f "%(np.mean(test_recall), np.std(test_recall)))
print("Mean Test F1 score: %.5f, STD of F1 score: %.5f"%(np.mean(test_f1), np.std(test_f1)))

model.fit(np.array(X_train), np.array(y_train))
y_pred = model.predict(np.array(X_test))
predictions = [round(value) for value in y_pred]
#accuracy = accuracy_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))
#conf_matrix = confusion_matrix(y_test, y_pred)
#print conf_matrix

y_pred_proba = model.predict_proba(X_test)


fpr2 = dict()
tpr2 = dict()
roc_auc2 = dict()
for i in range(n_classes):
    fpr2[i], tpr2[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
    roc_auc2[i] = auc(fpr2[i], tpr2[i])

# Compute micro-average ROC curve and macro-average ROC
fpr2["micro"], tpr2["micro"], _ = roc_curve(y_test.ravel(), y_pred_proba.ravel())
roc_auc2["micro"] = auc(fpr2["micro"], tpr2["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr2[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr2[i], tpr2[i])
mean_tpr /= n_classes

fpr2["macro"] = all_fpr
tpr2["macro"] = mean_tpr
roc_auc2["macro"] = auc(fpr2["macro"], tpr2["macro"])

# Plot all ROC curves


fpr1["macro"] = all_fpr
tpr1["macro"] = mean_tpr
roc_auc1["macro"] = auc(fpr1["macro"], tpr1["macro"])

# Plot all ROC curves
fig1 = plt.figure()
plt.plot(fpr1["micro"], tpr1["micro"],
         label='micro-average ROC curve of Random Forest(area = {0:0.2f})'
               ''.format(roc_auc1["micro"]),
         color='navy', linewidth=4)

plt.plot(fpr2["micro"], tpr2["micro"],
         label='micro-average ROC curve of XGBoost(area = {0:0.2f})'
               ''.format(roc_auc2["micro"]),
         color='deeppink', linewidth=4)

lw=2
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC(Micro-Averaged) for Multi-Class')
plt.legend(loc="lower right")
fig1.savefig('micro_roc.eps', format='eps', dpi=fig1.dpi)
plt.show()


fig2 = plt.figure()
plt.plot(fpr1["macro"], tpr1["macro"],
         label='macro-average ROC curve of Random Forest(area = {0:0.2f})'
               ''.format(roc_auc1["macro"]),
         color='navy', linewidth=4)

plt.plot(fpr2["macro"], tpr2["macro"],
         label='macro-average ROC curve of XGBoost(area = {0:0.2f})'
               ''.format(roc_auc2["macro"]),
         color='deeppink', linewidth=4)

lw=2
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC(Macro Averaged) for Binary Classification')
plt.legend(loc="lower right")
fig2.savefig('macro_roc.eps', format='eps', dpi=fig.dpi)
plt.show()


#filename = 'finalized_model.sav'
#joblib.dump(model, filename)

# some time later...
 
# load the model from disk
#filename = 'finalized_model.sav'
#loaded_model = joblib.load(filename)
#result = loaded_model.score(X_test, y_test)
#print(result)

