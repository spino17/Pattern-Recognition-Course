import numpy as np
import pandas as pd
import os
import glob
import errno
from HMM import HMM

def max_class(a, b, c):
        if (a > b):
           val = a
           Class = 1
        else:
           val = b
           Class = 2
        if (val < c):
            val = c
            Class = 3
            
        return Class

# path to the directory of the data files
data_dir = "/home/bhavya/Desktop/machine learning /codes/ass_3/Group07/Test/re/*.mfcc"
files = glob.glob(data_dir)
dataset_1 = np.zeros(39).reshape(1, 39)
for f1 in files:
    data_file = pd.read_csv(f1, sep = " ", header = None)
    datafile = data_file.iloc[:, :-1].values
    dataset_1 = np.append(dataset_1, datafile, axis = 0)
    print(f1)


dataset_1 = np.delete(dataset_1, 0, axis = 0)


data_dir = "/home/bhavya/Desktop/machine learning /codes/ass_3/Group07/Test/ri/*.mfcc"
files = glob.glob(data_dir)
dataset_2 = np.zeros(39).reshape(1, 39)
for f1 in files:
    data_file = pd.read_csv(f1, sep = " ", header = None)
    datafile = data_file.iloc[:, :-1].values
    dataset_2 = np.append(dataset_2, datafile, axis = 0)
    print(f1)


dataset_2 = np.delete(dataset_2, 0, axis = 0)


data_dir = "/home/bhavya/Desktop/machine learning /codes/ass_3/Group07/Test/rI/*.mfcc"
files = glob.glob(data_dir)
dataset_3 = np.zeros(39).reshape(1, 39)
for f1 in files:
    data_file = pd.read_csv(f1, sep = " ", header = None)
    datafile = data_file.iloc[:, :-1].values
    dataset_3 = np.append(dataset_3, datafile, axis = 0)
    print(f1)


dataset_3 = np.delete(dataset_3, 0, axis = 0)


dataset = np.concatenate((dataset_1, dataset_2, dataset_3), axis = 0)

from sklearn.cluster import KMeans
M = 32
kmeans = KMeans(M, tol = 0.002)
kmeans.fit(dataset)

means = kmeans.cluster_centers_
print(means)


# path to the directory of the data files
data_dir = "/home/bhavya/Desktop/machine learning /codes/ass_3/Group07/Test/re/*.mfcc"
files = glob.glob(data_dir)
x1_train = []
for f1 in files:
    data_file = pd.read_csv(f1, sep = " ", header = None)
    datafile = data_file.iloc[:, :-1].values
    pred_seq = kmeans.predict(datafile)
    print(pred_seq)
    pred_seq = pred_seq.tolist()
    x1_train.append(pred_seq)
    print(f1)
    
data_dir = "/home/bhavya/Desktop/machine learning /codes/ass_3/Group07/Test/ri/*.mfcc"
files = glob.glob(data_dir)
x2_train = []
for f1 in files:
    data_file = pd.read_csv(f1, sep = " ", header = None)
    datafile = data_file.iloc[:, :-1].values
    pred_seq = kmeans.predict(datafile)
    print(pred_seq)
    pred_seq = pred_seq.tolist()
    x2_train.append(pred_seq)
    print(f1)
    
data_dir = "/home/bhavya/Desktop/machine learning /codes/ass_3/Group07/Test/rI/*.mfcc"
files = glob.glob(data_dir)
x3_train = []
for f1 in files:
    data_file = pd.read_csv(f1, sep = " ", header = None)
    datafile = data_file.iloc[:, :-1].values
    pred_seq = kmeans.predict(datafile)
    print(pred_seq)
    pred_seq = pred_seq.tolist()
    x3_train.append(pred_seq)
    print(f1)

x1_train = np.array(x1_train)
x2_train = np.array(x2_train)
x3_train = np.array(x3_train)

y1_train = np.full(208, 1)
y2_train = np.full(88, 2)
y3_train = np.full(318, 3)
    
    
np.savetxt('/home/bhavya/Desktop/machine learning /codes/ass_3/Group07/Test/32/x1_test.txt', x1_train, fmt = '%s')
np.savetxt('/home/bhavya/Desktop/machine learning /codes/ass_3/Group07/Test/32/x2_test.txt', x2_train, fmt = '%s')
np.savetxt('/home/bhavya/Desktop/machine learning /codes/ass_3/Group07/Test/32/x3_test.txt', x3_train, fmt = '%s')


x1_train = []
with open('/home/bhavya/Desktop/machine learning /codes/ass_3/Group07/Train/32/x1_train.txt') as f:
   for line in f:
       seq = [int(x) for x in line.split()]
       x1_train.append(seq)

x2_train = []
with open('/home/bhavya/Desktop/machine learning /codes/ass_3/Group07/Train/32/x2_train.txt') as f:
   for line in f:
       seq = [int(x) for x in line.split()]
       x2_train.append(seq)

x3_train = []
with open('/home/bhavya/Desktop/machine learning /codes/ass_3/Group07/Train/32/x3_train.txt') as f:
   for line in f:
       seq = [int(x) for x in line.split()]
       x3_train.append(seq)

x1_train = np.array(x1_train)
x2_train = np.array(x2_train)
x3_train = np.array(x3_train)

y1_train = np.full(208, 1)
y2_train = np.full(88, 2)
y3_train = np.full(318, 3)

x1_test = []
with open('/home/bhavya/Desktop/machine learning /codes/ass_3/Group07/Test/32/x1_test.txt') as f:
   for line in f:
       seq = [int(x) for x in line.split()]
       x1_test.append(seq)

x2_test = []
with open('/home/bhavya/Desktop/machine learning /codes/ass_3/Group07/Test/32/x2_test.txt') as f:
   for line in f:
       seq = [int(x) for x in line.split()]
       x2_test.append(seq)

x3_test = []
with open('/home/bhavya/Desktop/machine learning /codes/ass_3/Group07/Test/32/x3_test.txt') as f:
   for line in f:
       seq = [int(x) for x in line.split()]
       x3_test.append(seq)

x1_test = np.array(x1_test)
x2_test = np.array(x2_test)
x3_test = np.array(x3_test)

y1_test = np.full(52, 1)
y2_test = np.full(22, 2)
y3_test = np.full(80, 3)


# dataset is imported 

# testing for the M = 8
hmm_1= HMM(5, 32, x1_train)
hmm_1.fit(0.002)
print(hmm_1.A)
print(hmm_1.B)
print(hmm_1.Pi)

hmm_2 = HMM(5, 32, x2_train)
hmm_2.fit(0.002)
print(hmm_1.A)
print(hmm_1.B)
print(hmm_1.Pi)

hmm_3 = HMM(5, 32, x3_train)
hmm_3.fit(0.002)


from sklearn.metrics import confusion_matrix
y_true = np.concatenate((y1_test, y2_test, y3_test), axis = 0)
y_pred_1 = []
for x_vec in x1_test:
    score_1 = hmm_1.score_alpha(x_vec)
    score_2 = hmm_2.score_alpha(x_vec)
    score_3 = hmm_3.score_alpha(x_vec)
    pred_class = max_class(score_1, score_2, score_3)
    y_pred_1.append(pred_class)
    
y_pred_1 = np.array(y_pred_1)

y_pred_2 = []
for x_vec in x2_test:
    score_1 = hmm_1.score_alpha(x_vec)
    score_2 = hmm_2.score_alpha(x_vec)
    score_3 = hmm_3.score_alpha(x_vec)
    pred_class = max_class(score_1, score_2, score_3)
    y_pred_2.append(pred_class)
    
y_pred_2 = np.array(y_pred_2)

y_pred_3 = []
for x_vec in x3_test:
    score_1 = hmm_1.score_alpha(x_vec)
    score_2 = hmm_2.score_alpha(x_vec)
    score_3 = hmm_3.score_alpha(x_vec)
    pred_class = max_class(score_1, score_2, score_3)
    y_pred_3.append(pred_class)
    
y_pred_3 = np.array(y_pred_3)

y_pred = np.concatenate((y_pred_1, y_pred_2, y_pred_3), axis = 0)

confusion_matrix(y_true, y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
print(accuracy)