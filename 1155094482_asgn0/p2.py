import pandas as pd
from os.path import join
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import math
import numpy as np
warnings.simplefilter("ignore")

def classifier(feature_value, prior, estimated_m, estimated_variance):
    log_posterior = [0.0] * len(prior)
    for i in range(len(log_posterior)):
        log_posterior[i] = -0.5 * math.log(2 * math.pi) - math.log(estimated_variance[i] ** 0.5) - ((feature_value - estimated_m[i]) ** 2) / (2*estimated_variance[i]) + math.log(prior[i])
    return log_posterior.index(max(log_posterior))+1


# read csv file
data = pd.read_csv("input_2.csv")
X = data.drop("class" , axis=1)
y= data.loc[:,'class']
print("----------------Before process------------------")
print("class_type:")
print(y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


class_no = 3
#find prior
cnt_class = [0] * class_no
prior = [0] * class_no
print("-------------------prior value-------------------")
row_count=len(y_train)
for row in y_train:
    cnt_class[row-1]+=1
for i in range(class_no):
    prior[i]= cnt_class[i]/row_count
    print("class "+str(i+1)+" is "+str(prior[i]))


#find estimated m
estimated_m = [0] * class_no
print("-----------------estimated m--------------------")
tmp_cnt=0
for index, row in X_train.iterrows():
    estimated_m[y_train[tmp_cnt]-1]+=row["feature_value"]
    tmp_cnt+=1
for i in range(class_no):
    estimated_m[i] = estimated_m[i] / cnt_class[i]
    print("class "+str(i+1)+" is "+str(estimated_m[i]))


#find estimated variance
estimated_variance = [0] * class_no
print("----------------estimated variance----------------")
tmp_cnt=0
for index, row in X_train.iterrows():
    estimated_variance[y_train[tmp_cnt]-1]+=(row["feature_value"] - estimated_m[y_train[tmp_cnt]-1]) ** 2
    tmp_cnt+=1
for i in range(class_no):
    estimated_variance[i] = estimated_variance[i] / cnt_class[i]
    print("class "+str(i+1)+" is "+str(estimated_variance[i]))


#classify test_set and implement confustion matrix
print("----------------classify test set----------------")
classify_output_test = [0] * len(y_test)
tmp_cnt=0
for index, row in X_test.iterrows():
    classify_output_test[tmp_cnt] = classifier(row["feature_value"],prior,estimated_m,estimated_variance)
    tmp_cnt+=1


print("----------------confusion matrix(predict/actual)----------------")
labels = [1,2,3]
cm = confusion_matrix(y_test, classify_output_test, labels=labels)
print(pd.DataFrame(cm,index=labels,columns=labels),sep=" ")

print("---------------precision-------------------")
precision = np.diag(cm) / np.sum(cm,axis=0)
print(pd.Series(precision, index=labels))

print("----------------recall---------------------")
recall = np.diag(cm) / np.sum(cm,axis=1)
print(pd.Series(recall, index=labels))

print("---------------accuracy--------------------")
print(np.sum(np.diag(cm)) / np.sum(cm))

print("---------------f1_score--------------------")
seperate_f1_score = f1_score(y_test, classify_output_test, labels=labels, average=None)
print(pd.Series(seperate_f1_score,index=labels))

print("-------------unweighted average f1_score----------------")
print(np.average(seperate_f1_score))
