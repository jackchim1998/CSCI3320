import pandas as pd
from os.path import join
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import math
import numpy as np
warnings.simplefilter("ignore")

def classifier(feature_value,prior,estimated_p):
    log_posterior = [0.0] * len(prior)
    for i in range(len(log_posterior)):
        log_posterior[i] = feature_value * math.log(estimated_p[i]) + (1 - feature_value) * math.log(1 - estimated_p[i]) + math.log(prior[i])
    return log_posterior.index(max(log_posterior))+1
# read csv file
data = pd.read_csv("input_1.csv")
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


#find estimated p
estimated_p = [0] * class_no
print("-----------------estimated p--------------------")
tmp_cnt=0
for index, row in X_train.iterrows():
    if row["feature_value"] == 1:
        estimated_p[y_train[tmp_cnt]-1]+=1
    tmp_cnt+=1
for i in range(class_no):
    estimated_p[i] = estimated_p[i] / cnt_class[i]
    print("class "+str(i+1)+" is "+str(estimated_p[i]))


#classify test_set and implement confustion matrix
print("----------------classify test set----------------")
classify_output_test = [0] * len(y_test)
for index, row in X_test.iterrows():
    classify_output_test[index - len(y_train)] = classifier(row["feature_value"],prior,estimated_p)


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
