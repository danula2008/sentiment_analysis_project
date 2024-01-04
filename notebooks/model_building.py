import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
import string
from nltk import PorterStemmer

from collections import Counter

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import pickle

data = pd.read_csv("artifacts\sentiment_analysis.csv")

#make all lowercase
data["tweet"] = data["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))

#remove links
data["tweet"] = data['tweet'].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))

#remove punctuations
data["tweet"] = data["tweet"].apply(lambda text: ''.join([char for char in text if char not in string.punctuation]))

#remove numbers
data["tweet"] = data['tweet'].str.replace('\d+', '', regex=True)

with open("static\model\stopwords\english", 'r') as file:
    sw = file.read().splitlines()

#remove stopwords
data["tweet"] = data["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

ps = PorterStemmer()

data["tweet"] = data["tweet"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))

vocab = Counter()

for sentence in data["tweet"]:
    vocab.update(sentence.split())

tokens = [key for key in vocab if vocab[key] > 10]

def save_vocab(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w', encoding='utf-8')
    file.write(data)
    file.close()

save_vocab(tokens, 'static\\model\\vocabulaty.txt')

x = data['tweet']
y = data['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

def vectorizer(ds, vocabulary):
    vectorized_lst = []
    
    for sentence in ds:
        sentence_lst = np.zeros(len(vocabulary))
        
        for i in range(len(vocabulary)):
            if vocabulary[i] in sentence.split():
                sentence_lst[i] = 1
                
        vectorized_lst.append(sentence_lst)
        
    vectorized_lst_new = np.asarray(vectorized_lst, dtype=np.float32)
    
    return vectorized_lst_new

vectorized_x_train = vectorizer(x_train, tokens)

vectorized_x_test = vectorizer(x_test, tokens)

# plt.pie(np.array([y_train.value_counts()[0], y_train.value_counts()[1]]), labels=['Positive', 'Negative'])
# plt.show()

smote = SMOTE()
vectorized_x_train_smote, y_train_smote = smote.fit_resample(vectorized_x_train, y_train)
# print(vectorized_x_train_smote.shape, y_train_smote.shape)


def training_scores(y_act, y_pred):
    acc = round(accuracy_score(y_act, y_pred), 3)
    pr = round(precision_score(y_act, y_pred), 3)
    rec = round(recall_score(y_act, y_pred), 3)
    f1 = round(f1_score(y_act, y_pred), 3)
    print(f'Training Scores:\n\tAccuracy = {acc}\n\tPrecision = {pr}\n\tRecall = {rec}\n\tF1-Score = {f1}')
    
def validation_scores(y_act, y_pred):
    acc = round(accuracy_score(y_act, y_pred), 3)
    pr = round(precision_score(y_act, y_pred), 3)
    rec = round(recall_score(y_act, y_pred), 3)
    f1 = round(f1_score(y_act, y_pred), 3)
    print(f'Testing Scores:\n\tAccuracy = {acc}\n\tPrecision = {pr}\n\tRecall = {rec}\n\tF1-Score = {f1}')

#Logistic Regression
lr = LogisticRegression()
lr.fit(vectorized_x_train_smote, y_train_smote)

y_train_pred = lr.predict(vectorized_x_train_smote)
y_test_pred = lr.predict(vectorized_x_test)

# print(training_scores(y_train_smote, y_train_pred))
# print("------------------------------")
# print(validation_scores(y_test, y_test_pred))

#Naive Bayes
# mnb = MultinomialNB()
# mnb.fit(vectorized_x_train_smote, y_train_smote)

# y_train_pred = mnb.predict(vectorized_x_train_smote)
# y_test_pred = mnb.predict(vectorized_x_test)

# print(training_scores(y_train_smote, y_train_pred))
# print("------------------------------")
# print(validation_scores(y_test, y_test_pred))

#Decision Tree
# dt = DecisionTreeClassifier()
# dt.fit(vectorized_x_train_smote, y_train_smote)

# y_train_pred = dt.predict(vectorized_x_train_smote)
# y_test_pred = dt.predict(vectorized_x_test)

# print(training_scores(y_train_smote, y_train_pred))
# print("------------------------------")
# print(validation_scores(y_test, y_test_pred))

#Random Forest
# rf = RandomForestClassifier()
# rf.fit(vectorized_x_train_smote, y_train_smote)

# y_train_pred = rf.predict(vectorized_x_train_smote)
# y_test_pred = rf.predict(vectorized_x_test)

# print(training_scores(y_train_smote, y_train_pred))
# print("------------------------------")
# print(validation_scores(y_test, y_test_pred))

#Support Vector Machine(Support Vector Classifier)
# svm = SVC()
# svm.fit(vectorized_x_train_smote, y_train_smote)

# y_train_pred = svm.predict(vectorized_x_train_smote)
# y_test_pred = svm.predict(vectorized_x_test)

# print(training_scores(y_train_smote, y_train_pred))
# print("------------------------------")
# print(validation_scores(y_test, y_test_pred))

with open('static\model\model.pickle', 'wb') as file:
    pickle.dump(lr, file)