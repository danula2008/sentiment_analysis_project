import pickle
import numpy as np
import pandas as pd
import re
import string

from nltk.stem import PorterStemmer

with open('static\model\model.pickle', 'rb') as file:
    model = pickle.load(file)


def preprocessing(txt):

    data = pd.DataFrame([txt], columns=["tweet"])

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


    #steming
    ps = PorterStemmer()
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))

    return data["tweet"]

vocab = pd.read_csv('static/model/vocabulaty.txt', header=None)
tokens = vocab[0].tolist()

def vectorizer(ds):
    vectorized_lst = []
    
    for sentence in ds:
        sentence_lst = np.zeros(len(tokens))
        
        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_lst[i] = 1
                
        vectorized_lst.append(sentence_lst)
        
    vectorized_lst_new = np.asarray(vectorized_lst, dtype=np.float32)
    
    return vectorized_lst_new


def get_prediction(vectorized_text):
    prediction = model.predict(vectorized_text)
    if prediction == 1:
        return 'negative'
    else:
        return 'positive'


