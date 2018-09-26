import re
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score,f1_score, recall_score
from IPython.display import clear_output
from nltk import word_tokenize
from scipy import sparse
import pymystem3
import pickle
import json
from utils.extra_feats import extra_features_extractor

class CategoryClassifier():

    def __init__(self,config_path):
        config = json.load(open(config_path))
        with open(config['exf_path'],'rb') as f:
            self.exf= pickle.load(f)
        with open(config['vect_path'],'rb') as f:
            self.vect= pickle.load(f)
        with open(config['num2title_path'],'rb') as f:
            self.num2title= pickle.load(f)
        with open(config['model_path'],'rb') as f:
            self.model= pickle.load(f)
        self.decode_dict = {
            1.0: 'scam',
            0.0: 'no scam'
        }
        self.mystem = pymystem3.Mystem()

    def __clean_message(self,text):
        text = re.sub('\W', ' ', str(text)).lower()
        tokens = word_tokenize(text)
        tokens = [self.mystem.lemmatize(t)[0] for t in tokens]
        return ' '.join(tokens)


    def run(self,text):
        text_cleaned = self.__clean_message(text)
        extra_feats = self.exf.transform([text])
        bow_feats = self.vect.transform([text_cleaned])
        feats = sparse.csc_matrix(np.hstack([bow_feats.toarray(), extra_feats]))
        decision = self.num2title[self.model.predict(feats)[0]]
        confidence = self.model.predict_proba(feats)
        return {
            'decision':decision,
            'confidence': np.max(confidence)
        }
if __name__ == '__main__':
    sd = CategoryClassifier('config.json')
    while True:
        mes = input()
        if mes == 'q':
            break
        print(sd.run(mes))