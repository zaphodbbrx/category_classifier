import re
import numpy as np
from nltk import word_tokenize
from scipy import sparse
import pymystem3
import pickle
import json
from utils.extra_feats import extra_features_extractor

class CategoryClassifier():

    def __init__(self,config_path):
        config = json.load(open(config_path))
        self.exf= extra_features_extractor().fit()
        
        with open(config['vect_path'],'rb') as f:
            self.vect= pickle.load(f)
        with open(config['num2title_path'],'rb') as f:
            self.num2title= pickle.load(f)
        with open(config['model_path'],'rb') as f:
            self.model= pickle.load(f)
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