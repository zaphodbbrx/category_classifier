from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
import re

class extra_features_extractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.extractors = [
            #   natasha.AddressExtractor(),
            #   natasha.LocationExtractor(),
            #   natasha.DatesExtractor(),
            #   natasha.NamesExtractor(),
            #   natasha.MoneyExtractor(),
        ]
        pass

    def __get_extra_feats(self, text):
        def extract_things(text, extractor):
            return len(extractor(text))

        res = [extract_things(text, e) for e in self.extractors]
        res += [len(re.findall(r'[\w\.-]+@[\w\.-]+', text)),
                len(re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', text)),
                len(re.findall(r'[\w]+', text)),
                len(re.findall(r'[\d]+', text)),
                len(re.findall(r'[.]+', text))
                ]
        return res

    def transform(self, X, y=None, **fit_params):
        return [self.__get_extra_feats(line) for line in X]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X=None, y=None, **fit_params):
        return self