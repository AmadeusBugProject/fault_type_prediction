import re

from sklearn.base import BaseEstimator, TransformerMixin


class DeCamelCaser(BaseEstimator, TransformerMixin):
    def __init__(self, decamlecase=True):
        self.decamlecase = decamlecase

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.decamlecase:
            return X
        return [self.split_camel_cased(i) for i in X]

    def split_camel_cased(self, text):
        for rec in rec_map:
            text = rec[0].sub(rec[1], text)
        return text

rec_map = [
    (re.compile(r"(?<=[a-z0-9])([A-Z])"), r" \1"),
]



