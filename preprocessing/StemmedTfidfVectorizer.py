import nltk
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english", ignore_stopwords=True)


class StemmedTfidfVectorizer(TfidfVectorizer):
    def __init__(self, stemming=True, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=numpy.int64):
        self.stemming = stemming
        super().__init__(input=input, encoding=encoding,
                         decode_error=decode_error, strip_accents=strip_accents,
                         lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer,
                         stop_words=stop_words, token_pattern=token_pattern,
                         ngram_range=ngram_range, analyzer=analyzer,
                         max_df=max_df, min_df=min_df, max_features=max_features,
                         vocabulary=vocabulary, binary=binary, dtype=dtype)

    def build_analyzer(self):
        if self.stemming:
            analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
            return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
        else:
            return super(StemmedTfidfVectorizer, self).build_analyzer()
