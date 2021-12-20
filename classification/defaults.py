from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from artifact_detection_model.transformer.ArtifactRemoverTransformer import ArtifactRemoverTransformer, DO_NOT_REPLACE, \
    KEEP_EXCEPTION_NAMES
from preprocessing.DeCamelCaser import DeCamelCaser
from preprocessing.StemmedCountVectorizer import StemmedCountVectorizer


def make_classifier_specific_gridsearch_parameters(clf_name):
    return {'MultinomialNB(fit_prior=False)': {
                                # 'clf__alpha': [0.1, 0.5, 1],
                            },
                      'LinearSVC(random_state=42)': {
                                # 'clf__loss': ('hinge', 'squared_hinge'),
                                # 'clf__C': [0.1, 0.5, 1]
                                },
                      'RandomForestClassifier(random_state=42)': {
                                # 'clf__criterion': ["gini"]
                            },
                      'LogisticRegression(random_state=42)': {
                                # 'clf__solver': ['liblinear', 'newton-cg'],
                                # 'clf__C': [1, 4, 10]
                            }
                      }[clf_name]


def make_default_gridsearch_parameters():
    return {
        'vect__stop_words': [None, 'english'],
        'vect__stemming': [True, False],
        'artifactspred__replacement_strategy': [DO_NOT_REPLACE, KEEP_EXCEPTION_NAMES],
        'decamelcase__decamlecase': [True, False],
        # 'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': [True, False],
    }


def make_default_pipeline(clf):
    return Pipeline([('artifactspred', ArtifactRemoverTransformer()),
                         ('decamelcase', DeCamelCaser()),
                         ('vect', StemmedCountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', clf)])
