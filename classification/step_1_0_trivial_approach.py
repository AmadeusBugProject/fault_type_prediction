import os
import pathlib

import pandas
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from classification.defaults import make_default_gridsearch_parameters, make_classifier_specific_gridsearch_parameters
from classification.utils import load_dataset, get_best_classifier, bootstrap, nested_cross_validation, \
    get_classifier_by_name
from file_anchor import root_dir
from helpers.Logger import Logger
from preprocessing.DeCamelCaser import DeCamelCaser
from preprocessing.StemmedCountVectorizer import StemmedCountVectorizer

log = Logger()
output_path = root_dir() + 'classification/output/' + pathlib.Path(__file__).stem + '/'
os.makedirs(output_path, exist_ok=True)


def make_simple_pipeline(clf):
    return Pipeline([('decamelcase', DeCamelCaser()),
                     ('vect', StemmedCountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', clf)])


def main():
    docs, target, target_names = load_dataset()
    clf_name = 'LinearSVC(random_state=42)'
    gs_params = make_default_gridsearch_parameters()
    gs_params.update(make_classifier_specific_gridsearch_parameters(clf_name))
    gs_params.pop('artifactspred__replacement_strategy')

    pipeline = make_simple_pipeline(get_classifier_by_name(clf_name))

    ncv_results = nested_cross_validation(docs, target, target_names, output_path + 'LinearSVC', pipeline, gs_params, scoring='f1_macro')
    best_clf, params = get_best_classifier(ncv_results, 'F1 macro average', make_simple_pipeline)
    bootstrap(docs, target, target_names, best_clf, params, output_path + 'LinearSVC')


if __name__ == "__main__":
    main()
