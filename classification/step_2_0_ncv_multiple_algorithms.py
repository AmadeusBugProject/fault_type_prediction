import os
import pathlib

from classification.defaults import make_default_pipeline, make_default_gridsearch_parameters, \
    make_classifier_specific_gridsearch_parameters
from classification.default_classifiers import classifiers
from classification.utils import load_dataset, get_classifier_by_name, nested_cross_validation, \
    get_best_classifier, bootstrap, clf_short_name
from file_anchor import root_dir
from helpers.Logger import Logger

log = Logger()
output_path = root_dir() + 'classification/output/' + pathlib.Path(__file__).stem + '/'
os.makedirs(output_path, exist_ok=True)


def main():
    docs, target, target_names = load_dataset()
    for clf_name in classifiers:
        gs_params = make_default_gridsearch_parameters()
        gs_params.update(make_classifier_specific_gridsearch_parameters(clf_name))
        pipeline = make_default_pipeline(get_classifier_by_name(clf_name))

        ncv_results = nested_cross_validation(docs, target, target_names, output_path + clf_short_name(clf_name), pipeline, gs_params, scoring='f1_macro')
        best_clf, params = get_best_classifier(ncv_results, 'F1 macro average', make_default_pipeline)
        bootstrap(docs, target, target_names, best_clf, params, output_path + clf_short_name(clf_name))


if __name__ == "__main__":
    main()
