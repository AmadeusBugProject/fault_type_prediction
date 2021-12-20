import os
import pathlib

import pandas
from sklearn.metrics import make_scorer

from classification.defaults import make_default_pipeline, make_default_gridsearch_parameters, \
    make_classifier_specific_gridsearch_parameters
from classification.default_classifiers import classifiers
from classification.utils import load_dataset, get_classifier_by_name, nested_cross_validation, \
    clf_short_name, \
    class_f1_scorer
from file_anchor import root_dir
from helpers.Logger import Logger

log = Logger()
output_path = root_dir() + 'classification/output/' + pathlib.Path(__file__).stem + '/'
os.makedirs(output_path, exist_ok=True)


def main():
    docs, target, target_names = load_dataset()
    for bug_cat in target_names:
        bug_cat_report_df = pandas.DataFrame()

        for clf_name in classifiers:
            gs_params = make_default_gridsearch_parameters()
            gs_params.update(make_classifier_specific_gridsearch_parameters(clf_name))

            if clf_name != 'MultinomialNB(fit_prior=False)':
                gs_params.update({'clf__class_weight': [{target_names.index(bug_cat): 4}]})
            pipeline = make_default_pipeline(get_classifier_by_name(clf_name))
            scorer = make_scorer(class_f1_scorer, class_idx=target_names.index(bug_cat))
            ncv_results = nested_cross_validation(docs, target, target_names, output_path + bug_cat + '_' + clf_short_name(clf_name), pipeline, gs_params, scoring=scorer)
            bug_cat_report_df = bug_cat_report_df.append(ncv_results)
        bug_cat_report_df.to_csv(output_path + bug_cat + '_ncv_classifiers.csv')


if __name__ == "__main__":
    main()
