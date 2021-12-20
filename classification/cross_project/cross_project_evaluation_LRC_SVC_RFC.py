import json
import os
import pathlib

import pandas
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer

from classification.defaults import make_default_pipeline, make_default_gridsearch_parameters, \
    make_classifier_specific_gridsearch_parameters
from classification.default_classifiers import classifiers
from classification.utils import load_dataset, get_classifier_by_name, nested_cross_validation, \
    clf_short_name, \
    class_f1_scorer, unpandas_parameters, report_classifier_performance, get_best_classifier
from file_anchor import root_dir
from helpers.Logger import Logger

output_path = root_dir() + 'classification/cross_project/output/' + pathlib.Path(__file__).stem + '/'
os.makedirs(output_path, exist_ok=True)
log = Logger(log_file=output_path + 'log.txt')


def main():
    with open(root_dir() + 'classification/cross_project/output/project_selection/project_combos.json', 'r') as fd:
        project_combos = json.load(fd)

    for i, projects_split in enumerate(project_combos):
        log.s('at ' + str(i) + ' ' + str(projects_split))
        projects = projects_split['projects']
        eval_project_split(projects)


def eval_project_split(projects):
    train_docs, train_target, target_names = load_dataset(projects_list=projects, pick='ProjectsNotInList')
    validation_docs, validation_target, target_names = load_dataset(projects_list=projects, pick='ProjectsFromList')

    classifiers_df = pandas.DataFrame()

    for clf_name in classifiers:
        gs_params = make_default_gridsearch_parameters()
        gs_params.update(make_classifier_specific_gridsearch_parameters(clf_name))
        pipeline = make_default_pipeline(get_classifier_by_name(clf_name))

        ncv_results = nested_cross_validation(train_docs, train_target, target_names, None, pipeline, gs_params, scoring='f1_macro')
        best_clf, params = get_best_classifier(ncv_results, 'F1 macro average', make_default_pipeline)

        pipeline.fit(train_docs, train_target)
        y_predicted = pipeline.predict(validation_docs)

        test_set_prediction_df = pandas.DataFrame({'doc': validation_docs, 'target': validation_target, 'prediction': y_predicted})
        test_set_prediction_df.to_csv(output_path + 'predictions_' + clf_name + '_'.join(projects) + '.csv.zip', compression='zip')

        report = report_classifier_performance(validation_target, y_predicted, target_names, 0, params, clf_name)
        classifiers_df = classifiers_df.append(pandas.DataFrame(report))

    classifiers_df.to_csv(output_path + 'performance' + '_'.join(projects) + '.csv')


if __name__ == "__main__":
    main()
