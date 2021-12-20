import json
import os
import os
import pathlib

import numpy as np
import pandas
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from classification.defaults import make_default_pipeline
from classification.utils import load_dataset, get_classifier_by_name, bootstrap, plot_numpy_confusion_matrix, \
    report_classifier_performance, \
    unpandas_parameters
from file_anchor import root_dir
from helpers.Logger import Logger

log = Logger()
output_path = root_dir() + 'classification/output/' + pathlib.Path(__file__).stem + '/'
os.makedirs(output_path, exist_ok=True)


def main():
    for num_classifiers_per_cat in [1, 2, 5]:
        run_ensemble(num_classifiers_per_cat)


def run_ensemble(num_classifiers_per_cat):
    docs, targets, target_names = load_dataset()
    classifiers_df = select_best_classifiers(num_classifiers_per_cat, target_names)
    estimators = []
    for split, target, parameters, classifier in zip(classifiers_df['split_id'], classifiers_df['target'], classifiers_df['params'], classifiers_df['classifier']):
        name = target + str(split) + classifier
        pipeline = make_default_pipeline(get_classifier_by_name(classifier))
        pipeline.set_params(**unpandas_parameters(parameters))
        estimators.append((name, pipeline))

    ensemble_name = 'EnsembleTop' + str(num_classifiers_per_cat)
    final_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    final_params = {}
    classifiers_df.to_csv(output_path + ensemble_name + 'parameters.csv')
    bootstrap(docs, targets, target_names, final_clf, final_params, output_path + ensemble_name, ensemble_name)
    cross_validation_for_confusion_matrix(docs, targets, target_names, output_path, final_clf, ensemble_name)


def select_best_classifiers(num_classifiers_per_cat, target_names):
    classifiers_df = pandas.DataFrame()
    for target in target_names:
        df = pandas.read_csv(root_dir() +
            'classification/output/step_3_ncv_multiple_algorithms_weighted_classes/' + target + '_ncv_classifiers.csv')
        df = df.sort_values('F1_' + target)
        df['target'] = target
        classifiers_df = classifiers_df.append(df.tail(num_classifiers_per_cat))

    return classifiers_df


def cross_validation_for_confusion_matrix(docs, target, target_names, output_path, pipeline, ensemble_name, scoring='f1_macro'):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    report_df = pandas.DataFrame()

    conf_matrix = np.zeros((len(target_names), len(target_names)))
    for i, (train_idx, test_idx) in enumerate(outer_cv.split(docs, target)):
        train_x = docs[train_idx]
        train_y = target[train_idx]

        test_x = docs[test_idx]
        test_y = target[test_idx]

        pipeline.fit(train_x, train_y)
        y_predicted = pipeline.predict(test_x)

        single_report = report_classifier_performance(test_y, y_predicted, target_names, i, {}, ensemble_name)
        conf_matrix = conf_matrix + np.array(json.loads(single_report['CM']))

        report = pandas.DataFrame(single_report)
        report_df = report_df.append(report)

    plot_numpy_confusion_matrix(conf_matrix, target_names, output_path + ensemble_name + '_confusion_matrix_combined.png')
    report_df.to_csv(output_path + ensemble_name + '_cross_validation.csv')
    return report_df


if __name__ == "__main__":
    main()
