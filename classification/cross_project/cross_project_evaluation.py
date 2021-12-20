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
    class_f1_scorer, unpandas_parameters, report_classifier_performance
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

    nested_cross_val_classifiers = nested_cross_validation_for_model_selection(projects, train_docs, train_target, target_names)
    for num_classifiers_per_cat in [1, 2, 5]:
        ensembe_performance = ensemble_best_classifiers(projects, nested_cross_val_classifiers, num_classifiers_per_cat, train_docs, train_target, validation_docs, validation_target, target_names)


def ensemble_best_classifiers(projects, nested_cross_val_classifiers, num_classifiers_per_cat, train_docs, train_target, validation_docs, validation_target, target_names):
    classifiers_df = select_best_classifiers(num_classifiers_per_cat, target_names, nested_cross_val_classifiers)
    estimators = []
    for split, target, parameters, classifier in zip(classifiers_df['split_id'], classifiers_df['target'], classifiers_df['params'], classifiers_df['classifier']):
        name = target + str(split) + classifier
        pipeline = make_default_pipeline(get_classifier_by_name(classifier))
        pipeline.set_params(**unpandas_parameters(parameters))
        estimators.append((name, pipeline))

    ensemble_name = 'EnsembleTop' + str(num_classifiers_per_cat) + '_'.join(projects)
    final_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    final_params = {}
    classifiers_df.to_csv(output_path + 'parameters' + ensemble_name + '.csv')

    final_clf.fit(train_docs, train_target)
    y_predicted = final_clf.predict(validation_docs)

    test_set_prediction_df = pandas.DataFrame({'doc': validation_docs, 'target': validation_target, 'prediction': y_predicted})
    test_set_prediction_df.to_csv(output_path + 'predictions_' + ensemble_name + '.csv.zip', compression='zip')

    report = report_classifier_performance(validation_target, y_predicted, target_names, 0, final_params, final_params)
    log.s(ensemble_name)
    log.s(str(report))
    df = pandas.DataFrame(report)
    df.to_csv(output_path + 'performance' + ensemble_name + '.csv')
    return df


def nested_cross_validation_for_model_selection(projects, train_docs, train_target, target_names):
    classifiers_df = pandas.DataFrame()

    for bug_cat in target_names:
        bug_cat_report_df = pandas.DataFrame()

        for clf_name in classifiers:
            gs_params = make_default_gridsearch_parameters()
            gs_params.update(make_classifier_specific_gridsearch_parameters(clf_name))

            if clf_name != 'MultinomialNB(fit_prior=False)':
                gs_params.update({'clf__class_weight': [{target_names.index(bug_cat): 4}]})
            pipeline = make_default_pipeline(get_classifier_by_name(clf_name))
            scorer = make_scorer(class_f1_scorer, class_idx=target_names.index(bug_cat))
            ncv_results = nested_cross_validation(train_docs, train_target, target_names, None, pipeline,
                                                  gs_params, scoring=scorer)
            bug_cat_report_df = bug_cat_report_df.append(ncv_results)

        bug_cat_report_df = bug_cat_report_df.sort_values('F1_' + bug_cat)
        bug_cat_report_df['target'] = bug_cat
        bug_cat_report_df['projects'] = json.dumps(projects)
        classifiers_df = classifiers_df.append(bug_cat_report_df)

    classifiers_df.to_csv(output_path + 'ncv_classifiers_' + '_'.join(projects) + '.csv')

    return classifiers_df


def select_best_classifiers(num_classifiers_per_cat, target_names, nested_cross_val_classifiers):
    classifiers_df = pandas.DataFrame()
    for target in target_names:
        df = nested_cross_val_classifiers[nested_cross_val_classifiers['target'] == target].copy()
        df = df.sort_values('F1_' + target)
        df['target'] = target
        classifiers_df = classifiers_df.append(df.tail(num_classifiers_per_cat))

    return classifiers_df


if __name__ == "__main__":
    main()
