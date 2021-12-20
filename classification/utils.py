import glob
import json
import re
import numpy as np
import pandas
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils import resample

from classification.default_classifiers import classifiers
from file_anchor import root_dir
from helpers.Logger import Logger

log = Logger()


def load_dataset(projects_list=[], pick='ProjectsNotInList'):
    df = pandas.read_csv(root_dir() + 'dataset/classification_dataset.csv.zip', compression='zip')

    if pick == 'ProjectsNotInList':
        df = df[~df['projectName'].isin(projects_list)]
    elif pick == 'ProjectsFromList':
        df = df[df['projectName'].isin(projects_list)]

    df['title'] = df['title'].fillna('')
    df['body'] = df['body'].fillna('')

    targets = df['RootCause'].value_counts().index.to_list()
    targets.sort()
    targets = {cause: i for i, cause in enumerate(targets)}

    df['target'] = df['RootCause'].replace(targets)
    df['data'] = df["title"] + '\n' + df["body"]
    target = df.pop('target').values
    data = df.pop('data').values
    target_names = list(targets.keys())
    return data, target, target_names


def plot_numpy_confusion_matrix(cm, target_names, path):
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=target_names)
    disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal', values_format=None)
    plt.gcf().subplots_adjust(left=0.2)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def report_classifier_performance(y_test, y_predicted, target_names, split_id, params, clf):
    # log.s('\n' + metrics.classification_report(y_test, y_predicted, target_names=target_names))

    report = {
              'split_id': split_id,
              'precision macro average': [metrics.precision_score(y_test, y_predicted, average="macro")],
              'precision weighted average': [metrics.precision_score(y_test, y_predicted, average="weighted")],
              'recall macro average': [metrics.recall_score(y_test, y_predicted, average="macro")],
              'recall weighted average': [metrics.recall_score(y_test, y_predicted, average="weighted")],
              'accuracy': [metrics.accuracy_score(y_test, y_predicted)],
              'F1 macro average': [metrics.f1_score(y_test, y_predicted, average="macro")],
              'F1 weighted average': [metrics.f1_score(y_test, y_predicted, average="weighted")],
              'CM': json.dumps(confusion_matrix(y_test, y_predicted).tolist())}

    for idx, name in enumerate(target_names):
        if len(metrics.f1_score(y_test, y_predicted, average=None)) == 4:
            report.update({'F1_' + name: metrics.f1_score(y_test, y_predicted, average=None)[idx],
                           'recall_' + name: metrics.recall_score(y_test, y_predicted, average=None)[idx],
                           'precision_' + name: metrics.precision_score(y_test, y_predicted, average=None)[idx]})

    report.update({'params': json.dumps(params)})
    report.update({'classifier': str(clf)})

    return report


def evaluate_bootstrap(df, metrics, output_path):
    conf_intervals_df = pandas.DataFrame()
    for metric in metrics:
        if output_path:
            # plot scores
            df[metric].plot(kind='hist')
            plt.savefig(output_path + '_boootstrap_hist_' + metric + '.png')
            plt.close()
        # confidence intervals
        mean = df[metric].mean()
        median = df[metric].median()
        alpha = 0.95
        p = ((1.0 - alpha) / 2.0) * 100
        lower = max(0.0, np.percentile(df[metric], p))
        p = (alpha + ((1.0 - alpha) / 2.0)) * 100
        upper = min(1.0, np.percentile(df[metric], p))

        conf_intervals_df = conf_intervals_df.append(
            pandas.DataFrame([{'metric': metric,
                               'alpha': alpha * 100,
                               'lower': lower * 100,
                               'upper': upper * 100,
                               'mean': mean,
                               'median': median}]))

        log.s(metric + ': %.1f confidence interval %.1f%% and %.1f%%' % (alpha * 100, lower * 100, upper * 100))
    if output_path:
        conf_intervals_df.to_csv(output_path + '_bootstrap_results.csv')
    return conf_intervals_df


def get_classifier_by_name(clf_name):
    clf_index = [x.split('(')[0] for x in classifiers].index(clf_name.split('(')[0])
    return eval(classifiers[clf_index])


def clf_short_name(clf_name):
    return clf_name.split('(')[0]


def get_best_classifier(ncv_results, metric, make_pipeline):
    res_df = ncv_results.sort_values(metric).copy()
    clf_str = res_df['classifier'].to_list()[-1]
    params = unpandas_parameters(res_df['params'].to_list()[-1])
    pipeline = make_pipeline(get_classifier_by_name(clf_str))
    pipeline.set_params(**params)
    return pipeline, params


def unpandas_parameters(parameters):
    parameters_dict = json.loads(parameters)
    if 'clf__class_weight' in parameters_dict.keys():
        parameters_dict['clf__class_weight'] = {int(key): int(value) for key, value in parameters_dict['clf__class_weight'].items()}
    return parameters_dict


def bootstrap(docs, target, target_names, pipeline, params, output_path, clf_name=None):
    n_iterations = 100
    n_size = int(len(docs) * 0.8)

    performance_report_df = pandas.DataFrame()
    test_set_prediction_df = pandas.DataFrame()
    for i in range(n_iterations):
        log.s('at bootstrap iteration ' + str(i))
        docs_indices = list(range(0, len(docs)))
        train_idx, t_ = resample(docs_indices, target, n_samples=n_size, stratify=target)
        train_x = docs[train_idx]
        train_y = target[train_idx]

        test_idx = [x for x in docs_indices if x not in list(train_idx)]
        test_x = docs[test_idx]
        test_y = target[test_idx]

        pipeline.fit(train_x, train_y)
        y_predicted = pipeline.predict(test_x)
        if not clf_name:
            classifier_name = pipeline.named_steps['clf']
        else:
            classifier_name = clf_name
        report = report_classifier_performance(test_y, y_predicted, target_names, i,
                                             params,
                                             classifier_name)
        performance_report_df = performance_report_df.append(pandas.DataFrame(report))
        test_set_prediction_df = test_set_prediction_df.append(pandas.DataFrame({'doc': test_x, 'target': test_y, 'prediction': y_predicted}))

    performance_report_df.to_csv(output_path + '_bootstrap_iterations.csv')
    test_set_prediction_df.to_csv(output_path + '_bootstrap_iterations_predictions.csv.zip', compression='zip')
    evaluate_bootstrap(performance_report_df, ['accuracy', 'F1 macro average', 'F1 weighted average', 'F1_concurrency', 'F1_memory', 'F1_other', 'F1_semantic'], output_path)


def nested_cross_validation(docs, target, target_names, output_path, pipeline, gs_params, scoring='f1_macro'):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    report_df = pandas.DataFrame()

    conf_matrix = np.zeros((len(target_names), len(target_names)))
    for i, (train_idx, test_idx) in enumerate(outer_cv.split(docs, target)):
        train_x = docs[train_idx]
        train_y = target[train_idx]

        test_x = docs[test_idx]
        test_y = target[test_idx]

        single_report = run_inner_cv(i, train_x, test_x, train_y, test_y, target_names, pipeline, gs_params, scoring)
        conf_matrix = conf_matrix + np.array(json.loads(single_report['CM']))

        report = pandas.DataFrame(single_report)
        report_df = report_df.append(report)

    if output_path:
        plot_numpy_confusion_matrix(conf_matrix, target_names, output_path + '_confusion_matrix_combined.png')
        report_df.to_csv(output_path + '_cross_validation.csv')
    return report_df


def run_inner_cv(split_id, X_train, X_test, y_train, y_test, target_names, pipeline, gs_params, scoring):
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, gs_params, n_jobs=-1, cv=inner_cv, scoring=scoring) # verbose=10,
    grid_search.fit(X_train, y_train)
    y_predicted = grid_search.predict(X_test)

    return report_classifier_performance(y_test, y_predicted, target_names, split_id, grid_search.best_params_, grid_search.best_estimator_.named_steps['clf'])


def class_f1_scorer(y, y_pred, class_idx=0):
     return metrics.f1_score(y, y_pred, average=None)[class_idx]

