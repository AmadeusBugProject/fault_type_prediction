import glob
import json
import os
import pathlib
import re
from itertools import chain

import matplotlib.pyplot as plt
import pandas
from sklearn import metrics

from classification.cross_project.cp_ZZZ_misclassification_analysis import load_cross_project_predictions
from classification.step_ZZZ_misclassification_analysis import map_project_names
from classification.step_Z_plot_evaluation import plot_bootstrap_boxdiagram
from classification.utils import evaluate_bootstrap
from file_anchor import root_dir
from helpers.Logger import Logger

log = Logger()
output_path = root_dir() + 'classification/cross_project/output/' + pathlib.Path(__file__).stem + '/'
os.makedirs(output_path, exist_ok=True)


def load_cross_project_evaluation():
    cross_project_scores = pandas.DataFrame()
    for perf_fn in glob.glob(root_dir() + 'classification/cross_project/output/cross_project_evaluation/performance*'):
        df = pandas.read_csv(perf_fn)
        name = re.search(r'EnsembleTop\d', perf_fn).group()
        df['classifier'] = name
        df['projects'] = perf_fn.split(name)[-1].strip('.csv')
        cross_project_scores = cross_project_scores.append(df)

    for perf_fn in glob.glob(root_dir() + 'classification/cross_project/output/cross_project_evaluation_LRC_SVC_RFC/performance*'):
        df = pandas.read_csv(perf_fn)
        df['projects'] = perf_fn.split('/')[-1].lstrip('performance').strip('.csv')
        cross_project_scores = cross_project_scores.append(df)

    classifier_names = {'LinearSVC(random_state=42)': 'SVM',
                        'LogisticRegression(random_state=42)': 'LRC',
                        'RandomForestClassifier(random_state=42)': 'RFC',
                        'MultinomialNB(fit_prior=False)': 'MNB',
                        'EnsembleTop1': 'Ens. 1',
                        'EnsembleTop2': 'Ens. 2',
                        'EnsembleTop5': 'Ens. 5'}

    for old, new in classifier_names.items():
        cross_project_scores['classifier'] = cross_project_scores['classifier'].str.replace(old, new, regex=False)

    return classifier_names, cross_project_scores


def cross_project_evaluation(classifier_names, cross_project_scores, filename_postfix = ''):
    bootstrap_intervals = pandas.DataFrame()
    for classifier in classifier_names.values():
        df = evaluate_bootstrap(cross_project_scores[cross_project_scores['classifier'] == classifier], ['F1 macro average'], None)
        df['label'] = classifier
        bootstrap_intervals = bootstrap_intervals.append(df)

    bootstrap_intervals.to_csv(output_path + 'cross_project_bootstrap_intervals' + filename_postfix + '.csv')
    cross_project_scores.to_csv(output_path + 'cross_project_scores' + filename_postfix + '.csv')

    fig, ax = plt.subplots(figsize=(5, 3))
    plot_bootstrap_boxdiagram(fig, ax, "", "maF1", bootstrap_intervals)
    plt.tight_layout()
    plt.savefig(output_path + 'cross_project_f1_macro_average_bootsrap_boxplot' + filename_postfix + '.pdf')
    plt.close()


# def redo based on separate reports, this only averages the performance of the whole split if a project occurs in it
def worst_performing_in_cross_project():
    classifier_names, cross_project_scores = load_cross_project_evaluation()

    ens1_df = cross_project_scores[cross_project_scores['classifier'] == 'Ensemble Top 1'].copy()

    dataset_df = pandas.read_csv(root_dir() + 'dataset/classification_dataset.csv.zip', compression='zip')
    projects_df = pandas.DataFrame({'projectName': dataset_df['projectName'].value_counts().index, 'numBugReports': dataset_df['projectName'].value_counts().values})

    projects_df['mean macro averaged F1'] = projects_df['projectName'].apply(lambda x: ens1_df[ens1_df['projects'].str.contains(x)]['F1 macro average'].mean())
    projects_df = projects_df.sort_values(by='mean macro averaged F1', ignore_index=True)


def origin_project_analysis(original_dataset, prediction, classifier_name):
    project_count_df = pandas.DataFrame({'projectName': original_dataset['projectName'].value_counts().index, 'issue_count_for_project': original_dataset['projectName'].value_counts().values})

    more_than_ten_issues = project_count_df[project_count_df['issue_count_for_project'] >= 10].copy()
    less_than_ten_issues = project_count_df[project_count_df['issue_count_for_project'] < 10].copy()

    project_count_df = more_than_ten_issues.append(pandas.DataFrame({'projectName': ['ALL_OTHERS'],
                                                                     'issue_count_for_project': [less_than_ten_issues['issue_count_for_project'].sum()]}))

    all_other_replace_dict = {x: 'ALL_OTHERS' for x in list(less_than_ten_issues['projectName'])}

    prediction = prediction.replace({'projectName': all_other_replace_dict})

    prediction_project_issues_count = pandas.DataFrame({'projectName': prediction['projectName'].value_counts().index, 'issue_count_for_project': prediction['projectName'].value_counts().values})
    project_count_df = project_count_df.merge(prediction_project_issues_count, on='projectName')

    with open(root_dir() + 'classification/cross_project/output/project_selection/project_combos.json', 'r') as fd:
        project_combos = json.load(fd)
    projects_df = pandas.DataFrame({'projectName': list(chain.from_iterable([x['projects'] for x in project_combos]))})

    splits_df = pandas.DataFrame({'projectName': projects_df['projectName'].value_counts().index, 'projectOccurrencesInProjectSplits': projects_df['projectName'].value_counts().values})
    splits_df = splits_df[splits_df['projectName'].isin(more_than_ten_issues['projectName'])]

    other_project_combos = [x for x in project_combos if (y in list(less_than_ten_issues['projectName']) for y in x['projects'])]

    splits_df = splits_df.append(pandas.DataFrame({'projectName': ['ALL_OTHERS'],
                                                   'projectOccurrencesInProjectSplits': [len(other_project_combos)]}))
    project_count_df = project_count_df.merge(splits_df, on='projectName')

    # project_count_df['accuracy'] = project_count_df['projectName'].apply(lambda x: metrics.accuracy_score(
    #     prediction[prediction['projectName'] == x]['target'], prediction[prediction['projectName'] == x]['prediction']))

    project_count_df['precision_weighted_average'] = project_count_df['projectName'].apply(lambda x: metrics.precision_score(
        prediction[prediction['projectName'] == x]['target'], prediction[prediction['projectName'] == x]['prediction'], average="weighted"))

    project_count_df['recall_weighted_average'] = project_count_df['projectName'].apply(lambda x: metrics.recall_score(
        prediction[prediction['projectName'] == x]['target'], prediction[prediction['projectName'] == x]['prediction'], average="weighted"))

    project_count_df['f1_weighted_average'] = project_count_df['projectName'].apply(lambda x: metrics.f1_score(
        prediction[prediction['projectName'] == x]['target'], prediction[prediction['projectName'] == x]['prediction'], average="weighted"))

    project_count_df = project_count_df.sort_values(by='f1_weighted_average', ascending=False, ignore_index=True)

    project_count_df.to_csv(output_path + 'origin_project_analysis_more_than_10_tickets' + classifier_name + '.csv')
    project_count_df.to_latex(output_path + 'origin_project_analysis_more_than_10_tickets' + classifier_name + '.tex')


def main():
    classifier_names, cross_project_scores = load_cross_project_evaluation()
    cross_project_evaluation(classifier_names, cross_project_scores)

    no_elastic_df = cross_project_scores[~cross_project_scores['projects'].str.contains('elasticsearch')]
    cross_project_evaluation(classifier_names, no_elastic_df, filename_postfix='without_elasticsearch')

    cross_proj_pred = load_cross_project_predictions()
    original_dataset, predictions = map_project_names(cross_proj_pred.copy())
    for classifier in classifier_names.values():
        origin_project_analysis(original_dataset, predictions[predictions['classifier'] == classifier], classifier)

    # redo the following to do per misclassified issues, not by performance of split:
    # worst_performing_in_cross_project()


if __name__ == '__main__':
    main()
