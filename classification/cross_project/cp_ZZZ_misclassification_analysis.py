import glob
import os
import pathlib
import re

import pandas
from matplotlib import pyplot as plt

from classification.step_ZZZ_misclassification_analysis import map_project_names, word_histogram
from file_anchor import root_dir
from helpers.Logger import Logger

log = Logger()
output_path = root_dir() + 'classification/cross_project/output/' + pathlib.Path(__file__).stem + '/'
os.makedirs(output_path, exist_ok=True)
target_names = ['concurrency', 'memory', 'other', 'semantic']


def step_5_cross_project():
    cross_proj_pred = load_cross_project_predictions()
    original_dataset, predictions = map_project_names(cross_proj_pred.copy())
    best_and_worst_bug_tickets(predictions)

def load_cross_project_predictions():
    classifier_names = {'LinearSVC(random_state=42)': 'SVM',
                        'LogisticRegression(random_state=42)': 'LRC',
                        'RandomForestClassifier(random_state=42)': 'RFC',
                        'MultinomialNB(fit_prior=False)': 'MNB',
                        'EnsembleTop1': 'Ens. 1',
                        'EnsembleTop2': 'Ens. 2',
                        'EnsembleTop5': 'Ens. 5'}

    cross_project_predictions = pandas.DataFrame()
    for perf_fn in glob.glob(root_dir() + 'classification/cross_project/output/cross_project_evaluation/predictions_*'):
        df = pandas.read_csv(perf_fn)
        name = re.search(r'EnsembleTop\d', perf_fn).group()
        df['classifier'] = name
        df['projects'] = perf_fn.split(name)[-1].strip('.csv.zip')
        cross_project_predictions = cross_project_predictions.append(df)

    for perf_fn in glob.glob(root_dir() + 'classification/cross_project/output/cross_project_evaluation_LRC_SVC_RFC/predictions_*'):
        df = pandas.read_csv(perf_fn)
        name = perf_fn.split('/')[-1].lstrip('predictions_').split(')')[0] + ')'
        df['classifier'] = name
        df['projects'] = perf_fn.split(name)[-1].strip('.csv.zip')
        cross_project_predictions = cross_project_predictions.append(df)

    for old, new in classifier_names.items():
        cross_project_predictions['classifier'] = cross_project_predictions['classifier'].str.replace(old, new, regex=False)

    return cross_project_predictions


def best_and_worst_bug_tickets(predictions):
    predictions['correct'] = predictions['target'] == predictions['prediction']
    mean_correct = predictions.groupby(['doc', 'url']).mean()
    count_occurence = predictions.groupby(['doc', 'url']).count().rename(columns={'correct': 'count'})
    df = pandas.merge(mean_correct[['correct', 'target', 'prediction']], count_occurence[['count']], on=['doc', 'url'])
    df = df.reset_index()
    df = df.sort_values('correct')

    always_correct = df[df['correct'] == 1]
    always_wrong = df[df['correct'] == 0]

    always_correct['target'].astype(int).apply(lambda x: target_names[x]).value_counts().to_csv(output_path + 'always_correct_bug_tickets_root_causes.csv')
    always_wrong['target'].astype(int).apply(lambda x: target_names[x]).value_counts().to_csv(output_path + 'always_wrong_bug_tickets_root_causes.csv')

    # issue_size_analysis(df[df['correct'] == 1].append(df[df['correct'] == 0]), output_path + 'always_correct_or_always_incorrect_bug_ticket_sizes.csv')

    always_correct.to_csv(output_path + 'always_correct_bug_tickets.csv')
    always_wrong.to_csv(output_path + 'always_wrong_bug_tickets.csv')

    # size_for_subgroup(always_correct.copy(), 'always_correct_bug_tickets')
    # size_for_subgroup(always_wrong.copy(), 'always_wrong_bug_tickets')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    word_histogram(always_wrong.copy(), 'always_wrong_bug_tickets', ax1)
    word_histogram(always_correct.copy(), 'always_correct_bug_tickets', ax2)
    plt.tight_layout()
    plt.savefig(output_path + 'words_bug_tickets_one_hot.png')
    plt.close()


def main():
    step_5_cross_project()


if __name__ == '__main__':
    main()
