import os
import os
import pathlib

import pandas
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import confusion_matrix

from artifact_detection_model.transformer.ArtifactRemoverTransformer import ArtifactRemoverTransformer, \
    KEEP_EXCEPTION_NAMES
from classification.utils import plot_numpy_confusion_matrix
from file_anchor import root_dir
from helpers.Logger import Logger

log = Logger()
output_path = root_dir() + 'classification/output/' + pathlib.Path(__file__).stem + '/'
os.makedirs(output_path, exist_ok=True)
target_names = ['concurrency', 'memory', 'other', 'semantic']


def step_4():
    step_4_ens1 = pandas.read_csv(root_dir() + 'classification/output/step_4_ensembles/EnsembleTop1_bootstrap_iterations_predictions.csv.zip')

    boostrap_confusion_matrix(step_4_ens1.copy(), output_path + 'step_4_ens1_bootstrap_confusion_matrix.pdf')
    issue_size_analysis(step_4_ens1.copy(), output_path + 'issue_size_analysis.txt')

    original_dataset, predictions = map_project_names(step_4_ens1.copy())
    origin_project_analysis(original_dataset, predictions.copy())

    best_and_worst_bug_tickets(predictions.copy())


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

    issue_size_analysis(df[df['correct'] == 1].append(df[df['correct'] == 0]), output_path + 'always_correct_or_always_incorrect_bug_ticket_sizes.csv')

    always_correct.to_csv(output_path + 'always_correct_bug_tickets.csv')
    always_wrong.to_csv(output_path + 'always_wrong_bug_tickets.csv')

    size_for_subgroup(always_correct.copy(), 'always_correct_bug_tickets')
    size_for_subgroup(always_wrong.copy(), 'always_wrong_bug_tickets')

    # fig, ax = plt.subplots()
    # word_histogram(always_correct.copy(), 'always_correct_bug_tickets', ax)
    # plt.savefig(output_path + 'always_correct_bug_tickets_one_hot.png')
    # plt.tight_layout()
    # plt.close()
    #
    # fig, ax = plt.subplots()
    # word_histogram(always_wrong.copy(), 'always_wrong_bug_tickets', ax)
    # plt.savefig(output_path + 'always_wrong_bug_tickets_one_hot.png')
    # plt.tight_layout()
    # plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 5))
    word_histogram(always_wrong.copy(), 'always wrong bug tickets', ax1)
    word_histogram(always_correct.copy(), 'always correct bug tickets', ax2)
    plt.tight_layout()
    plt.savefig(output_path + 'words_bug_tickets_one_hot.pdf')
    plt.close()


def word_histogram(df, label, ax):
    all_words = pandas.DataFrame({'words': ' '.join(df['doc'].to_list()).split()})
    all_words['words'] = all_words['words'].str.lower()
    word_frequency = pandas.DataFrame({'word': all_words['words'].value_counts().index, 'count': all_words['words'].value_counts().values})
    word_frequency = word_frequency[word_frequency['word'].str.fullmatch(r'[A-Za-z]+')]
    word_frequency = word_frequency[~word_frequency['word'].isin(ENGLISH_STOP_WORDS)]

    df['words'] = df['doc'].str.lower().str.split()
    word_frequency['one_hot'] = word_frequency['word'].apply(
        lambda x: len(df[df['words'].apply(lambda y: x in y)]))

    ax = word_frequency.sort_values('one_hot').tail(20).plot.barh(ax=ax, x='word', y='one_hot', legend=None)
    ax.set_xlabel('word frequency')
    ax.set_ylabel('')
    ax.set_title(label)

    # word_frequency.sort_values('count').tail(20).plot.bar(x='word', y='count')
    # plt.savefig(output_path + label + '_count.png')
    # plt.tight_layout()
    # plt.close()


def size_for_subgroup(df, label):
    df['doc_length'] = df['doc'].str.len()
    df['amount_of_lines_in_doc'] = df['doc'].str.split('\n').apply(lambda x: len(x))
    doc_length = pandas.DataFrame().append(df['doc_length'].describe())
    amount_of_lines_in_doc = pandas.DataFrame().append(df['amount_of_lines_in_doc'].describe())
    doc_length['description'] = label
    amount_of_lines_in_doc['description'] = label
    doc_length.to_csv(output_path + label + '_char_length.csv')
    amount_of_lines_in_doc.to_csv(output_path + label + '_lines_length.csv')


def map_project_names(predictions):
    df = pandas.read_csv(root_dir() + 'dataset/classification_dataset.csv.zip', compression='zip')
    df['title'] = df['title'].fillna('')
    df['body'] = df['body'].fillna('')

    targets = df['RootCause'].value_counts().index.to_list()
    targets.sort()
    targets = {cause: i for i, cause in enumerate(targets)}

    df['target_cat'] = df['RootCause'].replace(targets)
    df['doc'] = df["title"] + '\n' + df["body"]
    project_count_df = pandas.DataFrame({'projectName': df['projectName'].value_counts().index, 'issue_count_for_project': df['projectName'].value_counts().values})
    df = df.merge(project_count_df, on='projectName')
    return df, predictions.merge(df, on='doc')


def origin_project_analysis(original_dataset, prediction):
    project_count_df = pandas.DataFrame({'projectName': original_dataset['projectName'].value_counts().index, 'issue_count_for_project': original_dataset['projectName'].value_counts().values})

    prediction_project_issues_count = pandas.DataFrame({'projectName': prediction['projectName'].value_counts().index, 'issue_count_for_project': prediction['projectName'].value_counts().values})
    project_count_df = project_count_df.merge(prediction_project_issues_count, on='projectName')

    project_count_df['accuracy'] = project_count_df['projectName'].apply(lambda x: metrics.accuracy_score(
        prediction[prediction['projectName'] == x]['target'], prediction[prediction['projectName'] == x]['prediction']))

    project_count_df['precision_weighted_average'] = project_count_df['projectName'].apply(lambda x: metrics.precision_score(
        prediction[prediction['projectName'] == x]['target'], prediction[prediction['projectName'] == x]['prediction'], average="weighted"))

    project_count_df['recall_weighted_average'] = project_count_df['projectName'].apply(lambda x: metrics.recall_score(
        prediction[prediction['projectName'] == x]['target'], prediction[prediction['projectName'] == x]['prediction'], average="weighted"))

    project_count_df['f1_weighted_average'] = project_count_df['projectName'].apply(lambda x: metrics.f1_score(
        prediction[prediction['projectName'] == x]['target'], prediction[prediction['projectName'] == x]['prediction'], average="weighted"))

    project_count_df = project_count_df.sort_values(by='f1_weighted_average', ascending=False, ignore_index=True)
    print(project_count_df.to_string())
    project_count_df.to_csv(output_path + 'origin_project_analysis.csv')

    project_count_df[project_count_df['issue_count_for_project_x'] >= 10].to_csv(output_path + 'origin_project_analysis_more_than_10_tickets.csv')


def issue_size_analysis(df, out_file):
    df['correct'] = df['target'] == df['prediction']
    # comp_df = pandas.DataFrame()

    output = []
    output.append('char length of doc')
    df['doc_length'] = df['doc'].str.len()
    # print(is_normal(df['doc_length'], 'doc_length').to_string())
    output.append(correct_vs_incorrect_predictions(df, 'doc_length').to_string())

    output.append('char lenght of doc with artifacts removed')
    art = ArtifactRemoverTransformer(replacement_strategy=KEEP_EXCEPTION_NAMES)
    df['artifacts_removed_doc'] = art.transform(df['doc'])
    df['artifacts_removed_doc_length'] = df['artifacts_removed_doc'].str.len()
    # print(is_normal(df['artifacts_removed_doc_length'], 'doc_length').to_string())
    output.append(correct_vs_incorrect_predictions(df, 'artifacts_removed_doc_length').to_string())

    output.append('char length of removed artifacts')
    df['amount_of_removed_artifacts'] = df['doc_length'] - df['artifacts_removed_doc_length']
    # print(is_normal(df['amount_of_removed_artifacts'], 'doc_length').to_string())
    # stats.probplot(df['amount_of_removed_artifacts'], dist="norm", plot=plt)
    # plt.show()
    output.append(correct_vs_incorrect_predictions(df, 'amount_of_removed_artifacts').to_string())

    output.append('lines of doc')
    df['amount_of_lines_in_doc'] = df['doc'].str.split('\n').apply(lambda x: len(x))
    # print(is_normal(df['amount_of_lines_in_doc'], 'doc_length').to_string())
    output.append(correct_vs_incorrect_predictions(df, 'amount_of_lines_in_doc').to_string())

    output.append('lines of doc with artifacts removed')
    df['artifacts_removed_amount_of_lines_in_doc'] = df['artifacts_removed_doc'].str.split('\n').apply(lambda x: len(x))
    # print(is_normal(df['artifacts_removed_amount_of_lines_in_doc'], 'doc_length').to_string())
    output.append(correct_vs_incorrect_predictions(df, 'artifacts_removed_amount_of_lines_in_doc').to_string())

    output.append('lines removed as artifacts')
    df['amount_of_removed_lines_as_artifacts'] = df['amount_of_lines_in_doc'] - df['artifacts_removed_amount_of_lines_in_doc']
    # print(is_normal(df['amount_of_removed_lines_as_artifacts'], 'doc_length').to_string())
    # stats.probplot(df['amount_of_removed_artifacts'], dist="norm", plot=plt)
    # plt.show()
    output.append(correct_vs_incorrect_predictions(df, 'amount_of_removed_lines_as_artifacts').to_string())

    output.append('if doc contains exception names (mean ends up as percentage)')
    exception_name_re = r"(?:(?:[A-Z][a-z0-9]*)+(?:Exception|Error))"
    df['contains_exception_name'] = df['doc'].str.contains(exception_name_re) * 1
    # print(is_normal(df['contains_exception_name'], 'doc_length').to_string())
    output.append(correct_vs_incorrect_predictions(df, 'contains_exception_name').to_string())

    output.append('number of exception names contained in doc')
    df['count_exception_name'] = df['doc'].str.count(exception_name_re)
    # print(is_normal(df['count_exception_name'], 'doc_length').to_string())
    output.append(correct_vs_incorrect_predictions(df, 'count_exception_name').to_string())

    print('\n'.join(output))
    with open(out_file, 'w') as fd:
        fd.write('\n'.join(output))


def correct_vs_incorrect_predictions(df, metric):
    comp_df = pandas.DataFrame()

    correct = pandas.DataFrame().append(df[df['correct']][metric].describe())
    correct['description'] = 'correct'
    comp_df = comp_df.append(correct)

    incorrect = pandas.DataFrame().append(df[~df['correct']][metric].describe())
    incorrect['description'] = 'incorrect'
    comp_df = comp_df.append(incorrect)

    return comp_df


def boostrap_confusion_matrix(df, filename):
    conf_matrix = confusion_matrix(df['target'], df['prediction'])
    plot_numpy_confusion_matrix(conf_matrix, target_names, filename)


def main():
    step_4()


if __name__ == '__main__':
    main()
