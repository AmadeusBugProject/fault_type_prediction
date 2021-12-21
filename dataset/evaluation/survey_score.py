import numpy as np
import pandas
import matplotlib.pyplot as plt

from classification.step_Z_plot_evaluation import get_bootstrap_results
from classification.utils import evaluate_bootstrap, report_classifier_performance
from dataset.evaluation.krippendorff import krippendorff
from dataset.excluded import excluded_projects
from file_anchor import root_dir
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import sklearn.metrics as metrics

OUTPUT_DIR = root_dir() + 'datatset/evaluation/output/'

target_names = {'concurrency': 0, 'memory': 1, 'other': 2, 'semantic': 3}

def load_and_normalize_csv(path):
    df = pandas.read_csv(path)
    # df = df[~df['RootCause'].isnull()]
    df = df[['url', 'RootCause']]
    df['RootCause'] = df['RootCause'].fillna('')
    df['RootCause'] = df['RootCause'].str.lower()
    df['RootCause'] = df['RootCause'].str.strip()
    df.loc[~df['RootCause'].isin(['semantic', 'memory', 'concurrency', 'other']), 'RootCause'] = np.nan
    df['ProjectName'] = df['url'].apply(lambda x: x.split('/')[4])
    # df['ConfidenceNumeric'] = pandas.to_numeric(df['Confidence'])
    return df


def rater_1():
    rater_1_df = pandas.DataFrame()
    rater_1_df = rater_1_df.append(load_and_normalize_csv(root_dir() + 'dataset/manualClassification/keyword_memory_concurrency_bugs.csv'))
    rater_1_df = rater_1_df.append(load_and_normalize_csv(root_dir() + 'dataset/manualClassification/randomly_selected_bugs.csv'))
    rater_1_df = rater_1_df.append(load_and_normalize_csv(root_dir() + 'dataset/manualClassification/randomly_selected_other.csv'))
    rater_1_df = rater_1_df[~rater_1_df['ProjectName'].isin(excluded_projects)]
    return rater_1_df


def survey():
    df = pandas.read_csv(root_dir() + 'dataset/survey/survey.csv')
    # df = df[['url', 'answer']]
    df['ProjectName'] = df['url'].apply(lambda x: x.split('/')[4])
    df = df.rename(columns={'answer': 'RootCause'})
    df['RootCause'] = df['RootCause'].fillna('')
    df['RootCause'] = df['RootCause'].str.lower()
    df['RootCause'] = df['RootCause'].str.strip()
    df.loc[~df['RootCause'].isin(['semantic', 'memory', 'concurrency', 'other']), 'RootCause'] = np.nan

    df = df[~df['ProjectName'].isin(excluded_projects)]
    return df


def map_raters(r1_df, r2_df, remove_nan):
    r1k_df = r1_df.copy()
    r2k_df = r2_df.copy()
    r1k_df = r1k_df.set_index('url')
    r2k_df = r2k_df.set_index('url')
    merged = r1k_df.merge(r2k_df, left_index=True, right_index=True, how='outer')
    if remove_nan:
        merged = merged[~merged['RootCause_x'].isnull()]
        merged = merged[~merged['RootCause_y'].isnull()]
    return merged['RootCause_x'], merged['RootCause_y']


def evaluate(rater_1_df, rater_2_df):
    r1_df = rater_1_df.copy()
    r2_df = rater_2_df.copy()
    r1_df['RootCause'] = r1_df['RootCause'].replace(target_names)
    r2_df['RootCause'] = r2_df['RootCause'].replace(target_names)

    r1_target, r2_target = map_raters(r1_df, r2_df, remove_nan=True)

    r1_target.value_counts().to_csv(OUTPUT_DIR + 'survey_category_balance.csv')

    cm = confusion_matrix(r1_target, r2_target, normalize='all')
    disp = plot_numpy_confusion_matrix(cm, list(target_names.keys()))
    # ax1.set_ylabel('volts')
    disp.ax_.set_ylabel('Rater 1')
    disp.ax_.set_xlabel('Survey Participants')
    plt.gcf().subplots_adjust(left=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'survey_confusion_matrix.pdf')

    r1_kd_target, r2_kd_target = map_raters(r1_df, r2_df, remove_nan=False)

    # 'precision macro average': [metrics.precision_score(y_test, y_predicted, average="macro")],
    # 'precision weighted average': [metrics.precision_score(y_test, y_predicted, average="weighted")],
    # 'recall macro average': [metrics.recall_score(y_test, y_predicted, average="macro")],
    # 'recall weighted average': [metrics.recall_score(y_test, y_predicted, average="weighted")],
    # 'accuracy': [metrics.accuracy_score(y_test, y_predicted)],
    # 'F1 macro average': [metrics.f1_score(y_test, y_predicted, average="macro")],
    # 'F1 weighted average': [metrics.f1_score(y_test, y_predicted, average="weighted")],
    # 'GS Best Params': [grid_search.best_params_]

    print(metrics.classification_report(r1_target, r2_target, target_names=list(target_names.keys())))

    ir_metrics = {'cohens_kappa': metrics.cohen_kappa_score(r1_target, r2_target),
                  'weighted_f1': metrics.f1_score(r1_target, r2_target, average='weighted'),
                  'macro_f1': metrics.f1_score(r1_target, r2_target, average='macro'),
               'weighted_precision': metrics.precision_score(r1_target, r2_target, average='weighted'),
               'macro_precision': metrics.precision_score(r1_target, r2_target, average='macro'),
                   'weighted_recall': metrics.precision_score(r1_target, r2_target, average='weighted'),
                   'macro_recall': metrics.precision_score(r1_target, r2_target, average='macro'),
                  'accuracy': metrics.accuracy_score(r1_target, r2_target),
                  'krippendorff_alpha': krippendorff.alpha([r1_kd_target, r2_kd_target])}
    for name, idx in target_names.items():
        ir_metrics.update({'F1_' + name: metrics.f1_score(r1_target, r2_target, average=None)[idx],
                       'recall_' + name: metrics.recall_score(r1_target, r2_target, average=None)[idx],
                       'precision_' + name: metrics.precision_score(r1_target, r2_target, average=None)[idx]})
    pandas.DataFrame([ir_metrics]).to_csv(OUTPUT_DIR + 'survey_agreement.csv')

    print(metrics.cohen_kappa_score(r1_target, r2_target))
    print(metrics.f1_score(r1_target, r2_target, average='weighted'))
    print(metrics.accuracy_score(r1_target, r2_target))
    print(krippendorff.alpha([r1_kd_target, r2_kd_target]))


def plot_numpy_confusion_matrix(cm, target_names):
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=target_names)
    disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal', values_format=None)
    plt.gcf().subplots_adjust(left=0.2)
    plt.tight_layout()
    return disp


def plot_survey_performance(survey_df, rater_1_df):
    r1_df = rater_1_df.copy()
    s_df = survey_df.copy()
    r1_df['RootCause'] = r1_df['RootCause'].replace(target_names)
    s_df['RootCause'] = s_df['RootCause'].replace(target_names)
    s_df.describe().to_csv(OUTPUT_DIR + 'survey_describe.csv')

    submission_scores_df = pandas.DataFrame()
    for submission_id in s_df['submission_id'].value_counts().index.to_list():
        submission_df = s_df[s_df['submission_id'] == submission_id].copy()
        r1_target, r2_target = map_raters(r1_df, submission_df, remove_nan=True)

        report = pandas.DataFrame(report_classifier_performance(r1_target, r2_target, list(target_names.keys()), 0, {}, submission_id))
        submission_scores_df = submission_scores_df.append(pandas.DataFrame(report))

    submission_scores_df.describe().to_csv(OUTPUT_DIR + 'Survey_submission_scores.csv')

    # submission_scores_df['F1 weighted average'].plot(kind='hist')
    barplot_hist(submission_scores_df['F1 weighted average'], 'waF1')
    plt.savefig(OUTPUT_DIR + 'Survey_F1_weighted_average_hist.pdf')
    plt.close()

    # submission_scores_df['F1 macro average'].plot(kind='hist')
    barplot_hist(submission_scores_df['F1 macro average'], 'maF1')
    plt.savefig(OUTPUT_DIR + 'Survey_F1_macro_average_hist.pdf')
    plt.close()


def barplot_hist(series, metric):
    buckets = np.arange(0.0, 1.1, 0.1)
    bucket_labels = [("(%.1f-%.1f]" % (x, x + 0.1)).replace('0.', '.') for x in buckets][:-1]
    buckets_df = pandas.cut(series, buckets, labels=bucket_labels).value_counts().sort_index()
    ax = buckets_df.plot(kind='bar')
    plt.xticks(rotation=0)
    ax.set_ylabel('Occurences')
    ax.set_xlabel(metric)
    plt.tight_layout()


def main():
    rater_1_df = rater_1()
    survey_df = survey()

    plot_survey_performance(survey_df, rater_1_df)
    # evaluate(rater_1_df, survey_df[['url', 'RootCause', 'ProjectName']].copy())
    #
    # # bug tickets classified more than 3 times in survey:
    # r1_df = rater_1_df.copy()
    # s_df = survey_df.copy()
    # r1_df['RootCause'] = r1_df['RootCause'].replace(target_names)
    # s_df['RootCause'] = s_df['RootCause'].replace(target_names)
    #
    # r1k_df = r1_df.set_index('url')
    # r2k_df = s_df.set_index('url')
    # merged = r1k_df.merge(r2k_df, left_index=True, right_index=True, how='outer')
    # merged = merged[~merged['RootCause_x'].isnull()]
    # df = merged[~merged['RootCause_y'].isnull()].copy()
    # df = df.reset_index()
    # # df = df[df['url'].isin([x for x in df['url'].value_counts().index if df['url'].value_counts()[x] > 3])]
    # df['correct'] = df['RootCause_x'] == df['RootCause_y']



if __name__ == '__main__':
    main()
