import matplotlib.pyplot as plt
import numpy as np
import pandas

from classification.step_Z_plot_evaluation import plot_bootstrap_boxdiagram
from classification.utils import report_classifier_performance
from dataset.excluded import excluded_projects
from file_anchor import root_dir

OUTPUT_DIR = root_dir() + 'evaluation/output/'

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


def bootstrap_survey_submissions(survey_df, rater_1_df):
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

    boxes = pandas.DataFrame()
    boxes = boxes.append(do_boostrap(submission_scores_df['F1 weighted average'], ' weighted\naverage'))
    boxes = boxes.append(do_boostrap(submission_scores_df['F1_concurrency'], 'con.'))
    boxes = boxes.append(do_boostrap(submission_scores_df['F1_memory'], 'mem.'))
    boxes = boxes.append(do_boostrap(submission_scores_df['F1_other'], 'oth.'))
    boxes = boxes.append(do_boostrap(submission_scores_df['F1_semantic'], 'sem.'))

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_bootstrap_boxdiagram(fig, ax, "", "F1", boxes) #, widths=(0.6, 0.6, 0.6, 0.6)
    # plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.ylim([0.4, 0.87])
    plt.savefig(OUTPUT_DIR + 'survey_class_specific_f1_bootsrap_boxplot_dense.pdf')
    boxes.to_csv(OUTPUT_DIR + 'survey_class_specific_f1_bootsrap_boxplot_dense.csv')
    plt.close()



def do_boostrap(series, label):
    repl = [np.mean(np.random.choice(series.dropna(), size=len(series))) for x in range(0, 1000)]
    return evaluate_bootstrap(np.array(repl), label)


def evaluate_bootstrap(series, label):
    mean = series.mean()
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(series, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(series, p))

    return pandas.DataFrame([{'alpha': alpha * 100,
                           'lower': lower * 100,
                           'upper': upper * 100,
                           'mean': mean,
                           'label': label}])

def main():
    rater_1_df = rater_1()
    survey_df = survey()

    bootstrap_survey_submissions(survey_df, rater_1_df)


if __name__ == '__main__':
    main()
