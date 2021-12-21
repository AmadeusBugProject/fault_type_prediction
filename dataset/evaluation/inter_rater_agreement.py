import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import metrics

from dataset.evaluation.krippendorff import krippendorff
from dataset.excluded import excluded_projects
from file_anchor import root_dir
from sklearn.metrics import cohen_kappa_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

OUTPUT_DIR = root_dir() + 'dataset/evaluation/output/'


def load_and_normalize_csv(path):
    df = pandas.read_csv(path)
    # df = df[~df['RootCause'].isnull()]
    df = df[['url', 'RootCause', 'RootCauseDetail']]
    df['RootCauseDetail'] = df['RootCauseDetail'].fillna('')
    df['RootCause'] = df['RootCause'].fillna('')
    # df['ConfidenceNumeric'] = df['RootCause'].fillna(0)
    df['RootCauseDetail'] = df['RootCauseDetail'].str.lower()
    df['RootCauseDetail'] = df['RootCauseDetail'].str.strip()
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
    # rater_1_df = rater_1_df[~rater_1_df['ProjectName'].isin(excluded_projects)]
    return rater_1_df


def rater_2():
    rater_2_df = load_and_normalize_csv(root_dir() + 'dataset/manualClassification/validation/rater_2.csv')
    # rater_2_df = rater_2_df[~rater_2_df['ProjectName'].isin(excluded_projects)]
    return rater_2_df


def rater_1_internal_val():
    rater_1_inter_df = load_and_normalize_csv(root_dir() + 'dataset/manualClassification/validation/random_internal.csv')
    # rater_1_inter_df = rater_1_inter_df[~rater_1_inter_df['ProjectName'].isin(excluded_projects)]
    return rater_1_inter_df


def map_removing_nan(r1_df, r2_df):
    r1_df = r1_df[~r1_df['RootCause'].isnull()].copy()
    r2_df = r2_df[~r2_df['RootCause'].isnull()].copy()
    r1_df = r1_df[r1_df['url'].isin(r2_df['url'])]
    r2_df = r2_df[r2_df['url'].isin(r1_df['url'])]
    r1_df = r1_df.sort_values(by=['url'])
    r2_df = r2_df.sort_values(by=['url'])
    r1_target = r1_df['RootCause']
    r2_target = r2_df['RootCause']
    return r1_target, r2_target


def map_keep_nans(r1_df, r2_df):
    r1k_df = r1_df.copy()
    r2k_df = r2_df.copy()
    r1k_df = r1k_df.set_index('url')
    r2k_df = r2k_df.set_index('url')
    merged = r1k_df.merge(r2k_df, left_index=True, right_index=True,how='outer')
    return merged['RootCause_x'], merged['RootCause_y']


def evaluate(rater_1_df, rater_2_df, out_filename):
    r1_df = rater_1_df.copy()
    r2_df = rater_2_df.copy()
    cat_map = {'concurrency': 0, 'memory': 1, 'other': 2, 'semantic': 3}
    r1_df['RootCause'] = r1_df['RootCause'].replace(cat_map)
    r2_df['RootCause'] = r2_df['RootCause'].replace(cat_map)

    print('num classifications rater 1: ' + str(r1_df[~r1_df['RootCause'].isnull()].shape[0]))
    print('num classifications rater 2: ' + str(r2_df[~r2_df['RootCause'].isnull()].shape[0]))

    r1_target, r2_target = map_removing_nan(r1_df, r2_df)
    cm = confusion_matrix(r1_target, r2_target)
    disp = plot_numpy_confusion_matrix(cm, list(cat_map.keys()))
    # ax1.set_ylabel('volts')
    disp.ax_.set_ylabel('Rater 1')
    disp.ax_.set_xlabel('Rater 2')
    plt.gcf().subplots_adjust(left=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + out_filename + '.png')

    r1_kd_target, r2_kd_target = map_keep_nans(r1_df, r2_df)

    ir_metrics = {'cohens_kappa': cohen_kappa_score(r1_target, r2_target),
                  'weighted_f1': f1_score(r1_target, r2_target, average='weighted'),
                  'macro_f1': f1_score(r1_target, r2_target, average='macro'),
                  'accuracy': accuracy_score(r1_target, r2_target),
                  'krippendorff_alpha': krippendorff.alpha([r1_kd_target, r2_kd_target])}
    for name, idx in cat_map.items():
        ir_metrics.update({'F1_' + name: metrics.f1_score(r1_target, r2_target, average=None)[idx],
                       'recall_' + name: metrics.recall_score(r1_target, r2_target, average=None)[idx],
                       'precision_' + name: metrics.precision_score(r1_target, r2_target, average=None)[idx]})
    pandas.DataFrame([ir_metrics]).to_csv(OUTPUT_DIR + out_filename + '.csv')

    print(cohen_kappa_score(r1_target, r2_target))
    print(f1_score(r1_target, r2_target, average='weighted'))
    print(accuracy_score(r1_target, r2_target))
    print(krippendorff.alpha([r1_kd_target, r2_kd_target]))


def investigate_wrong_classifications(r1_df, r2_df):
    r1k_df = r1_df.copy()
    r2k_df = r2_df.copy()
    r1k_df = r1k_df.set_index('url')
    r2k_df = r2k_df.set_index('url')
    merged = r1k_df.merge(r2k_df, left_index=True, right_index=True,how='outer')
    #merged['RootCause_x'], merged['RootCause_y']
    merged = merged[~merged['RootCause_x'].isnull()]
    merged = merged[~merged['RootCause_y'].isnull()]
    print(merged[merged['RootCause_x'] != merged['RootCause_y']].to_string())


def plot_numpy_confusion_matrix(cm, target_names):
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=target_names)
    disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal', values_format=None)
    return disp


def main():
    rater_1_df = rater_1()
    rater_2_df = rater_2()
    evaluate(rater_1_df, rater_2_df, 'inter_rater_agreement_external')

    rater_1_inter_df = rater_1_internal_val()
    evaluate(rater_1_df, rater_1_inter_df, 'inter_rater_agreement_internal')

    # investigate_wrong_classifications(rater_1_df, rater_2_df)


if __name__ == '__main__':
    main()
