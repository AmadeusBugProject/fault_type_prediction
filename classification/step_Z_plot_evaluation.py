import glob
import os
import pathlib
import re

import pandas
import matplotlib.pyplot as plt

from classification.default_classifiers import classifiers
from file_anchor import root_dir
from helpers.Logger import Logger

log = Logger()
output_path = root_dir() + 'classification/output/' + pathlib.Path(__file__).stem + '/'
os.makedirs(output_path, exist_ok=True)


def get_bootstrap_results(report_paths, clf_names, metric='F1 macro average'):
    df = pandas.DataFrame()
    for clf, report_path in zip(clf_names, report_paths):
        clf_df = pandas.read_csv(root_dir() + report_path)
        clf_df['label'] = clf
        df = df.append(clf_df[clf_df['metric'] == metric])
    return df.sort_values('label')


def step_2(report_path, step_name):
    step_2_clf_names = [x.split('(')[0] for x in classifiers]
    step_2_report_paths = [report_path + x + '_bootstrap_results.csv' for x in step_2_clf_names]
    df = get_bootstrap_results(step_2_report_paths, step_2_clf_names, metric='F1 macro average')

    clf_name_repl = {'MultinomialNB': 'MNB',
                        'LinearSVC': 'SVM',
                        'RandomForestClassifier': 'RF',
                        'LogisticRegression': 'LR'}
    df = df.replace({'label': clf_name_repl})

    df.replace()
    df.to_csv(output_path + 'step_2_bootstrap_results' + step_name + '.csv')

    fig, ax = plt.subplots(figsize=(2.5, 3))
    plot_bootstrap_boxdiagram(fig, ax, "", "maF1", df, widths=(0.6, 0.6, 0.6, 0.6))
    plt.tight_layout()
    plt.ylim([0.28, 0.88])
    plt.savefig(output_path + step_name + '_f1_macro_average_bootsrap_boxplot.pdf')
    plt.close()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(8, 3.4))
    plot_bootstrap_boxdiagram(fig, ax1, "Concurrency", "F1", get_bootstrap_results(step_2_report_paths, step_2_clf_names, metric='F1_concurrency').replace({'label': clf_name_repl}), widths=(0.6, 0.6, 0.6, 0.6))
    plot_bootstrap_boxdiagram(fig, ax2, "Memory", "", get_bootstrap_results(step_2_report_paths, step_2_clf_names, metric='F1_memory').replace({'label': clf_name_repl}), widths=(0.6, 0.6, 0.6, 0.6))
    plot_bootstrap_boxdiagram(fig, ax3, "Other", "", get_bootstrap_results(step_2_report_paths, step_2_clf_names, metric='F1_other').replace({'label': clf_name_repl}), widths=(0.6, 0.6, 0.6, 0.6))
    plot_bootstrap_boxdiagram(fig, ax4, "Semantic", "", get_bootstrap_results(step_2_report_paths, step_2_clf_names, metric='F1_semantic').replace({'label': clf_name_repl}), widths=(0.6, 0.6, 0.6, 0.6))
    plt.tight_layout()
    plt.ylim([0.28, 0.88])
    plt.savefig(output_path + step_name + '_class_specific_f1_bootsrap_boxplot.pdf')
    plt.close()


def step_1():
    step_1_report_paths = ['classification/output/step_1_0_trivial_approach/LinearSVC_bootstrap_results.csv',
                           'classification/output/step_1_1_trivial_approach_with_artifact_replacement/LinearSVC_bootstrap_results.csv']
    step_1_classifier_names = ['SVM', 'SVM + a.r.']
    bootstrap_results_df = get_bootstrap_results(step_1_report_paths, step_1_classifier_names, metric='F1 macro average')

    bootstrap_results_df.to_csv(output_path + 'step_1_bootstrap_results.csv')

    fig, ax = plt.subplots(figsize=(2.5, 3))
    # fig.set_figwidth(3)
    plot_bootstrap_boxdiagram(fig, ax, "", "maF1", bootstrap_results_df, widths=(0.4, 0.4))
    plt.tight_layout()
    plt.ylim([0.35, 0.85])
    plt.savefig(output_path + 'step_1_f1_macro_average_bootsrap_boxplot.pdf')
    plt.close()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(8, 3.4))
    plot_bootstrap_boxdiagram(fig, ax1, "Concurrency", "F1", get_bootstrap_results(step_1_report_paths, step_1_classifier_names, metric='F1_concurrency'), widths=(0.4, 0.4))
    plot_bootstrap_boxdiagram(fig, ax2, "Memory", "", get_bootstrap_results(step_1_report_paths, step_1_classifier_names, metric='F1_memory'), widths=(0.4, 0.4))
    plot_bootstrap_boxdiagram(fig, ax3, "Other", "", get_bootstrap_results(step_1_report_paths, step_1_classifier_names, metric='F1_other'), widths=(0.4, 0.4))
    plot_bootstrap_boxdiagram(fig, ax4, "Semantic", "", get_bootstrap_results(step_1_report_paths, step_1_classifier_names, metric='F1_semantic'), widths=(0.4, 0.4))
    plt.tight_layout()
    plt.ylim([0.35, 0.85])
    plt.savefig(output_path + 'step_1_class_specific_f1_bootsrap_boxplot.pdf')
    plt.close()


def step_4():
    report_paths = ['classification/output/step_2_0_ncv_multiple_algorithms/LinearSVC_bootstrap_results.csv',
                    'classification/output/step_2_0_ncv_multiple_algorithms/LogisticRegression_bootstrap_results.csv',
                    'classification/output/step_2_0_ncv_multiple_algorithms/RandomForestClassifier_bootstrap_results.csv',
                    'classification/output/step_2_0_ncv_multiple_algorithms/MultinomialNB_bootstrap_results.csv',
                    'classification/output/step_4_ensembles/EnsembleTop1_bootstrap_results.csv',
                    'classification/output/step_4_ensembles/EnsembleTop2_bootstrap_results.csv',
                    'classification/output/step_4_ensembles/EnsembleTop5_bootstrap_results.csv'
                    ]
    classifier_names = ['SVM',
                        'LR',
                        'RF',
                        'MNB',
                        'Ens. 1',
                        'Ens. 2',
                        'Ens. 5'
                        ]
    bootstrap_results_df = get_bootstrap_results(report_paths, classifier_names, metric='F1 macro average')

    bootstrap_results_df.to_csv(output_path + 'step_4_bootstrap_results.csv')

    fig, ax = plt.subplots(figsize=(2.5, 3))
    plot_bootstrap_boxdiagram(fig, ax, "", "maF1", bootstrap_results_df)
    plt.tight_layout()
    plt.savefig(output_path + 'step_4_f1_macro_average_bootsrap_boxplot.pdf')
    plt.close()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(8, 3.4))
    plot_bootstrap_boxdiagram(fig, ax1, "Concurrency", "F1", get_bootstrap_results(report_paths, classifier_names, metric='F1_concurrency'))
    plot_bootstrap_boxdiagram(fig, ax2, "Memory", "", get_bootstrap_results(report_paths, classifier_names, metric='F1_memory'))
    plot_bootstrap_boxdiagram(fig, ax3, "Other", "", get_bootstrap_results(report_paths, classifier_names, metric='F1_other'))
    plot_bootstrap_boxdiagram(fig, ax4, "Semantic", "", get_bootstrap_results(report_paths, classifier_names, metric='F1_semantic'))
    plt.tight_layout()
    plt.savefig(output_path + 'step_4_class_specific_f1_bootsrap_boxplot.pdf')
    plt.close()

def step_4_dense():
    report_paths_LR = ['classification/output/step_2_0_ncv_multiple_algorithms/LogisticRegression_bootstrap_results.csv',
                    'classification/output/step_4_ensembles/EnsembleTop1_bootstrap_results.csv',
                    'classification/output/step_4_ensembles/EnsembleTop2_bootstrap_results.csv',
                    'classification/output/step_4_ensembles/EnsembleTop5_bootstrap_results.csv'
                    ]
    classifier_names_LR = ['LR',
                        'ENS1',
                        'ENS2',
                        'ENS5'
                        ]

    report_paths = ['classification/output/step_2_0_ncv_multiple_algorithms/LinearSVC_bootstrap_results.csv',
                    'classification/output/step_2_0_ncv_multiple_algorithms/LogisticRegression_bootstrap_results.csv',
                    'classification/output/step_2_0_ncv_multiple_algorithms/RandomForestClassifier_bootstrap_results.csv',
                    'classification/output/step_2_0_ncv_multiple_algorithms/MultinomialNB_bootstrap_results.csv',
                    'classification/output/step_4_ensembles/EnsembleTop1_bootstrap_results.csv',
                    'classification/output/step_4_ensembles/EnsembleTop2_bootstrap_results.csv',
                    'classification/output/step_4_ensembles/EnsembleTop5_bootstrap_results.csv'
                    ]
    classifier_names = ['SVM',
                        'LR',
                        'RF',
                        'MNB',
                        'ENS1',
                        'ENS2',
                        'ENS5'
                        ]
    report_paths_RF = ['classification/output/step_2_0_ncv_multiple_algorithms/RandomForestClassifier_bootstrap_results.csv',
                    'classification/output/step_4_ensembles/EnsembleTop1_bootstrap_results.csv',
                    'classification/output/step_4_ensembles/EnsembleTop2_bootstrap_results.csv',
                    'classification/output/step_4_ensembles/EnsembleTop5_bootstrap_results.csv'
                    ]
    classifier_names_RF = ['RF',
                        'ENS1',
                        'ENS2',
                        'ENS5'
                        ]
    report_paths_MNB = ['classification/output/step_2_0_ncv_multiple_algorithms/MultinomialNB_bootstrap_results.csv',
                    'classification/output/step_4_ensembles/EnsembleTop1_bootstrap_results.csv',
                    'classification/output/step_4_ensembles/EnsembleTop2_bootstrap_results.csv',
                    'classification/output/step_4_ensembles/EnsembleTop5_bootstrap_results.csv'
                    ]
    classifier_names_MNB = ['MNB',
                        'ENS1',
                        'ENS2',
                        'ENS5'
                        ]


    bootstrap_results_df = get_bootstrap_results(report_paths, classifier_names, metric='F1 macro average')
    bootstrap_results_df_LR = get_bootstrap_results(report_paths, classifier_names_LR, metric='F1 macro average')

    bootstrap_results_df.to_csv(output_path + 'step_4_bootstrap_results.csv')

    fig, ax = plt.subplots(figsize=(2.5, 3))
    plot_bootstrap_boxdiagram(fig, ax, "", "maF1", bootstrap_results_df_LR)
    plt.tight_layout()
    plt.ylim([0.4, 0.87])
    plt.savefig(output_path + 'step_4_f1_macro_average_bootsrap_boxplot_dense.pdf')
    plt.close()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(8, 3.4))
    plot_bootstrap_boxdiagram(fig, ax1, "Concurrency", "F1", get_bootstrap_results(report_paths_LR, classifier_names_LR, metric='F1_concurrency'))
    plot_bootstrap_boxdiagram(fig, ax2, "Memory", "", get_bootstrap_results(report_paths_RF, classifier_names_RF, metric='F1_memory'))
    plot_bootstrap_boxdiagram(fig, ax3, "Other", "", get_bootstrap_results(report_paths_MNB, classifier_names_MNB, metric='F1_other'))
    plot_bootstrap_boxdiagram(fig, ax4, "Semantic", "", get_bootstrap_results(report_paths_LR, classifier_names_LR, metric='F1_semantic'))
    plt.tight_layout()
    plt.ylim([0.4, 0.87])
    plt.savefig(output_path + 'step_4_class_specific_f1_bootsrap_boxplot_dense.pdf')
    plt.close()


def plot_bootstrap_boxdiagram(fig, ax, title, metric, bootstrap_results_df, widths=None):
    boxes = []
    colors = []
    for index, row in bootstrap_results_df.sort_values('label').iterrows():
        box = {
            'label': row['label'],
            'whislo': row['lower']/100,
            'q1': row['lower']/100,
            'med': row['mean'],
            'q3': row['upper']/100,
            'whishi': row['upper']/100,
            'fliers': []
        }
        if row['label'].lower().startswith('ens') or row['label'].lower().startswith(' weighted'):
            colors.append('lightblue')
        else:
            colors.append('white')
        boxes.append(box)

    boxplot = ax.bxp(boxes, showfliers=False, patch_artist=True, medianprops=dict(color="black", linewidth=1.5), widths=widths)

    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(metric)
    ax.set_title(title)
    plt.sca(ax)
    # plt.xticks(rotation=45)



def main():
    step_1()
    step_2('classification/output/step_2_0_ncv_multiple_algorithms/', 'step_2_0_ncv_multiple_algorithms')
    step_4_dense()


if __name__ == '__main__':
    main()
