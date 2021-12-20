import os
import pathlib

import numpy as np
import pandas

from classification.cross_project.cp_Z_plot_evaluation import load_cross_project_evaluation
from classification.stats_utils import t_test_x_greater_y, t_test_x_differnt_y
from file_anchor import root_dir
from helpers.Logger import Logger


log = Logger()
output_path = root_dir() + 'classification/cross_project/output/' + pathlib.Path(__file__).stem + '/'
os.makedirs(output_path, exist_ok=True)


def cross_project():
    classifier_names, cross_project_scores = load_cross_project_evaluation()

    x_proj_lsvc = cross_project_scores[cross_project_scores['classifier'] == 'LinearSVC'].copy()
    x_proj_lrc = cross_project_scores[cross_project_scores['classifier'] == 'LRC'].copy()
    x_proj_rfc = cross_project_scores[cross_project_scores['classifier'] == 'RFC'].copy()

    x_proj_ens1 = cross_project_scores[cross_project_scores['classifier'] == 'Ensemble Top 1'].copy()
    x_proj_ens2 = cross_project_scores[cross_project_scores['classifier'] == 'Ensemble Top 2'].copy()
    x_proj_ens5 = cross_project_scores[cross_project_scores['classifier'] == 'Ensemble Top 5'].copy()


    print('ens1 greater than lsvc?')
    df = t_test_x_greater_y(x_proj_ens1['F1 macro average'], x_proj_lsvc['F1 macro average'], 'xEnsemble1', 'xLSVC', output_path, 'cross_proj_ens1_vs_lsvc.csv')
    print(df.to_string())

    print('ens1 greater than lrc?')
    df = t_test_x_greater_y(x_proj_ens1['F1 macro average'], x_proj_lrc['F1 macro average'], 'xEnsemble1', 'xLRC', output_path, 'cross_proj_ens1_vs_lrc.csv')
    print(df.to_string())

    print('ens1 different to ens2?')
    df = t_test_x_differnt_y(x_proj_ens2['F1 macro average'], x_proj_ens1['F1 macro average'], 'xEnsemble2', 'xEnsemble1', output_path, 'cross_proj_ens2_vs_ens1.csv')
    print(df.to_string())

    print('ens1 different to ens5?')
    df = t_test_x_differnt_y(x_proj_ens5['F1 macro average'], x_proj_ens1['F1 macro average'], 'xEnsemble5', 'xEnsemble1', output_path, 'cross_proj_ens5_vs_ens1.csv')
    print(df.to_string())

    print('ens2 greater than ens1?')
    df = t_test_x_greater_y(x_proj_ens2['F1 macro average'], x_proj_ens1['F1 macro average'], 'xEnsemble2', 'xEnsemble1', output_path, 'cross_proj_ens2_gt_ens1.csv')
    print(df.to_string())

    print('ens5 greater than ens2?')
    df = t_test_x_greater_y(x_proj_ens5['F1 macro average'], x_proj_ens2['F1 macro average'], 'xEnsemble5', 'xEnsemble2', output_path, 'cross_proj_ens5_gt_ens2.csv')
    print(df.to_string())



def main():
    cross_project()


if __name__ == '__main__':
    main()
