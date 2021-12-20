import os
import pathlib

import numpy as np
import pandas

from classification.stats_utils import t_test_x_greater_y, t_test_x_differnt_y
from file_anchor import root_dir
from helpers.Logger import Logger


log = Logger()
output_path = root_dir() + 'classification/output/' + pathlib.Path(__file__).stem + '/'
os.makedirs(output_path, exist_ok=True)

def step_1():
    step_1_w_o_artifact_removal = pandas.read_csv(root_dir() + 'classification/output/step_1_0_trivial_approach/LinearSVC_bootstrap_iterations.csv')
    step_1_w_artifact_removal = pandas.read_csv(root_dir() + 'classification/output/step_1_1_trivial_approach_with_artifact_replacement/LinearSVC_bootstrap_iterations.csv')
    df = t_test_x_greater_y(step_1_w_artifact_removal['F1 macro average'], step_1_w_o_artifact_removal['F1 macro average'], 'artifact_removal', 'no_artifact_removal', output_path, 'step_1_artifact_detection_vs_no_artifact_detection.csv')

    print('artifact removal vs no artifact removal MEMORY')
    df = t_test_x_greater_y(step_1_w_artifact_removal['F1_memory'], step_1_w_o_artifact_removal['F1_memory'], 'artifact_removal', 'no_artifact_removal', output_path, 'step_1_artifact_detection_vs_no_artifact_detection_memory.csv')
    print(df.to_string())

    print('artifact removal vs no artifact removal SEMANTIC')
    df = t_test_x_greater_y(step_1_w_o_artifact_removal['F1_semantic'], step_1_w_artifact_removal['F1_semantic'], 'artifact_removal', 'no_artifact_removal', output_path, 'step_1_no_artifact_detection_vs_artifact_detection_semantic.csv')
    print(df.to_string())


def step_2():
    step_2_lsvc = pandas.read_csv(root_dir() + 'classification/output/step_2_0_ncv_multiple_algorithms/LinearSVC_bootstrap_iterations.csv')
    step_2_rfc = pandas.read_csv(root_dir() + 'classification/output/step_2_0_ncv_multiple_algorithms/RandomForestClassifier_bootstrap_iterations.csv')
    step_2_lrc = pandas.read_csv(root_dir() + 'classification/output/step_2_0_ncv_multiple_algorithms/LogisticRegression_bootstrap_iterations.csv')
    step_2_mnb = pandas.read_csv(root_dir() + 'classification/output/step_2_0_ncv_multiple_algorithms/MultinomialNB_bootstrap_iterations.csv')

    print('mnb vs lsvc')
    df = t_test_x_differnt_y(step_2_mnb['F1 macro average'], step_2_lsvc['F1 macro average'], 'MNB', 'LSVC', output_path, 'step_2_mnb_vs_lsvc.csv')
    print(df.to_string())

    print('lsvc vs rfc')
    df = t_test_x_differnt_y(step_2_lsvc['F1 macro average'], step_2_rfc['F1 macro average'], 'LSVC', 'RFC', output_path, 'step_2_lsvc_vs_rfc.csv')
    print(df.to_string())

    print('rfc vs lrc')
    df = t_test_x_differnt_y(step_2_rfc['F1 macro average'], step_2_lrc['F1 macro average'], 'RFC', 'LRC', output_path, 'step_2_rfc_vs_lrc.csv')
    print(df.to_string())


def step_2_classes():
    step_2_lsvc = pandas.read_csv(root_dir() + 'classification/output/step_2_0_ncv_multiple_algorithms/LinearSVC_bootstrap_iterations.csv')
    step_2_rfc = pandas.read_csv(root_dir() + 'classification/output/step_2_0_ncv_multiple_algorithms/RandomForestClassifier_bootstrap_iterations.csv')

    df = t_test_x_greater_y(step_2_rfc['F1_memory'], step_2_lsvc['F1_memory'], 'RFC memory', 'LSVC memory', output_path, 'step_2_lsvc_vs_rfc_memory.csv')
    df = t_test_x_greater_y(step_2_lsvc['F1_other'], step_2_rfc['F1_other'], 'LSVC other', 'RFC other', output_path, 'step_2_rfc_vs_lsvc_other.csv')


def step_4():
    step_2_lsvc = pandas.read_csv(root_dir() + 'classification/output/step_2_0_ncv_multiple_algorithms/LinearSVC_bootstrap_iterations.csv')
    step_2_lrc = pandas.read_csv(root_dir() + 'classification/output/step_2_0_ncv_multiple_algorithms/LogisticRegression_bootstrap_iterations.csv')

    step_4_ens1 = pandas.read_csv(root_dir() + 'classification/output/step_4_ensembles/EnsembleTop1_bootstrap_iterations.csv')
    step_4_ens2 = pandas.read_csv(root_dir() + 'classification/output/step_4_ensembles/EnsembleTop2_bootstrap_iterations.csv')
    step_4_ens5 = pandas.read_csv(root_dir() + 'classification/output/step_4_ensembles/EnsembleTop5_bootstrap_iterations.csv')

    print('ens1 greater than lsvc?')
    df = t_test_x_greater_y(step_4_ens1['F1 macro average'], step_2_lsvc['F1 macro average'], 'Ensemble1', 'LSVC w Artifact removal', output_path, 'step_4_ens1_vs_lsvc.csv')
    print(df.to_string())

    print('ens1 greater than lrc?')
    df = t_test_x_greater_y(step_4_ens1['F1 macro average'], step_2_lrc['F1 macro average'], 'Ensemble1', 'LRC', output_path, 'step_4_ens1_vs_lrc.csv')
    print(df.to_string())

    print('ens1 different to ens2?')
    df = t_test_x_differnt_y(step_4_ens2['F1 macro average'], step_4_ens1['F1 macro average'], 'Ensemble2', 'Ensemble1', output_path, 'step_4_ens2_vs_ens1.csv')
    print(df.to_string())

    print('ens1 different to ens5?')
    df = t_test_x_differnt_y(step_4_ens5['F1 macro average'], step_4_ens1['F1 macro average'], 'Ensemble5', 'Ensemble1', output_path, 'step_4_ens5_vs_ens1.csv')
    print(df.to_string())


def step_4_classes():
    step_2_lrc = pandas.read_csv(root_dir() + 'classification/output/step_2_0_ncv_multiple_algorithms/LogisticRegression_bootstrap_iterations.csv')
    step_2_rfc = pandas.read_csv(root_dir() + 'classification/output/step_2_0_ncv_multiple_algorithms/RandomForestClassifier_bootstrap_iterations.csv')
    step_4_ens1 = pandas.read_csv(root_dir() + 'classification/output/step_4_ensembles/EnsembleTop1_bootstrap_iterations.csv')

    print('ens1 greater than lrc concurrency?')
    df = t_test_x_greater_y(step_4_ens1['F1_concurrency'], step_2_lrc['F1_concurrency'], 'Ensemble1 concurrency', 'LRC concurrency', output_path, 'step_4_ens1_vs_lrc_concurrency.csv')
    print(df.to_string())

    print('ens1 greater than lrc memory?')
    df = t_test_x_greater_y(step_4_ens1['F1_memory'], step_2_lrc['F1_memory'], 'Ensemble1 memory', 'LRC memory', output_path, 'step_4_ens1_vs_lrc_memory.csv')
    print(df.to_string())

    print('ens1 greater than lrc other?')
    df = t_test_x_greater_y(step_4_ens1['F1_other'], step_2_lrc['F1_other'], 'Ensemble1 other', 'LRC other', output_path, 'step_4_ens1_vs_lrc_other.csv')
    print(df.to_string())

    print('ens1 greater than lrc semantic?')
    df = t_test_x_greater_y(step_4_ens1['F1_semantic'], step_2_lrc['F1_semantic'], 'Ensemble1 semantic', 'LRC semantic', output_path, 'step_4_ens1_vs_lrc_semantic.csv')
    print(df.to_string())

    print('ens1 different than lrc concurrency?')
    df = t_test_x_differnt_y(step_4_ens1['F1_concurrency'], step_2_lrc['F1_concurrency'], 'Ensemble1 concurrency', 'LRC concurrency', output_path, 'step_4_ens1_different_lrc_concurrency.csv')
    print(df.to_string())

    print('ens1 different than rfc memory?')
    df = t_test_x_differnt_y(step_4_ens1['F1_memory'], step_2_rfc['F1_memory'], 'Ensemble1 memory', 'RFC memory', output_path, 'step_4_ens1_different_rfc_memory.csv')
    print(df.to_string())

    print('ens1 greater than rfc memory?')
    df = t_test_x_greater_y(step_4_ens1['F1_memory'], step_2_rfc['F1_memory'], 'Ensemble1 memory', 'RFC memory', output_path, 'step_4_ens1_vs_rfc_memory.csv')
    print(df.to_string())

    print('rfc greater than ens1 memory?')
    df = t_test_x_greater_y(step_2_rfc['F1_memory'], step_4_ens1['F1_memory'], 'RFC memory', 'Ensemble1 memory', output_path, 'step_4_rfc_vs_ens1_memory.csv')
    print(df.to_string())


def main():
    step_1()
    step_2()
    step_2_classes()
    step_4()
    step_4_classes()


if __name__ == '__main__':
    main()
