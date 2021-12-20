import json

import pandas

from dataset.excluded import excluded_projects
from file_anchor import root_dir
from trainingsetCreation import Filters


def get_all_issues_df():
    df = pandas.read_csv(root_dir() + 'dataset/github_issues_dataset.csv.zip', compression='zip')
    df['commitsDetails'] = df['commitsDetails'].apply(json.loads)
    df['labels'] = df['labels'].apply(json.loads)
    df['changesInPackagesSPOON'] = df['changesInPackagesSPOON'].fillna('[]')
    df['changesInPackagesSPOON'] = df['changesInPackagesSPOON'].apply(json.loads)
    df['filteredCommits'] = df['filteredCommits'].apply(json.loads)
    df['changesInPackagesGIT'] = df['changesInPackagesGIT'].apply(json.loads)
    return df


def get_filtered_issues():
    df_all = get_all_issues_df()
    df_all = Filters.remove_pull_request_issues(df_all)
    df_all = Filters.only_bugs_with_valid_commit_set(df_all)

    # df_all = df_all[~df_all['projectName'].isin(excluded_projects)].copy()

    df_java_changes = df_all[(df_all['spoonStatsSummary.TOT'] != 0) &
                             (df_all['spoonStatsSummary.TOT'].astype(str) != 'nan')]

    df_other_changes = df_all[(df_all['spoonStatsSummary.TOT'] == 0) |
                             (df_all['spoonStatsSummary.TOT'].astype(str) == 'nan')]

    return df_java_changes, df_other_changes


def get_classification_df():
    return pandas.read_csv(root_dir() + 'dataset/classification_dataset.csv.zip', compression='zip')