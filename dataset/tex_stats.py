import json

import pandas

from dataset.LoadGithubIssuesDataframes import get_filtered_issues
from file_anchor import root_dir
OUTPUT_DIR = root_dir() + 'dataset/dataset_stats/'

tex_var_format = '\\newcommand{\\%s}{%s\\xspace}\n'

target_names = {'concurrency': 0, 'memory': 1, 'other': 2, 'semantic': 3}

def load_and_normalize_csv(path):
    df = pandas.read_csv(path)
    # df = df[~df['RootCause'].isnull()]
    df = df[['url', 'RootCause']]
    df['RootCause'] = df['RootCause'].fillna('')
    df['RootCause'] = df['RootCause'].str.lower()
    df['RootCause'] = df['RootCause'].str.strip()
    # df.loc[~df['RootCause'].isin(['semantic', 'memory', 'concurrency', 'other']), 'RootCause'] = np.nan
    df['ProjectName'] = df['url'].apply(lambda x: x.split('/')[4])
    # df['ConfidenceNumeric'] = pandas.to_numeric(df['Confidence'])
    return df



def full_github_issue_dataset_stats():
    df_unfilterd = pandas.read_csv(root_dir() + 'dataset/github_issues_dataset.csv.zip', compression='zip')
    df_java_changes, df_other_changes = get_filtered_issues()

    with open(OUTPUT_DIR + 'dataset_numbers.tex', 'w') as fd:
        numRepos = '%d' % len(df_unfilterd['projectName'].value_counts())
        fd.write(tex_var_format % ('numRepos', numRepos))

        numFilteredIssues = '%d' % (len(df_java_changes) + len(df_other_changes))
        fd.write(tex_var_format % ('numFilteredIssues', numFilteredIssues))

        numFilteredIssuesJavaChanges = '%d' % len(df_java_changes)
        fd.write(tex_var_format % ('numFilteredIssuesJavaChanges', numFilteredIssuesJavaChanges))

        numFilteredIssuesOtherChanges = '%d' % len(df_other_changes)
        fd.write(tex_var_format % ('numFilteredIssuesOtherChanges', numFilteredIssuesOtherChanges))



def rater_1():
    keyword_df = load_and_normalize_csv(root_dir() + 'dataset/manualClassification/keyword_memory_concurrency_bugs.csv')
    rand_java_changes = load_and_normalize_csv(root_dir() + 'dataset/manualClassification/randomly_selected_bugs.csv')
    rand_other_changes = load_and_normalize_csv(root_dir() + 'dataset/manualClassification/randomly_selected_other.csv')

    with open(OUTPUT_DIR + 'rater_1_numbers.tex', 'w') as fd:
        keywordSearchTotal = '%d' % len(keyword_df)
        fd.write(tex_var_format % ('keywordSearchTotal', keywordSearchTotal))

        keywordSearchClassified = '%d' % len(keyword_df[keyword_df['RootCause'].isin(target_names.keys())])
        fd.write(tex_var_format % ('keywordSearchClassified', keywordSearchClassified))

        keywordSearchInvestigated = '%d' % len(keyword_df[keyword_df['RootCause'] != ''])
        fd.write(tex_var_format % ('keywordSearchInvestigated', keywordSearchInvestigated))

        keywordSearchExcluded = '%d' % (len(keyword_df[~keyword_df['RootCause'].isin(target_names.keys())]) - len(keyword_df[keyword_df['RootCause'] == '']))
        fd.write(tex_var_format % ('keywordSearchExcluded', keywordSearchExcluded))


        randJavaTotal = '%d' % len(rand_java_changes)
        fd.write(tex_var_format % ('randJavaTotal', randJavaTotal))

        randJavaClassified = '%d' % len(rand_java_changes[rand_java_changes['RootCause'].isin(target_names.keys())])
        fd.write(tex_var_format % ('randJavaClassified', randJavaClassified))

        randJavaInvestigated = '%d' % len(rand_java_changes[rand_java_changes['RootCause'] != ''])
        fd.write(tex_var_format % ('randJavaInvestigated', randJavaInvestigated))

        randJavaExcluded = '%d' % (len(rand_java_changes[~rand_java_changes['RootCause'].isin(target_names.keys())]) - len(rand_java_changes[rand_java_changes['RootCause'] == '']))
        fd.write(tex_var_format % ('randJavaExcluded', randJavaExcluded))


        randOtherTotal = '%d' % len(rand_other_changes)
        fd.write(tex_var_format % ('randOtherTotal', randOtherTotal))

        randOtherClassified = '%d' % len(rand_other_changes[rand_other_changes['RootCause'].isin(target_names.keys())])
        fd.write(tex_var_format % ('randOtherClassified', randOtherClassified))

        randOtherInvestigated = '%d' % len(rand_other_changes[rand_other_changes['RootCause'] != ''])
        fd.write(tex_var_format % ('randOtherInvestigated', randOtherInvestigated))

        randOtherExcluded = '%d' % (len(rand_other_changes[~rand_other_changes['RootCause'].isin(target_names.keys())]) - len(rand_other_changes[rand_other_changes['RootCause'] == '']))
        fd.write(tex_var_format % ('randOtherExcluded', randOtherExcluded))


        numTotalClassifiedReseracherOne = '%d' % (
            len(rand_other_changes[rand_other_changes['RootCause'].isin(target_names.keys())]) +
            len(rand_java_changes[rand_java_changes['RootCause'].isin(target_names.keys())]) +
            len(keyword_df[keyword_df['RootCause'].isin(target_names.keys())]))
        fd.write(tex_var_format % ('numTotalClassifiedReseracherOne', numTotalClassifiedReseracherOne))

        numTotalInvestigatedReseracherOne = '%d' % (
            len(rand_other_changes[rand_other_changes['RootCause'] != '']) +
            len(rand_java_changes[rand_java_changes['RootCause'] != '']) +
            len(keyword_df[keyword_df['RootCause'] != '']))
        fd.write(tex_var_format % ('numTotalInvestigatedReseracherOne', numTotalInvestigatedReseracherOne))

        df = pandas.DataFrame()
        df = df.append(rand_other_changes)
        df = df.append(rand_java_changes)
        df = df.append(keyword_df)
        df = df[df['RootCause'] != '']
        numReposInvestigatedReseracherOne = '%d' % len(df['ProjectName'].value_counts())
        fd.write(tex_var_format % ('numReposInvestigatedReseracherOne', numReposInvestigatedReseracherOne))

        numReposClassifiedReseracherOne = '%d' % len(df[df['RootCause'].isin(target_names)]['ProjectName'].value_counts())
        fd.write(tex_var_format % ('numReposClassifiedReseracherOne', numReposClassifiedReseracherOne))

        numExcludedIssuesReseracherOne = '%d' % len(df[~df['RootCause'].isin(target_names)])
        fd.write(tex_var_format % ('numExcludedIssuesReseracherOne', numExcludedIssuesReseracherOne))

        numConcurrencyReseracherOne = '%d' % df['RootCause'].value_counts()['concurrency']
        fd.write(tex_var_format % ('numConcurrencyReseracherOne', numConcurrencyReseracherOne))
        numMemoryReseracherOne = '%d' % df['RootCause'].value_counts()['memory']
        fd.write(tex_var_format % ('numMemoryReseracherOne', numMemoryReseracherOne))
        numOtherReseracherOne = '%d' % df['RootCause'].value_counts()['other']
        fd.write(tex_var_format % ('numOtherReseracherOne', numOtherReseracherOne))
        numSemanticReseracherOne = '%d' % df['RootCause'].value_counts()['semantic']
        fd.write(tex_var_format % ('numSemanticReseracherOne', numSemanticReseracherOne))


def actual_trainingset_stats():
    df = pandas.read_csv(root_dir() + 'dataset/classification_dataset.csv.zip', compression='zip')

    with open(OUTPUT_DIR + 'trainingset_numbers.tex', 'w') as fd:
        numReposInTrainingSet = '%d' % len(df['projectName'].value_counts())
        fd.write(tex_var_format % ('numReposInTrainingSet', numReposInTrainingSet))

        numSizeTrainingSet = '%d' % len(df)
        fd.write(tex_var_format % ('numSizeTrainingSet', numSizeTrainingSet))

        trainingSetNumConcurrencyReseracherOne = '%d' % df['RootCause'].value_counts()['concurrency']
        fd.write(tex_var_format % ('trainingSetNumConcurrencyReseracherOne', trainingSetNumConcurrencyReseracherOne))
        trainingSetNumMemoryReseracherOne = '%d' % df['RootCause'].value_counts()['memory']
        fd.write(tex_var_format % ('trainingSetNumMemoryReseracherOne', trainingSetNumMemoryReseracherOne))
        trainingSetNumOtherReseracherOne = '%d' % df['RootCause'].value_counts()['other']
        fd.write(tex_var_format % ('trainingSetNumOtherReseracherOne', trainingSetNumOtherReseracherOne))
        trainingSetNumSemanticReseracherOne = '%d' % df['RootCause'].value_counts()['semantic']
        fd.write(tex_var_format % ('trainingSetNumSemanticReseracherOne', trainingSetNumSemanticReseracherOne))


def main():
    training_set_bug_report_lenghts()
    full_github_issue_dataset_stats()
    rater_1()
    actual_trainingset_stats()


def training_set_bug_report_lenghts():
    df = pandas.read_csv(root_dir() + 'dataset/classification_dataset.csv.zip', compression='zip')
    df['body'] = df['body'].fillna('')
    df['title'] = df['title'].fillna('')

    targets = df['RootCause'].value_counts().index.to_list()
    targets.sort()
    targets = {cause: i for i, cause in enumerate(targets)}
    df['target'] = df['RootCause'].replace(targets)
    df['doc'] = df["title"] + '\n' + df["body"]

    df['doc_len'] = df['doc'].str.len()
    df = df.sort_values(by='doc_len')

    df[['url', 'doc_len', 'RootCause']].to_csv(OUTPUT_DIR + 'bug_report_lengths.csv')


if __name__ == '__main__':
    main()
