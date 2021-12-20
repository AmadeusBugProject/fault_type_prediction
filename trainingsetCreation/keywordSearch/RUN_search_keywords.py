import pandas

from dataset.LoadGithubIssuesDataframes import get_all_issues_df
from dataset.excluded import excluded_projects
from file_anchor import root_dir
from trainingsetCreation import Filters
from trainingsetCreation.keywordSearch.constants import KEYWORDS_MEMORY_RE, KEYWORDS_CONCURRENCY_RE, \
    KEYWORDS_IMPACT_RE

DUMP_KEY_ISSUE_URL = "url"
DUMP_KEY_ISSUE_COMMIT_DETAILS = "commitsDetails"
DUMP_KEY_ISSUE_COMMIT_DETAILS_MESSAGE = "commitMessage"


def main():
    write_keyword_search_results_to_file()


def write_keyword_search_results_to_file():
    df_all = get_all_issues_df()
    df_all = Filters.remove_pull_request_issues(df_all)
    df_all = Filters.only_bugs_with_valid_commit_set(df_all)

    df_all = df_all[~df_all['projectName'].isin(excluded_projects)].copy()

    df_java_changes = df_all[(df_all['spoonStatsSummary.TOT'] != 0) &
                             (df_all['spoonStatsSummary.TOT'].astype(str) != 'nan')]

    df_other_changes = df_all[(df_all['spoonStatsSummary.TOT'] == 0) |
                             (df_all['spoonStatsSummary.TOT'].astype(str) == 'nan')]

    df = df_java_changes.explode(DUMP_KEY_ISSUE_COMMIT_DETAILS)
    df = pandas.concat([df.drop([DUMP_KEY_ISSUE_COMMIT_DETAILS], axis=1), df[DUMP_KEY_ISSUE_COMMIT_DETAILS].apply(pandas.Series)], axis=1)

    memory_df = df[df[DUMP_KEY_ISSUE_COMMIT_DETAILS_MESSAGE].str.contains('|'.join(KEYWORDS_MEMORY_RE), case=False)]
    memory_url_ser = memory_df[DUMP_KEY_ISSUE_URL].value_counts()

    concurrency_df = df[df[DUMP_KEY_ISSUE_COMMIT_DETAILS_MESSAGE].str.contains('|'.join(KEYWORDS_CONCURRENCY_RE), case=False)]
    concurrency_url_ser = concurrency_df[DUMP_KEY_ISSUE_URL].value_counts()

    impact_df = df[df[DUMP_KEY_ISSUE_COMMIT_DETAILS_MESSAGE].str.contains('|'.join(KEYWORDS_IMPACT_RE), case=False)]
    impact_url_ser = impact_df[DUMP_KEY_ISSUE_URL].value_counts()

    all_keywords = []
    all_keywords.extend(KEYWORDS_MEMORY_RE)
    all_keywords.extend(KEYWORDS_IMPACT_RE)
    all_keywords.extend(KEYWORDS_CONCURRENCY_RE)

    random_any_other_df = df[~df[DUMP_KEY_ISSUE_COMMIT_DETAILS_MESSAGE].str.contains('|'.join(all_keywords), case=False)]
    random_any_other_url_ser = random_any_other_df.sample(120)[DUMP_KEY_ISSUE_URL].value_counts()

    other_url_ser = df_other_changes.sample(250)[DUMP_KEY_ISSUE_URL].value_counts()

    print('\nsearching in commit message:')
    print('memory ' + str(memory_df.shape[0]) + ' unique ' + str(memory_url_ser.shape[0]))
    print('concurrency ' + str(concurrency_df.shape[0]) + ' unique ' + str(concurrency_url_ser.shape[0]))
    print('suspicious impacts ' + str(impact_df.shape[0]) + ' unique ' + str(impact_url_ser.shape[0]))

    impact_url_ser.to_csv(root_dir() + 'trainingsetCreation/keywordSearch/data/impacts.csv')
    memory_url_ser.to_csv(root_dir() + 'trainingsetCreation/keywordSearch/data/memory.csv')
    concurrency_url_ser.to_csv(root_dir() + 'trainingsetCreation/keywordSearch/data/concurrency.csv')
    random_any_other_url_ser.to_csv(root_dir() + 'trainingsetCreation/keywordSearch/data/random.csv')
    other_url_ser.to_csv(root_dir() + 'trainingsetCreation/keywordSearch/data/other2.csv')


if __name__ == "__main__":
    main()
