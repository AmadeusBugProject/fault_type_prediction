import pandas

DUMP_KEY_ISSUE_STATS_SKIPPED_REASON_TOO_MANY_COMMITS = "tooManyCommits"
DUMP_KEY_ISSUE_STATS_SKIPPED_REASON_TOO_MANY_FILES = "tooManyFiles"
DUMP_KEY_ISSUE_STATS_SKIPPED_REASON_TOO_MANY_CHANGES = "tooManyChanges"
DUMP_KEY_ISSUE_COMMIT_DETAILS = "commitsDetails"
DUMP_KEY_ISSUE_COMMIT_DETAILS_GH_USER = "commitUser"
DUMP_KEY_ISSUE_USER = "user"


def drop_keys(issues, keys):
    issues_red = issues.copy()
    for issue in issues_red:
        for key in keys:
            issue.pop(key)
    return issues_red


def only_bugs_with_valid_commit_set_and_java_code_changes(issues_df):
    if 'statsSkippedReason' not in issues_df.columns:
        issues_df['statsSkippedReason'] = ''
    if 'spoonStatsSummary.TOT' not in issues_df.columns:
        issues_df['spoonStatsSummary.TOT'] = 0
    if 'filteredCommitsReason.alsoFixesPhrase' not in issues_df.columns:
        issues_df['filteredCommitsReason.alsoFixesPhrase'] = 0
    if 'filteredCommitsReason.moreThanOneParent' not in issues_df.columns:
        issues_df['filteredCommitsReason.moreThanOneParent'] = 0
    if 'filteredCommitsReason.multipleIssueFixes' not in issues_df.columns:
        issues_df['filteredCommitsReason.multipleIssueFixes'] = 0
    if 'filteredCommitsReason.unavailable' not in issues_df.columns:
        issues_df['filteredCommitsReason.unavailable'] = 0

    return issues_df[(issues_df['filteredCommitsReason.alsoFixesPhrase'] == 0) &
                     (issues_df['filteredCommitsReason.moreThanOneParent'] == 0) &
                     (issues_df['filteredCommitsReason.multipleIssueFixes'] == 0) &
                     (issues_df['filteredCommitsReason.unavailable'] == 0) &
                     (issues_df['spoonStatsSummary.TOT'] != 0) &
                     (issues_df['spoonStatsSummary.TOT'].astype(str) != 'nan') &
                     (issues_df['statsSkippedReason'].astype(str) == '')]


def only_bugs_with_valid_commit_set(issues_df):
    if 'statsSkippedReason' not in issues_df.columns:
        issues_df['statsSkippedReason'] = ''
    if 'spoonStatsSummary.TOT' not in issues_df.columns:
        issues_df['spoonStatsSummary.TOT'] = 0
    if 'filteredCommitsReason.alsoFixesPhrase' not in issues_df.columns:
        issues_df['filteredCommitsReason.alsoFixesPhrase'] = 0
    if 'filteredCommitsReason.moreThanOneParent' not in issues_df.columns:
        issues_df['filteredCommitsReason.moreThanOneParent'] = 0
    if 'filteredCommitsReason.multipleIssueFixes' not in issues_df.columns:
        issues_df['filteredCommitsReason.multipleIssueFixes'] = 0
    if 'filteredCommitsReason.unavailable' not in issues_df.columns:
        issues_df['filteredCommitsReason.unavailable'] = 0

    too_big = [DUMP_KEY_ISSUE_STATS_SKIPPED_REASON_TOO_MANY_COMMITS,
               DUMP_KEY_ISSUE_STATS_SKIPPED_REASON_TOO_MANY_FILES,
               DUMP_KEY_ISSUE_STATS_SKIPPED_REASON_TOO_MANY_CHANGES]

    return issues_df[(issues_df['filteredCommitsReason.alsoFixesPhrase'] == 0) &
                     (issues_df['filteredCommitsReason.moreThanOneParent'] == 0) &
                     (issues_df['filteredCommitsReason.multipleIssueFixes'] == 0) &
                     (issues_df['filteredCommitsReason.unavailable'] == 0) &
                     (issues_df['gitStatsSummary.gitFilesChange'] != 0) &
                     (issues_df['gitStatsSummary.gitFilesChange'].astype(str) != 'nan') &
                     (issues_df['statsSkippedReason'].isna()) &
                     ~(issues_df['commitsDetails'].astype(str).str.contains('|'.join(too_big)))]


def remove_pull_request_issues(issues_df):
    # return issues_df[issues_df['filteredCommitsReason.mergeCommitUsed'] == 0]
    return issues_df[issues_df['url'].str.contains('/issues/', regex=False)]


def remove_reporter_is_committer_issues(issues_df):
    df = issues_df.explode(DUMP_KEY_ISSUE_COMMIT_DETAILS)
    df = pandas.concat([df.drop([DUMP_KEY_ISSUE_COMMIT_DETAILS], axis=1), df[DUMP_KEY_ISSUE_COMMIT_DETAILS].apply(pandas.Series)], axis=1)
    report_is_committer_df = df[df[DUMP_KEY_ISSUE_COMMIT_DETAILS_GH_USER] == df[DUMP_KEY_ISSUE_USER]]
    report_is_committer_ser = report_is_committer_df['url'].value_counts()
    return issues_df[~issues_df['url'].isin(report_is_committer_ser.index)]
