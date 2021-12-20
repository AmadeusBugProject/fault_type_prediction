import json
import os
import pathlib
from itertools import chain

import pandas

from file_anchor import root_dir

output_path = root_dir() + 'classification/cross_project/output/' + pathlib.Path(__file__).stem + '/'
os.makedirs(output_path, exist_ok=True)


def main():
    with open(root_dir() + 'classification/cross_project/output/project_selection/project_combos.json', 'r') as fd:
        project_combos = json.load(fd)

    projects_df = pandas.DataFrame({'projectName': list(chain.from_iterable([x['projects'] for x in project_combos]))})
    splits_df = pandas.DataFrame({'projectName': projects_df['projectName'].value_counts().index, 'projectOccurrencesInProjectSplits': projects_df['projectName'].value_counts().values})

    dataset_df = pandas.read_csv(root_dir() + 'dataset/classification_dataset.csv.zip', compression='zip')
    orig_df = pandas.DataFrame({'projectName': dataset_df['projectName'].value_counts().index, 'numBugReports': dataset_df['projectName'].value_counts().values})

    project_occurences_df = orig_df.merge(splits_df, on='projectName', how='outer')
    project_occurences_df.to_csv(output_path + 'count_project_occurences_in_splits.csv')


if __name__ == '__main__':
    main()
