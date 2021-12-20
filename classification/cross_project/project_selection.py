import json
import os
import pathlib

import pandas

from file_anchor import root_dir
from helpers.Logger import Logger

log = Logger()
output_path = root_dir() + 'classification/cross_project/output/' + pathlib.Path(__file__).stem + '/'
os.makedirs(output_path, exist_ok=True)


def main():
    project_combos = []

    df = pandas.read_csv(root_dir() + 'dataset/classification_dataset.csv.zip', compression='zip')
    num_combos = 0
    while num_combos < 100:
        projects = []
        while True:
            random_project = df['projectName'].sample().tolist()[0]
            while random_project in projects:
                random_project = df['projectName'].sample().tolist()[0]
            projects.append(random_project)
            projects.sort()

            total_items = len(df[df['projectName'].isin(projects)])
            if total_items > 100:
                break

            targets = df[df['projectName'].isin(projects)]['RootCause'].value_counts().sort_index().to_list()
            if stratified_and_size_limit(targets):
                if projects in project_combos:
                    log.s('duplicate: ' + str(projects))
                    break

                project_combos.append(projects)
                update_file({'num_items': total_items, 'projects': projects, 'targets': targets})
                num_combos += 1
                log.s(str(num_combos) + '  ' + str(projects))
                break


def update_file(data):
    file_path = output_path + 'project_combos.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            j_data = json.load(file)
    else:
        j_data = []
    j_data.append(data)
    with open(file_path, 'w') as file:
        json.dump(j_data, file, indent=4)


def stratified_and_size_limit(targets):
    min_items = 90
    max_items = 100
    epsilon = 0.5

    if len(targets) != 4 or not min_items <= sum(targets) <= max_items:
        return False
    mean = sum(targets)/4.0
    return all(mean - epsilon <= i <= mean + epsilon for i in targets)


if __name__ == "__main__":
    main()
