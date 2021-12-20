import json

import pandas

from file_anchor import root_dir
from trainingsetCreation.InputDataCreator import InputDataCreator


def main():
    get_pandas_csv()


def get_pandas_csv():
    idc = InputDataCreator()
    df = pandas.DataFrame()
    df = df.append(idc.get_memory_df())
    df = df.append(idc.get_concurrency_df())
    df = df.append(idc.get_semantic_df())
    df = df.append(idc.get_others_df())
    sample_size = 250

    df.sample(sample_size)['url'].to_csv(root_dir() + 'trainingsetCreation/keywordSearch/data/random_sample.csv')


if __name__ == "__main__":
    main()


