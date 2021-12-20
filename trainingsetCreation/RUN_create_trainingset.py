
from file_anchor import root_dir
from trainingsetCreation.InputDataCreator import InputDataCreator


def main():
    idc = InputDataCreator()
    df = idc.get_traing_set_with_others_as_pandas_df()

    df = df[['projectName', 'RootCause', 'title', 'body', 'url']]
    df.to_csv(root_dir() + 'dataset/classification_dataset.csv.zip', compression='zip')


if __name__ == "__main__":
    main()


