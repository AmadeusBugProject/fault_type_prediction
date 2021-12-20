import os

import pandas

from file_anchor import root_dir
from helpers.Logger import Logger

log = Logger()


class InputDataCreator:
    def __init__(self):
        self.all_bugs_df = pandas.read_csv(root_dir() + 'dataset/github_issues_dataset.csv.zip', compression='zip')

    def load_csv(self, path):
        df = pandas.read_csv(path)
        df['RootCauseDetail'] = df['RootCauseDetail'].fillna('')
        df['RootCause'] = df['RootCause'].fillna('')
        df['ConfidenceNumeric'] = df['RootCause'].fillna(0)
        df['RootCauseDetail'] = df['RootCauseDetail'].str.lower()
        df['RootCauseDetail'] = df['RootCauseDetail'].str.strip()
        df['RootCause'] = df['RootCause'].str.lower()
        df['RootCause'] = df['RootCause'].str.strip()
        df['ConfidenceNumeric'] = pandas.to_numeric(df['Confidence'])
        return df

    def get_memory_df(self):
        print(os.system('ls'))
        csv_df = self.load_csv(root_dir() + 'dataset/manualClassification/keyword_memory_concurrency_bugs.csv')
        mem_df = csv_df[csv_df['RootCause'] == 'memory']
        log.s('keyword search memory: ' + str(mem_df.shape))

        rand_df = self.load_csv(root_dir() + 'dataset/manualClassification/randomly_selected_bugs.csv')
        rand_df = rand_df[rand_df['RootCause'] == 'memory']
        log.s('random selected memory: ' + str(rand_df.shape))

        mem_df = mem_df.append(rand_df, ignore_index=True)
        log.s('full memory training set: ' + str(mem_df.shape))
        df = self.all_bugs_df[self.all_bugs_df['url'].isin(mem_df['url'].values)].copy()
        df['RootCause'] = 'memory'
        return df

    def get_concurrency_df(self):
        csv_df = self.load_csv(root_dir() + 'dataset/manualClassification/keyword_memory_concurrency_bugs.csv')
        con_df = csv_df[csv_df['RootCause'] == 'concurrency']
        log.s('keyword search concurrency: ' + str(con_df.shape))

        rand_df = self.load_csv(root_dir() + 'dataset/manualClassification/randomly_selected_bugs.csv')
        rand_df = rand_df[rand_df['RootCause'] == 'concurrency']
        log.s('random selected concurrency: ' + str(rand_df.shape))

        con_df = con_df.append(rand_df, ignore_index=True)
        log.s('full concurrency training set: ' + str(con_df.shape))
        df = self.all_bugs_df[self.all_bugs_df['url'].isin(con_df['url'].values)].copy()
        df['RootCause'] = 'concurrency'
        return df

    def get_semantic_KW_bias_scaled_df(self):
        keyw_df = self.load_csv(root_dir() + 'dataset/manualClassification/keyword_memory_concurrency_bugs.csv')
        log.s('keyword search totals: ' + str(keyw_df.shape))
        log.s('keyword search manually classified: ' + str(keyw_df[keyw_df['ConfidenceNumeric'] > 0].shape))
        keyw_df = keyw_df[keyw_df['RootCause'] == 'semantic']
        log.s('keyword search semantic: ' + str(keyw_df.shape))
        keyw_df = keyw_df[keyw_df['ConfidenceNumeric'] > 6]
        log.s('keyword search semantic with confidence above 6: ' + str(keyw_df.shape))
        keyw_df = keyw_df.sample(int(keyw_df.shape[0]*0.05))
        log.s('keyword search semantic sample size: ' + str(keyw_df.shape))

        rand_df = self.load_csv(root_dir() + 'dataset/manualClassification/randomly_selected_bugs.csv')
        log.s('random selected totals: ' + str(rand_df.shape))
        rand_df = rand_df[rand_df['RootCause'] == 'semantic']
        log.s('random selected semantic: ' + str(rand_df.shape))
        rand_df = rand_df[rand_df['ConfidenceNumeric'] > 6]
        log.s('random selected semantic with confidence above 6: ' + str(rand_df.shape))

        sem_df = pandas.DataFrame()
        sem_df = sem_df.append(keyw_df, ignore_index=True)
        sem_df = sem_df.append(rand_df, ignore_index=True)
        log.s('full semantic training set: ' + str(sem_df.shape))
        df = self.all_bugs_df[self.all_bugs_df['url'].isin(sem_df['url'].values)].copy()
        df['RootCause'] = 'semantic'
        return df

    def get_semantic_df(self):
        keyw_df = self.load_csv(root_dir() + 'dataset/manualClassification/keyword_memory_concurrency_bugs.csv')
        log.s('keyword search totals: ' + str(keyw_df.shape))
        log.s('keyword search manually classified: ' + str(keyw_df[keyw_df['ConfidenceNumeric'] > 0].shape))
        keyw_df = keyw_df[keyw_df['RootCause'] == 'semantic']
        log.s('keyword search semantic: ' + str(keyw_df.shape))
        keyw_df = keyw_df[keyw_df['ConfidenceNumeric'] > 6]
        log.s('keyword search semantic with confidence above 6: ' + str(keyw_df.shape))

        rand_df = self.load_csv(root_dir() + 'dataset/manualClassification/randomly_selected_bugs.csv')
        log.s('random selected totals: ' + str(rand_df.shape))
        rand_df = rand_df[rand_df['RootCause'] == 'semantic']
        log.s('random selected semantic: ' + str(rand_df.shape))
        rand_df = rand_df[rand_df['ConfidenceNumeric'] > 6]
        log.s('random selected semantic with confidence above 6: ' + str(rand_df.shape))

        sem_df = pandas.DataFrame()
        sem_df = sem_df.append(keyw_df, ignore_index=True)
        sem_df = sem_df.append(rand_df, ignore_index=True)
        log.s('full semantic training set: ' + str(sem_df.shape))
        df = self.all_bugs_df[self.all_bugs_df['url'].isin(sem_df['url'].values)].copy()
        df['RootCause'] = 'semantic'
        return df

    def get_others_df(self):
        csv_df = self.load_csv(root_dir() + 'dataset/manualClassification/randomly_selected_other.csv')
        oth_df = csv_df[csv_df['RootCause'] == 'other']

        rand_df = self.load_csv(root_dir() + 'dataset/manualClassification/randomly_selected_bugs.csv')
        rand_df = rand_df[rand_df['RootCause'] == 'other']

        oth_df = oth_df.append(rand_df, ignore_index=True)

        log.s('full other training set: ' + str(oth_df.shape))
        df = self.all_bugs_df[self.all_bugs_df['url'].isin(oth_df['url'].values)].copy()
        df['RootCause'] = 'other'
        return df

    def get_traing_set_with_others_as_pandas_df(self):
        train_df = pandas.DataFrame()

        mem_df = self.get_memory_df()
        con_df = self.get_concurrency_df()
        sem_df = self.get_semantic_KW_bias_scaled_df()
        oth_df = self.get_others_df()

        train_df = train_df.append(mem_df.sample(124), ignore_index=True)
        train_df = train_df.append(con_df.sample(124), ignore_index=True)
        train_df = train_df.append(sem_df.sample(124), ignore_index=True)
        train_df = train_df.append(oth_df.sample(124), ignore_index=True)

        print('\ntrainingset shapes:\n')
        print('mem_df: ' + str(mem_df.shape) + '\n')
        print('con_df: ' + str(con_df.shape) + '\n')
        print('sem_df: ' + str(sem_df.shape) + '\n')
        print('oth_df: ' + str(oth_df.shape) + '\n')
        print('total: ' + str(train_df.shape) + '\n')

        return train_df
