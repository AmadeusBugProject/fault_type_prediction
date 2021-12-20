
from scipy import stats
import matplotlib.pyplot as plt
import pandas

def t_test_x_differnt_y(x, y, x_label, y_label, output_path=None, df_file=None): # two sided
    df = pandas.DataFrame()
    df = df.append(is_normal(x, x_label, output_path))
    df = df.append(is_normal(y, y_label, output_path))

    stat, p = stats.ttest_ind(x, y, equal_var=False, alternative='two-sided')
    h0 = x_label + ' is not different ' + y_label
    df = df.append(pandas.DataFrame({'h0': [h0], 'test': ['t_test_two_sided'], 'stat': [stat], 'p': [p]}))
    if output_path and df_file:
        df.to_csv(output_path + df_file)
    return df


def t_test_x_greater_y(x, y, x_label, y_label, output_path=None, df_file=None): # one sided, x greater y
    df = pandas.DataFrame()
    df = df.append(is_normal(x, x_label, output_path))
    df = df.append(is_normal(y, y_label, output_path))

    stat, p = stats.ttest_ind(x, y, equal_var=False, alternative='greater')
    h0 = x_label + ' is not greater than ' + y_label
    df = df.append(pandas.DataFrame({'h0': [h0], 'test': ['t_test_one_sided'], 'stat': [stat], 'p': [p]}))
    if output_path and df_file:
        df.to_csv(output_path + df_file)
    return df


def is_normal(series, add_label, output_path=None):
    if output_path:
        stats.probplot(series, dist="norm", plot=plt)
        plt.savefig(output_path + add_label + '_normality.png')
        plt.close()
    shapiro_stat, shapiro_p = stats.shapiro(series)
    dagostino_stat, dagostino_p = stats.normaltest(series)

    df = pandas.DataFrame({'test': ['shapiro_stat', 'dagostino_stat'], 'stat': [shapiro_stat, dagostino_stat], 'p': [shapiro_p, dagostino_p]})
    df['h0'] = add_label + ' is normal distributed'
    return df
