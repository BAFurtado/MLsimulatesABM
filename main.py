import os

import pandas as pd
from numpy import set_printoptions

import descriptive_stats
import generating_random_conf
import machines
import preparing_data
from sklearn.model_selection import train_test_split
set_printoptions(precision=4)


def get_data(pathways, col1, col2):
    name = 'pre_processed_data\\' + pathways[-4:] + '_x.csv'
    if os.path.exists(name):
        x = pd.read_csv(name, sep=';', index_col=False).drop('Unnamed: 0', axis=1)
        y = pd.read_csv(name.replace('x', 'y'), sep=';', index_col=False)
        print('Loaded!')
    else:
        x, y = preparing_data.main(pathways, col1, col2)
        x.to_csv(name, sep=';')
        y.to_csv(name.replace('x', 'y'), sep=';')
        print('Saved!')
    return train_test_split(x, y['target'], test_size=0.2, random_state=10)


def print_results(x, y, x_r, y_r):
    res = pd.concat([x, y], axis=1)
    res = res.groupby(by=['target']).agg('mean')
    res_r = pd.concat([x_r, y_r], axis=1)
    res_r = res_r.groupby(by=['t2']).agg('mean')

    final = pd.concat([res, res_r], axis=0)
    final.T.to_csv('outputs\\comparison.csv', sep=';')
    print('\n Final results: \n')
    print(final)


def main(x, xt, y, yt):
    # Output basic descriptive stats for original dataset
    print('\n Training dataset summary: \n')
    descriptive_stats.print_conf_stats({'bases': [pd.concat([x, xt], axis=0)], 'text': ['actual']})

    # Running model
    m2 = machines.run_random_forest_split(x, xt, y, yt)

    # Generating random configuration data to test against optimal results
    r = generating_random_conf.compound()
    print('Generated dataset summary')

    # Output of descriptive stats for random configuration data
    descriptive_stats.print_conf_stats({'bases': [r], 'text': ['generated']})

    # Predicting results using machine on generated set of random parameters
    y2 = machines.predict(m2, r[x.columns.tolist()])

    # Transforming array into DataFrame
    y2 = pd.DataFrame({'t2': y2.tolist()})

    # Output results of machine on random data
    print_results(pd.concat([x, xt], axis=0), pd.concat([y, yt], axis=0), r, y2)


if __name__ == "__main__":
    cols_names = ['months', 'price_index', 'gdp_index', 'gdp_growth', 'unemployment', 'average_workers',
                  'families_wealth', 'families_savings', 'firms_wealth', 'firms_profit', 'gini_index',
                  'average_utility', 'inflation', 'average_qli']
    path = r'\\storage4\carga\MODELO DINAMICO DE SIMULACAO\Exits_python\JULY'
    target1 = 'gdp_index'
    target2 = 'gini_index'
    x_train, x_test, y_train, y_test = get_data(path, target1, target2)
    main(x_train, x_test, y_train, y_test)
