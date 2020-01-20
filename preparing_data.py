""" Prepare the simulation data for Machine Learning procedures 
"""

import json
import operator
import os

import numpy as np
import pandas as pd

cols_names = ['months', 'price_index', 'gdp_index', 'gdp_growth', 'unemployment', 'average_workers',
              'families_wealth', 'families_savings', 'firms_wealth', 'firms_profit', 'gini_index',
              'average_utility', 'inflation', 'average_qli']


def read_json(p):
    # Interpret JSON file of configuration with simulation given parameters
    return json.load(open(p))


def json_to_dict(df):
    # Transforms JSON data into DataFrame, removing unchanging columns
    t = pd.DataFrame.from_dict(df, orient='index').drop(labels='RUN', axis=0).dropna(axis=1)
    t = t.drop(['LIST_NEW_AGE_GROUPS', 'TAXES_STRUCTURE', 'SIMPLIFY_POP_EVOLUTION'], axis=1)
    try:
        t = t.drop(['PROCESSING_STATES', 'HIRING_SAMPLE_SIZE'], axis=1)
    except:
        pass
    t['PROCESSING_ACPS'] = t['PROCESSING_ACPS'].apply(lambda x: x[0])
    # Spelling bug
    try:
        t = t.drop('HOUSE_VANCANCY', axis=1)
    except:
        pass
    return t


def read_conf_files(general_path):
    # Walks over directory collecting all conf.JSON files representing each simulation and its parameters
    return [os.path.join(dirpath, f)
            for dirpath, dirnames, files in os.walk(general_path)
            for f in files if f.startswith('conf')]


def process_each_file(files_list, cols, y=pd.DataFrame(), x=pd.DataFrame()):
    # Extract both parameters from conf.JSON files and results of that given simulation from 'avg' folder
    for each in files_list:
        # Removing 'conf.json' from path and accessing temp_stats.csv
        y_test = pd.read_csv(each[:-9] + r'\avg\temp_stats.csv', sep=';', header=None)
        # Testing last month of simulation
        if len(y_test) == 240:
            y = y.append(y_test)
            x = x.append(json_to_dict(read_json(each)))
    # Provides names for the columns of results of simulation
    y.columns = cols
    return x, y


def last_month(df):
    return df[df.months == 239].drop('months', axis=1)


def selecting_y(df, col):
    # Selects only results from last month of simulation
    return df[col]


def customizing_target(base, percentile=65, op=operator.gt):
    # Discretizes results for a given percentile and a given operator (greater than or less than)
    return pd.DataFrame({'target': [1 if op.__call__(x, np.percentile(base, percentile)) else 0 for x in base]})


def averaging_targets(df1, df2):
    # Summarizes two target columns into one when both results are one
    return pd.DataFrame({'target': [1 if x == 1 and y == 1 else 0 for x, y in zip(df1['target'], df2['target'])]})


def dummies(data):
    cat, num = [], []
    for i in data.columns:
        if data[i].dtype == object:
            cat.append(i)
        else:
            num.append(i)
    cat = data[cat]
    try:
        cat = cat.drop(['PROCESSING_STATES'], axis=1)
    except:
        pass
    cat = pd.get_dummies(cat)
    num = data[num]
    try:
        num = num.drop(['HIRING_SAMPLE_SIZE'], axis=1)
    except:
        pass
    return pd.concat([num, cat], axis=1)


def main(pathway, selected_col1, selected_col2):
    # Runs the script for a given directory and two given targets
    # Target1 set to percentile 80 and greater than
    # Target2 set to percentile 20 and less than
    file_list = read_conf_files(pathway)
    data_x, data_y = process_each_file(file_list, cols_names)
    # Getting last months' data
    data_y = last_month(data_y)

    # Excluding the binary operation on target and keeping all values

    first_col = customizing_target(selecting_y(data_y, selected_col1))
    second_col = customizing_target(selecting_y(data_y, selected_col2), 35, operator.lt)
    data_y = averaging_targets(first_col, second_col)

    data_x = dummies(data_x)
    name = 'pre_processed_data\\' + pathway[-4:] + '_' + selected_col1 + '_' + selected_col2 + '_x.csv'
    data_x.to_csv(name, index=False, sep=';')
    data_y.to_csv(name.replace('x.csv', 'y.csv'), index=False, sep=';')
    return data_x, data_y


if __name__ == "__main__":
    path = r'\\storage1\carga\MODELO DINAMICO DE SIMULACAO\Exits_python\JULY'
    target1 = 'average_qli'
    target2 = 'unemployment'

    x, y = main(path, target1, target2)
