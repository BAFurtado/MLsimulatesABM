import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format


def print_conf_stats(kwargs):
    # Dict contains X and Y as lists in a dictionary for actual and each model
    name = 'outputs\\output.csv'
    df = pd.DataFrame()
    for i, key in enumerate(kwargs.keys()):
        temp1 = pd.concat([kwargs[key][0], kwargs[key][1]], axis=1)
        temp1.rename(columns={temp1.columns[-1]: key}, inplace=True)
        temp2 = kwargs[key][0].mean(axis=0)
        temp3 = temp1.groupby([key]).agg('mean')
        res = pd.concat([temp2, temp3.T], axis=1)
        res.columns=['tot_' + key, key + '_0', key + '_1']
        df = pd.concat([df, res], axis=1)
    with open(name, 'wb'):
        df.to_csv(name, sep=';', float_format='%.6f')
