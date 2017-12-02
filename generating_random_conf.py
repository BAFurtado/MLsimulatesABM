import random

import pandas as pd

import preparing_data

d_uniform = {"HOUSE_VACANCY": [0.02, 0.4], "MARKUP": [0, .5], "MEMBERS_PER_FAMILY": [1, 5],
             "PRODUCTION_MAGNITUDE": [1, 200], "SIZE_MARKET": [5, 50], "PERCENTAGE_CHECK_NEW_LOCATION": [.01, .3]}
d_taxes = {"TAX_CONSUMPTION": [.004, .00004], "TAX_ESTATE_TRANSACTION": [7e-06, 7e-08],
           "TAX_FIRM": [.004, .00004], "TAX_LABOR": [.0015, .000015], "TAX_PROPERTY": [4e-06, 4e-08],
           "TREASURE_INTO_SERVICES": [.8, 1.2]}
d_bool = ["ALTERNATIVE0", "FPM_DISTRIBUTION", "WAGE_IGNORE_UNEMPLOYMENT"]
d_perc = ["ALPHA", "BETA", "LABOR_MARKET", "PCT_DISTANCE_HIRING", "PERCENTAGE_ACTUAL_POP", "STICKY_PRICES"]
d_acps = ['ARACAJU', 'BELEM', 'BELO HORIZONTE', 'BRASILIA', 'CAMPINA GRANDE', 'CAMPINAS', 'CAMPO GRANDE',
          'CAMPOS DOS GOYTACAZES', 'CAXIAS DO SUL', 'CUIABA', 'CURITIBA', 'LONDRINA', 'FEIRA DE SANTANA',
          'FLORIANOPOLIS', 'FORTALEZA', 'GOIANIA', 'ILHEUS - ITABUNA', 'IPATINGA', 'JOAO PESSOA', 'JOINVILLE',
          'JUAZEIRO DO NORTE - CRATO - BARBALHA', 'JUIZ DE FORA', 'JUNDIAI', 'MACAPA', 'MACEIO', 'MANAUS',
          'MARINGA', 'NATAL', 'NOVO HAMBURGO - SAO LEOPOLDO', 'PELOTAS - RIO GRANDE', 'PETROLINA - JUAZEIRO',
          'PORTO ALEGRE', 'RECIFE', 'RIBEIRAO PRETO', 'RIO DE JANEIRO', 'SALVADOR', 'SANTOS', 'SAO JOSE DO RIO PRETO',
          'SAO JOSE DOS CAMPOS', 'SAO LUIS', 'SAO PAULO', 'SOROCABA', 'TERESINA', 'UBERLANDIA', 'VITORIA',
          'VOLTA REDONDA - BARRA MANSA']


def generate(i=0):
    data = dict()
    for key in d_uniform.keys():
        data[key] = round(random.uniform(d_uniform[key][0], d_uniform[key][1]), 2)
    for key in d_taxes.keys():
        data[key] = round(random.uniform(d_taxes[key][0], d_taxes[key][1]), 8)
    for each in d_bool:
        data[each] = random.choice(['True', 'False'])
    for each in d_perc:
        data[each] = round(random.random(), 2)
    data['PROCESSSING_ACPS'] = random.choice(d_acps)
    return pd.DataFrame(data, index=[i])


def compound(n=500, df=pd.DataFrame()):
    for i in range(n):
        df = pd.concat([df, generate(i)])
    df = df.fillna(0)
    return preparing_data.dummies(df)


if __name__ == '__main__':
    d = compound()
    print(d.columns)