import numpy.random
import pandas as pd

import preparing_data

numpy.random.seed(0)

d_bool = ["ALTERNATIVE0", "FPM_DISTRIBUTION", "WAGE_IGNORE_UNEMPLOYMENT"]

d_normal = ["HOUSE_VACANCY", "MARKUP", "MEMBERS_PER_FAMILY", "PERCENTAGE_ACTUAL_POP", "PRODUCTION_MAGNITUDE",
            "SIZE_MARKET", "PERCENTAGE_CHECK_NEW_LOCATION", "TAX_CONSUMPTION", "TAX_ESTATE_TRANSACTION",
            "TAX_FIRM", "TAX_LABOR", "TAX_PROPERTY", "TREASURE_INTO_SERVICES", "ALPHA", "BETA", "LABOR_MARKET",
            "PCT_DISTANCE_HIRING", "STICKY_PRICES"]

d_acps = ['ARACAJU', 'BELEM', 'BELO HORIZONTE', 'BRASILIA', 'CAMPINA GRANDE', 'CAMPINAS', 'CAMPO GRANDE',
          'CAMPOS DOS GOYTACAZES', 'CAXIAS DO SUL', 'CUIABA', 'CURITIBA', 'LONDRINA', 'FEIRA DE SANTANA',
          'FLORIANOPOLIS', 'FORTALEZA', 'GOIANIA', 'ILHEUS - ITABUNA', 'IPATINGA', 'JOAO PESSOA', 'JOINVILLE',
          'JUAZEIRO DO NORTE - CRATO - BARBALHA', 'JUIZ DE FORA', 'JUNDIAI', 'MACAPA', 'MACEIO', 'MANAUS',
          'MARINGA', 'NATAL', 'NOVO HAMBURGO - SAO LEOPOLDO', 'PELOTAS - RIO GRANDE', 'PETROLINA - JUAZEIRO',
          'PORTO ALEGRE', 'RECIFE', 'RIBEIRAO PRETO', 'RIO DE JANEIRO', 'SALVADOR', 'SANTOS', 'SAO JOSE DO RIO PRETO',
          'SAO JOSE DOS CAMPOS', 'SAO LUIS', 'SAO PAULO', 'SOROCABA', 'TERESINA', 'UBERLANDIA', 'VITORIA',
          'VOLTA REDONDA - BARRA MANSA']


def pre_process(name):
    t = pd.read_csv(name, sep=';')
    return t.describe().T[['mean', 'std']]


def compound(name, n=100000):
    samples = pre_process(name)
    data = dict()
    for each in d_normal:
        data[each] = numpy.random.normal(samples.loc[each, 'mean'], samples.loc[each, 'std'] * 2, n)
    data['PROCESSING_ACPS'] = numpy.random.choice(d_acps, n)
    for each in d_bool:
        data[each] = numpy.random.choice(['True', 'False'], n)
    df = pd.DataFrame(data)
    temp1 = df[d_bool + ['PROCESSING_ACPS']]
    temp2 = df[df.columns.difference(d_bool + ['PROCESSING_ACPS'])]
    temp2 = temp2.fillna(0)
    for col in temp2.columns:
        temp2[col] = temp2[col].apply(lambda x: temp2[col].mean() if x < 0 else x)
    return preparing_data.dummies(pd.concat([temp1, temp2], axis=1))


if __name__ == '__main__':
    path = r'\\storage4\carga\MODELO DINAMICO DE SIMULACAO\Exits_python\JULY'
    file_name = 'pre_processed_data\\' + path[-4:] + '_x.csv'
    d = compound(file_name)
    print(d.head())
