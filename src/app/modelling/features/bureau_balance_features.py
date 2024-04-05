import pandas as pd
import os


DATA_FULL_PATH = "E:\\home-credit-default-risk\\"

data_path1 = os.path.join(DATA_FULL_PATH, 'bureau_balance.csv')
data_path2 = os.path.join(DATA_FULL_PATH, 'bureau.csv')
# features_path = os.path.join(DATA_FULL_PATH, 'bureau_balance_features.csv')
features_path = ('bureau_balance_features.csv')

def main(data_path1, data_path2, features_path):
    print("[INFO] Starting to create bureau_balance_features")
    
    bureau = pd.read_csv(data_path2)
    bureau_balance = pd.read_csv(data_path1)
    bureau_balance_features = pd.DataFrame(bureau_balance.drop_duplicates('SK_ID_BUREAU'))[['SK_ID_BUREAU']]
    bureau_balance_features['SK_ID_CURR'] = bureau['SK_ID_CURR']

    # Кол-во открытых кредитов
    # Группировка по 'SK_ID_BUREAU' и 'STATUS', а затем подсчет количества записей в каждой группе
    status_counts = bureau_balance.groupby(['SK_ID_BUREAU', 'STATUS'])['MONTHS_BALANCE'].count().unstack(fill_value=0)
    status_counts['COUNT_OPEN_CREDITS'] = status_counts[['0', '1', '2', '3', '4', '5']].sum(axis=1)
    bureau_balance_features = bureau_balance_features.merge(status_counts['COUNT_OPEN_CREDITS'], on='SK_ID_BUREAU', how='left')

    # Кол-во закрытых кредитов
    bureau_balance_features['COUNT_CLOSED_CREDITS'] = status_counts.reset_index()['C']

    # Кол-во просроченных кредитов по разным дням просрочки (смотреть дни по колонке STATUS)
    bureau_balance_features = bureau_balance_features.merge(status_counts[['0', '1', '2', '3', '4', '5']], on='SK_ID_BUREAU', how='left')
    bureau_balance_features = bureau_balance_features.rename(
        columns={
            '0': 'NO_DPD',
            '1': 'DPD_30',
            '2': 'DPD_60',
            '3': 'DPD_90',
            '4':'DPD_120',
            '5': 'DPD_120+_OR_SOLD'
            })

    # Кол-во кредитов
    bureau_balance_features = bureau_balance_features.merge(bureau_balance.groupby('SK_ID_BUREAU').agg(COUNT_ALL_CREDIT=('SK_ID_BUREAU', 'count')), on='SK_ID_BUREAU')

    # Доля закрытых кредитов
    bureau_balance_features['CLOSED_CREDIT_RATIO'] = bureau_balance_features['COUNT_CLOSED_CREDITS']/bureau_balance_features['COUNT_ALL_CREDIT']

    # Доля открытых кредитов
    bureau_balance_features['OPEN_CREDIT_RATIO'] = bureau_balance_features['COUNT_OPEN_CREDITS']/bureau_balance_features['COUNT_ALL_CREDIT']

    # Доля просроченных кредитов по разным дням просрочки (смотреть дни по колонке STATUS)
    # Список дней просрочки для создания столбцов
    dpd_days = ['DPD_30', 'DPD_60', 'DPD_90', 'DPD_120', 'DPD_120+_OR_SOLD']

    # Создание столбцов и вычисление доли просроченных кредитов
    for dpd_day in dpd_days:
        ratio_column_name = dpd_day + '_RATIO'
        bureau_balance_features[ratio_column_name] = bureau_balance_features[dpd_day] / bureau_balance_features[dpd_days].sum(axis=1)
        bureau_balance_features[ratio_column_name] = bureau_balance_features[ratio_column_name].fillna(0)
        
    # Интервал между последним закрытым кредитом и текущей заявкой
    bureau_balance_features = bureau_balance_features.merge(bureau_balance.loc[bureau_balance['STATUS'] == 'C'].groupby('SK_ID_BUREAU')[['MONTHS_BALANCE']].max().reset_index().rename(columns={'MONTHS_BALANCE':'INTERVAL_LAST_CLOSED_CREDIT'}), on='SK_ID_BUREAU')

    # Интервал между взятием последнего активного займа и текущей заявкой
    active_credits = bureau_balance.loc[(bureau_balance['MONTHS_BALANCE'] == -1)&(~bureau_balance['STATUS'].isin(['C', 'X']))]
    bba = bureau_balance.merge(active_credits['SK_ID_BUREAU'], on = 'SK_ID_BUREAU')

    interval = bba.groupby('SK_ID_BUREAU')[['MONTHS_BALANCE']].min().reset_index()
    interval.rename(columns={'MONTHS_BALANCE':'INTERVAL_LAST_OPEN_CREDIT'},inplace=True)
    bureau_balance_features = bureau_balance_features.merge(interval,how = 'outer', on='SK_ID_BUREAU')

    result_csv = bureau_balance_features.groupby('SK_ID_CURR')[bureau_balance_features.columns[2:]].sum()
    result_csv.to_csv(features_path, sep=',')
    
    print("[INFO] bureau_balance_features successfully created")


if __name__ == "__main__":
    main(data_path1, data_path2, features_path)  