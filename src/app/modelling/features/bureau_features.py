import pandas as pd
import os


DATA_FULL_PATH = "E:\\home-credit-default-risk\\"

data_path = os.path.join(DATA_FULL_PATH, 'bureau.csv')
features_path = os.path.join(DATA_FULL_PATH, 'bureau_features.csv')


def main(data_path,features_path):
    print("[INFO] Starting to create bureau_features")
    
    bureau = pd.read_csv(data_path)
    bureau_features = pd.DataFrame()

    # Максимальная сумма просрочки
    bureau_features = bureau.groupby('SK_ID_CURR').agg(MAX_DEBT_AMT = ('AMT_CREDIT_SUM_DEBT','max')).reset_index()

    # Минимальная сумма просрочки
    bureau_features['MIN_DEBT_AMT'] = bureau.groupby('SK_ID_CURR').agg(MIN_DEBT_AMT=('AMT_CREDIT_SUM_DEBT','min')).reset_index()['MIN_DEBT_AMT']

    # Какую долю суммы от открытого займа просрочил
    # Фильтрация активных кредитов
    bureau_active = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].copy()
    # Вычисление доли активной задолженности для каждого заемщика
    bureau_active.loc[:, 'OVERDUE_RATIO'] = bureau_active['AMT_CREDIT_SUM_DEBT'] / bureau_active['AMT_CREDIT_SUM']
    # Группировка по SK_ID_CURR и нахождение максимального значения доли активной задолженности
    bureau_active = bureau_active.groupby('SK_ID_CURR')['OVERDUE_RATIO'].max().reset_index()
    bureau_features = bureau_features.merge(bureau_active, how='left', on='SK_ID_CURR')

    # Кол-во кредитов определенного типа
    # Группировка данных по заемщикам и типам кредитов, подсчет количества записей в каждой группе
    type_of_credit = bureau.groupby(['SK_ID_CURR', 'CREDIT_TYPE']).size().unstack(fill_value=0)
    type_of_credit.columns = ['count_' + str(col) for col in type_of_credit.columns]
    type_of_credit = type_of_credit.reset_index()
    bureau_features = bureau_features.merge(type_of_credit, on='SK_ID_CURR', how='left')

    # Кол-во просрочек кредитов определенного типа
    bureau['FLAG_OVERDUE'] = 0
    bureau.loc[bureau['CREDIT_DAY_OVERDUE'] > 0, 'FLAG_OVERDUE'] = 1

    type_of_credit_overdue = bureau.groupby(['SK_ID_CURR', 'CREDIT_TYPE'])['FLAG_OVERDUE'].sum().unstack(fill_value=0).reset_index().add_prefix('count_overdue_')
    bureau_features = bureau_features.merge(type_of_credit_overdue, left_on='SK_ID_CURR', right_on='count_overdue_SK_ID_CURR', how='left')
    bureau_features.drop('count_overdue_SK_ID_CURR', axis=1, inplace=True)

    # Кол-во закрытых кредитов определенного типа
    bureau['FLAG_CLOSED'] = 1
    # Установка значения 0 в столбец 'FLAG_CLOSED' для строк, где 'DAYS_ENDDATE_FACT' пропущено (NaN)
    bureau.loc[bureau['DAYS_ENDDATE_FACT'].isna(), 'FLAG_CLOSED'] = 0
    # Группировка данных по заемщикам и типам кредитов, и подсчет суммы значений столбца 'FLAG_CLOSED' для каждой группы
    type_of_credit_closed = bureau.groupby(['SK_ID_CURR', 'CREDIT_TYPE'])['FLAG_CLOSED'].sum().unstack(fill_value=0).reset_index().add_prefix('count_closed_')
    bureau_features = bureau_features.merge(type_of_credit_closed, left_on='SK_ID_CURR', right_on='count_closed_SK_ID_CURR', how='left')
    bureau_features.drop('count_closed_SK_ID_CURR', axis=1, inplace=True)

    bureau_features.set_index('SK_ID_CURR', inplace=True)
    bureau_features.to_csv(features_path, sep=',')
    
    print("[INFO] bureau_features successfully created")


if __name__ == "__main__":
    main(data_path,features_path)   