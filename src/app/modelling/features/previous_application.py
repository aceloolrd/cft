import pandas as pd
import os


DATA_FULL_PATH = "E:\\home-credit-default-risk\\"

data_path = os.path.join(DATA_FULL_PATH, 'previous_application.csv')
features_path = os.path.join(DATA_FULL_PATH, 'previous_application_features.csv')

def main(data_path, features_path):
    print("[INFO] Starting to create previous_application_features")
    
    previous_application = pd.read_csv(data_path)
    previous_application_features = pd.DataFrame()
    previous_application_features['SK_ID_CURR'] = previous_application['SK_ID_CURR']
    previous_application_features['SK_ID_PREV'] = previous_application['SK_ID_PREV']

    # Описание: Отношение суммы запрашиваемого кредита к окончательной сумме кредита.
    # Интерпретация: Значения выше 1 указывают на то, что клиенту одобрено меньше, чем он запросил.
    previous_application_features['AMT_APPLICATION_PER_AMT_CREDIT'] = previous_application['AMT_APPLICATION'] / previous_application['AMT_CREDIT']

    # Описание: Отношение окончательной суммы кредита к цене товара, на который был взят кредит.
    # Интерпретация: Значения близкие к 1 указывают на то, что клиент покрыл большую часть стоимости товара кредитом.
    previous_application_features['AMT_CREDIT_PER_AMT_GOODS'] = previous_application['AMT_CREDIT']/previous_application['AMT_GOODS_PRICE']

    # Описание: Процент одобренных заявок кредита для каждого клиента.
    # Интерпретация: Этот показатель позволяет оценить успешность заявок клиента. Значения близкие к 1 указывают на то, что большинство заявок было одобрено.
    approved = pd.get_dummies(previous_application, columns=['NAME_CONTRACT_STATUS'], drop_first= False, dtype=float)[['SK_ID_CURR','NAME_CONTRACT_STATUS_Approved']]
    approval_ratio = approved.groupby('SK_ID_CURR').transform('mean')
    approval_ratio.rename(columns={'NAME_CONTRACT_STATUS_Approved':'APPROVAL_RATIO'}, inplace=True)
    previous_application_features = pd.concat([previous_application_features, approval_ratio], axis=1)

    # Описание: Отношение первоначального взноса кредита к окончательной сумме кредита.
    # Интерпретация: Какую часть от общей суммы кредита клиент выплатил как первоначальный взнос.
    previous_application_features['AMT_DOWNPAYMENT_TO_CREDIT_RATIO'] = previous_application['AMT_DOWN_PAYMENT'] / previous_application['AMT_CREDIT']

    # агрегированные характеристики
    aggs = ['sum', 'mean', 'median', 'min', 'max', 'prod']
    features = ['AMT_CREDIT', 'AMT_GOODS_PRICE'] # посчитаем для этих признаков

    agg_data = previous_application.groupby('SK_ID_PREV')[features].agg(aggs)
    agg_data.columns = [f"{col[0]}_{col[1]}" for col in agg_data.columns]

    previous_application_features = previous_application_features.merge(agg_data, on='SK_ID_PREV')

    result_csv = previous_application_features.groupby('SK_ID_CURR')[previous_application_features.columns[2:]].sum()
    result_csv.to_csv(features_path, sep=',')

    print("[INFO] previous_application_features successfully created")


if __name__ == "__main__":
    main(data_path, features_path)