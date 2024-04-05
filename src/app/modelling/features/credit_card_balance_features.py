import pandas as pd
import os


DATA_FULL_PATH = "E:\\home-credit-default-risk\\"

data_path = os.path.join(DATA_FULL_PATH, 'credit_card_balance.csv')
features_path = os.path.join(DATA_FULL_PATH, 'credit_card_balance_features.csv')

def main(data_path, features_path):
    print("[INFO] Starting to create credit_card_balance_features")
    
    credit_card_balance = pd.read_csv(data_path)
    credit_card_balance_features = pd.DataFrame()
    credit_card_balance_features['SK_ID_PREV'] = credit_card_balance['SK_ID_PREV']
    credit_card_balance_features['SK_ID_CURR'] = credit_card_balance['SK_ID_CURR']
    credit_card_balance_features.drop_duplicates(subset='SK_ID_PREV', inplace=True)

    aggs = ['sum', 'mean', 'median', 'min', 'max', 'std', 'var', 'prod']
    features = ['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL'] # посчитаем для этих признаков

    agg_data = credit_card_balance.groupby('SK_ID_PREV')[features].agg(aggs)
    agg_data.columns = [f"{col[0]}_{col[1]}" for col in agg_data.columns]

    last_3_months = credit_card_balance[credit_card_balance['MONTHS_BALANCE'] >= -3].groupby('SK_ID_PREV')[features].agg(aggs)
    last_3_months.columns = [f"{col[0]}_{col[1]}" for col in last_3_months.columns]

    ratio_all_time_to_last_3_months = (agg_data/last_3_months).add_suffix('_diff')

    credit_card_balance_features = credit_card_balance_features.merge(agg_data, on='SK_ID_PREV')
    credit_card_balance_features = credit_card_balance_features.merge(ratio_all_time_to_last_3_months, on='SK_ID_PREV')
    
    result_csv = credit_card_balance_features.groupby('SK_ID_CURR')[credit_card_balance_features.columns[2:]].sum()
    result_csv.to_csv(features_path, sep=',')
    print("[INFO] credit_card_balance_features successfully created")


if __name__ == "__main__":
    main(data_path, features_path)