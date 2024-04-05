import pandas as pd
import os

DATA_FULL_PATH = "E:\\home-credit-default-risk\\"

data_path = os.path.join(DATA_FULL_PATH, 'installments_payments.csv')
features_path = os.path.join(DATA_FULL_PATH, 'installments_payments_features.csv')

def main(data_path, features_path):
    print("[INFO] Starting to create installments_payments_features")
    
    installments_payments = pd.read_csv(data_path)
    installments_payments_features = pd.DataFrame()
    installments_payments_features['SK_ID_PREV'] = installments_payments['SK_ID_PREV']
    installments_payments_features['SK_ID_CURR'] = installments_payments['SK_ID_CURR']

    # Описание: разницу между предполагаемой датой оплаты и фактической датой оплаты в днях.
    # Интерпретация:  Отрицательные значения указывают на задержку в оплате, положительные значения указывают на преждевременную оплату.
    installments_payments_features['DIFF_DAYS'] = installments_payments['DAYS_INSTALMENT'] - installments_payments['DAYS_ENTRY_PAYMENT']

    # Описание: отношение фактически выплаченной суммы к предписанной сумме платежа.
    # Интерпретация: Значения меньше 1 указывают на то, что клиент выплатил менее, чем должен, значения больше 1 говорят о переплате.
    installments_payments_features['PAYMENT_RATIO'] = installments_payments['AMT_PAYMENT'] / installments_payments['AMT_INSTALMENT']

    # Описание: бинарный признак указывает, были ли задержки в платежах.
    # Интерпретация: Значение 1 указывает на то, что платеж был оплачен позже, чем ожидалось, значение 0 указывает на своевременную оплату.
    installments_payments_features['FLAG_DPD'] = (installments_payments['DAYS_ENTRY_PAYMENT'] > installments_payments['DAYS_INSTALMENT']).astype(int)

    # Описание: бинарный признак указывает, были ли неполные платежи.
    # Интерпретация: Значение 1 указывает на наличие задолженности (если PAYMENT_RATIO < 1), значение 0 указывает на своевременную оплату или переплату.
    installments_payments_features['FLAG_DEBT'] = installments_payments_features['PAYMENT_RATIO'].apply(lambda x: 0 if x >= 0 else 1) # .apply(lambda x: np.nan if np.isnan(x) else 0 if x >= 1 else 1)

    # Описание: бинарный признак отражает наличие изменений в версии погашения (NUM_INSTALMENT_VERSION).
    # Интерпретация: Значение True указывает на изменение в версии погашения.
    installments_payments_features['INSTALMENT_CHANGE'] = installments_payments.groupby('SK_ID_CURR')['NUM_INSTALMENT_VERSION'].transform('nunique').astype(bool)

    # Описание: доля дней просрочки.
    # Интерпретация: Большее значение указывает на более частые случаи просрочки платежей.
    installments_payments_features['RATIO_DPD'] = installments_payments_features.groupby('SK_ID_PREV')['FLAG_DPD'].transform('sum') / installments_payments_features.groupby('SK_ID_PREV')['FLAG_DPD'].transform('count')

    # Описание: доля неполных платежей.
    # Интерпретация: Большее значение указывает на более частые неполные взносы по кредиту.
    installments_payments_features['RATIO_DEBT'] = installments_payments_features.groupby('SK_ID_PREV')['FLAG_DEBT'].transform('sum') / installments_payments_features.groupby('SK_ID_PREV')['FLAG_DEBT'].transform('count')

    result_csv = installments_payments_features.groupby('SK_ID_CURR')[installments_payments_features.columns[2:]].mean()
    result_csv.to_csv(features_path, sep=',')
    print("[INFO] installments_payments_features successfully created")


if __name__ == "__main__":
    main(data_path, features_path)