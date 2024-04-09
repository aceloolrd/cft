import pandas as pd
from utils import preprocess_data, select_significant_features


def main(read_path: str, bureau_path: str, target_path: str, save_path: str, bootstrap: bool=True) -> None:
    """
    Основная функция для обработки данных и сохранения результата.
    
    Args:
        read_path (str): Путь к файлу с данными.
        bureau_path (str): Путь к файлу с данными ID.
        target_path (str): Путь к файлу с таргетом.
        save_path (str): Путь для сохранения результирующего файла CSV.
        bootstrap (bool, optional): Флаг использования бутстрэпа. По умолчанию True.
    """
    print("[INFO] Starting to create bureau_balance_result")

    bureau_balance = pd.read_csv(read_path)
    bureau = pd.read_csv(bureau_path, usecols=['SK_ID_CURR', 'SK_ID_BUREAU'])
    target = pd.read_csv(target_path, usecols=['TARGET','SK_ID_CURR'])

    bureau_balance = bureau_balance.merge(bureau, on='SK_ID_BUREAU')
    bureau_balance = bureau_balance.merge(target, on='SK_ID_CURR')

    binary_features, categorical_features, numeric_features = preprocess_data(bureau_balance, {'TARGET','SK_ID_CURR','SK_ID_BUREAU'})
    result_cols = select_significant_features(bureau_balance, binary_features, categorical_features, numeric_features, bootstrap=bootstrap) 
    result_cols.append('SK_ID_BUREAU')
    
    bureau_balance[result_cols].to_csv(save_path, index=False)
    print("[INFO] bureau_balance_result successfully created")

if __name__ == "__main__":
    read_path = 'E:\\home-credit-default-risk\\bureau_balance.csv'
    bureau_path = 'E:\\home-credit-default-risk\\bureau.csv'
    target_path = 'E:\\home-credit-default-risk\\application_train.csv'
    save_path = 'E:\\home-credit-default-risk\\bureau_balance_result.csv'
    main(read_path=read_path, bureau_path=bureau_path, target_path=target_path, save_path=save_path, bootstrap=False)