import pandas as pd
from utils import preprocess_data, select_significant_features

def main(read_path: str, target_path: str, save_path: str, bootstrap: bool=True) -> None:
    """
    Основная функция для обработки данных и сохранения результата.
    
    Args:
        read_path (str): Путь к файлу с данными.
        target_path (str): Путь к файлу с таргетом.
        save_path (str): Путь для сохранения результирующего файла CSV.
        bootstrap (bool, optional): Флаг использования бутстрэпа. По умолчанию True.
    """
    print("[INFO] Starting to create previous_application_result")
    
    prev_application = pd.read_csv(read_path)
    target = pd.read_csv(target_path, usecols=['TARGET','SK_ID_CURR'])
    prev_application = prev_application.merge(target, on = 'SK_ID_CURR')

    binary_features, categorical_features, numeric_features = preprocess_data(prev_application, {'TARGET','SK_ID_CURR', 'SK_ID_PREV'})
    result_cols = select_significant_features(prev_application, binary_features, categorical_features, numeric_features, bootstrap=bootstrap)
    result_cols.extend(['SK_ID_CURR', 'SK_ID_PREV'])

    prev_application[result_cols].to_csv(save_path, index=False)
    
    print("[INFO] previous_application_result successfully created")

if __name__ == "__main__":
    read_path = 'E:\\home-credit-default-risk\\previous_application.csv'
    target_path = 'E:\\home-credit-default-risk\\application_train.csv'
    save_path = 'E:\\home-credit-default-risk\\previous_application_result.csv'
    main(read_path=read_path, target_path=target_path, save_path=save_path, bootstrap=False)

