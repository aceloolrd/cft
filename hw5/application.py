import pandas as pd
from utils import preprocess_data, select_significant_features


def main(read_path: str, save_path: str, bootstrap: bool=True) -> None:
    """
    Основная функция для обработки данных и сохранения результата.
    
    Args:
        read_path (str): Путь к файлу с данными.
        save_path (str): Путь для сохранения результирующего файла CSV.
        bootstrap (bool, optional): Флаг использования бутстрэпа. По умолчанию True.
    """
    print("[INFO] Starting to create application_result")

    application = pd.read_csv(read_path)
    
    binary_features, categorical_features, numeric_features = preprocess_data(application, {'TARGET','SK_ID_CURR'})
    result_cols = select_significant_features(application, binary_features, categorical_features, numeric_features, bootstrap=bootstrap)
    result_cols.extend(['SK_ID_CURR', 'TARGET'])
    
    application[result_cols].to_csv(save_path, index=False)
    
    print("[INFO] application_result successfully created")


if __name__ == "__main__":
    read_path = 'E:\\home-credit-default-risk\\application_train.csv'
    save_path = 'E:\\home-credit-default-risk\\application_result.csv'
    main(read_path=read_path, save_path=save_path, bootstrap=True)
