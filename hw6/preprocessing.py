import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.externals import joblib


def process_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Обрабатывает категориальные признаки данных.

    Параметры:
        data (pd.DataFrame): Необработанные данные.

    Возвращает:
        pd.DataFrame: Обработанные данные с закодированными категориальными признаками.
    """
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Выбираем категориальные столбцы
    catcols = data.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_pipeline, catcols)
        ],
        remainder='passthrough'
    )

    # Обрабатываем категориальные признаки и объединяем с остальными признаками
    processed_data = preprocessor.fit_transform(data)
    encoded_cat_columns = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(catcols)
    all_columns = list(encoded_cat_columns) + list(data.columns.drop(catcols))
    all_columns = [col.replace(' ', '_').upper() for col in all_columns]

    # Создаем DataFrame с обработанными данными
    processed_data_df = pd.DataFrame(processed_data, columns=all_columns)

    return processed_data_df


def drop_missing_numerical_columns(data: pd.DataFrame, threshold: float = 1) -> pd.DataFrame:
    """
    Удаляет числовые столбцы с пропущенными значениями.

    Параметры:
        data (pd.DataFrame): Данные.
        threshold (float): Порог пропущенных значений для удаления столбца.

    Возвращает:
        pd.DataFrame: Данные с удаленными столбцами.
    """
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    missing_values = data[num_cols].isna().sum()
    missing_percentage = (missing_values * 100 / len(data)).sort_values(ascending=False)

    to_drop_num = missing_percentage[missing_percentage > threshold].index
    data = data.drop(to_drop_num, axis=1)
    
    return data


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Подготавливает данные для обучения модели.

    Параметры:
        data (pd.DataFrame): Необработанные данные.

    Возвращает:
        tuple[pd.DataFrame, pd.Series]: Обработанные признаки и целевая переменная.
    """
    processed_data = process_categorical_features(data).dropna(axis=0)

    X = processed_data.drop(columns=['SK_ID_CURR', 'TARGET'])
    y = processed_data['TARGET']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    scaler_filename = 'scaler.pkl'
    joblib.dump(scaler, scaler_filename)

    X_scaled_df = X_scaled.copy()
    y.reset_index(drop=True, inplace=True) 
    X_scaled_df = X_scaled_df.join(y)
    corr_matrix = X_scaled_df.corr(method='spearman')

    target_correlation = corr_matrix.loc['TARGET']
    selected_features = target_correlation.abs().nlargest(100).index[1:]

    return X_scaled[selected_features], y  


def main(input_path: str, output_path: str) -> None:
    """
    Основная функция для обработки данных и сохранения их в файл.

    Параметры:
        input_path (str): Путь к файлу необработанных данных.
        output_path (str): Путь для сохранения обработанных данных.

    Возвращает:
        None
    """
    # Загрузка данных
    data = pd.read_csv(input_path)
    
    # Обработка данных
    data = drop_missing_numerical_columns(data)
    X, y = prepare_data(data)
    
    # Сохранение обработанных данных в CSV
    processed_data = X.join(y) 
    processed_data.to_csv(output_path, index=False)
    print(f"Обработанные данные сохранены в {output_path}")

# Вызов функции main с указанием пути к необработанным данным и пути для сохранения обработанных данных
if __name__ == "__main__":
    input_path = 'E:\\home-credit-default-risk\\application_train.csv' # Замените на путь к вашему файлу данных
    output_path = 'processed_data.csv'   # Замените на путь для сохранения обработанных данных
    main(input_path, output_path)
