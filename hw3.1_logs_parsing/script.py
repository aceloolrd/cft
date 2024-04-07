import json
import pandas as pd
from dataclasses import dataclass

@dataclass
class AmtCredit:
    CREDIT_CURRENCY: str
    AMT_CREDIT_MAX_OVERDUE: float
    AMT_CREDIT_SUM: float
    AMT_CREDIT_SUM_DEBT: float
    AMT_CREDIT_SUM_LIMIT: float
    AMT_CREDIT_SUM_OVERDUE: float
    AMT_ANNUITY: float
    
@dataclass
class PosCashBalanceIDs:
    SK_ID_PREV : int
    SK_ID_CURR : int
    NAME_CONTRACT_STATUS : str


def load_data(file_path: str, num_lines: int = None) -> pd.DataFrame:
    """
    Загружает данные из файла лога и создает DataFrame.
    
    Args:
    - file_path (str): Путь к файлу лога.
    - num_lines (int): Количество строк для загрузки (по умолчанию None - загрузить весь файл).
    
    Returns:
    - pd.DataFrame: DataFrame, содержащий загруженные данные.
    """
    data = []
    try:
        with open(file_path) as f:
            # Если num_lines не указано, читаем весь файл
            if num_lines is None:
                for line in f:
                    data.append(json.loads(line))
            # Иначе читаем указанное количество строк
            else:
                for _ in range(num_lines):
                    try:
                        line = next(f)
                        data.append(json.loads(line))
                    except StopIteration:
                        break
    except FileNotFoundError:
        print(f"Файл {file_path} не найден.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return pd.DataFrame()
    return pd.DataFrame(data)


def extract_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Извлекает данные 'bureau' и 'POS_CASH_balance' из DataFrame и создает новые DataFrame.
    
    Args:
    - df (pd.DataFrame): DataFrame, содержащий данные из лога.
    
    Returns:
    - pd.DataFrame, pd.DataFrame: Два DataFrame с данными 'bureau' и 'POS_CASH_balance' соответственно.
    """
    bureau_df = pd.DataFrame(df[df['type'] == 'bureau']['data'].copy())
    pos_cash_df = pd.DataFrame(df[df['type'] == 'POS_CASH_balance']['data'].copy())
    # return bureau_df[:10], pos_cash_df[:10]
    return bureau_df, pos_cash_df


def normalize_bureau_data(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Нормализует данные 'bureau'.
    
    Args:
    - df (pd.DataFrame): DataFrame, содержащий данные из лога.
    - column (str): Название столбца, содержащего JSON-данные.
    
    Returns:
    - pd.DataFrame: Нормализованный DataFrame с данными 'bureau'.
    """
    normalized_df = pd.json_normalize(df[column])
    # Извлечение имен столбцов без префикса
    new_col = [col.split('.')[1] for col in normalized_df.columns[1:]]
    # Переименование столбцов с использованием новых имен
    normalized_df.columns = [normalized_df.columns[0]] + new_col
    
    return normalized_df


def normalize_pos_cash_data(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Нормализует данные 'POS_CASH_balance'.
    
    Args:
    - df (pd.DataFrame): DataFrame, содержащий данные из лога.
    - column (str): Название столбца, содержащего JSON-данные.
    
    Returns:
    - pd.DataFrame: Нормализованный DataFrame с данными 'POS_CASH_balance'.
    """
    normalized_df = pd.json_normalize(df[column])
    col = normalized_df.columns[1]
    normalized_df = normalized_df.explode(col).reset_index(drop=True)
    records_df = pd.json_normalize(normalized_df[col])
    normalized_df = pd.concat([normalized_df, records_df], axis=1).drop(columns=[col])

    return normalized_df


def normalize_data(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Нормализует данные из двух DataFrame.
    
    Args:
    - df1 (pd.DataFrame): DataFrame с данными 'bureau'.
    - df2 (pd.DataFrame): DataFrame с данными 'POS_CASH_balance'.
    
    Returns:
    - pd.DataFrame, pd.DataFrame: Нормализованные DataFrame для 'bureau' и 'POS_CASH_balance'.
    """
    bureau_df = normalize_bureau_data(df1, 'data')
    pos_cash_df = normalize_pos_cash_data(df2, 'data')
    return bureau_df, pos_cash_df


def extract_attributes(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Извлекает атрибуты из столбца и добавляет их в DataFrame.
    
    Args:
    - df (pd.DataFrame): DataFrame, содержащий данные из лога.
    - column (str): Название столбца, содержащего JSON-данные.
    
    Returns:
    - pd.DataFrame: DataFrame с извлеченными атрибутами.
    """
    df[column] = df[column].apply(lambda x: eval(x))
    
    attributes = {}
    for i, credit_obj in enumerate(df[column]):
        for attr_name in dir(credit_obj):
            if not attr_name.startswith('_'):
                attr_value = getattr(credit_obj, attr_name)
                if attr_name not in attributes:
                    attributes[attr_name] = [None] * len(df)
                attributes[attr_name][i] = attr_value
    attributes_df = pd.DataFrame(attributes)
    
    df = pd.concat([df, attributes_df], axis=1).drop(columns=[column])
    
    return df


def main(input_file: str, output_file1: str, output_file2: str):
    """
    Основная функция для выполнения всех этапов обработки данных.
    
    Args:
    - input_file (str): Путь к файлу лога.
    - output_dir1 (str): Путь для сохранения результата обработки данных 'bureau'.
    - output_dir2 (str): Путь для сохранения результата обработки данных 'POS_CASH_balance'.
    """
    # Загружаем данные
    df = load_data(input_file)

    # Извлекаем данные 'bureau'
    bureau_df, pos_cash_df = extract_data(df)
    
    bureau_df, pos_cash_df = normalize_data(bureau_df, pos_cash_df)

    bureau_df = extract_attributes(bureau_df, 'AmtCredit')
    pos_cash_df = extract_attributes(pos_cash_df, 'PosCashBalanceIDs')

     # Сохраняем результаты в .csv файлы
    bureau_df.to_csv(output_file1, index=False)
    pos_cash_df.to_csv(output_file2, index=False)
    
    
if __name__ == "__main__":
    input_file = input("Введите путь к входному .log файлу: ")
    output_dir1 = input("Укажите имя для первого .csv файла: ")
    output_dir2 = input("Укажите имя для второго .csv файла: ")
    main(input_file, output_dir1, output_dir2)