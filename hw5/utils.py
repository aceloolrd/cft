import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2, chi2_contingency
from sklearn.preprocessing import OrdinalEncoder


def preprocess_data(df: pd.DataFrame, dropcol: set) -> tuple:
    """
    Получет списки названий бинарных, категориальных и числовых признаков.
    
    Args:
        df (pandas.DataFrame): DataFrame с данными.
        dropcol (set): Множество названий столбцов, которые нужно исключить из обработки.
    
    Returns:
        tuple: Кортеж, содержащий списки названий бинарных, категориальных и числовых признаков.
    """
    binary_features = [col for col in df.columns if df[col].nunique() == 2]
    binary_features.remove('TARGET')

    categorical_features = df.select_dtypes(include='object').columns
    categorical_features = list(set(categorical_features) - set(binary_features))

    numeric_features = list(set(df.columns) - set(categorical_features) - set(binary_features) - dropcol)
    
    return binary_features, categorical_features, numeric_features


def select_significant_features(df: pd.DataFrame, binary_features: list, categorical_features: list, 
                                numeric_features: list, bootstrap: bool=True) -> list:
    """
    Выбирает значимые признаки на основе статистических тестов.
    
    Args:
        df (pandas.DataFrame): DataFrame с данными.
        binary_features (list): Список бинарных признаков.
        categorical_features (list): Список категориальных признаков.
        numeric_features (list): Список числовых признаков.
        bootstrap (bool, optional): Флаг использования бутстрэпа. По умолчанию True.
    
    Returns:
        list: Список названий отобранных значимых признаков.
    """
    result_cols = []

    for col in binary_features:
        no_nan = df.dropna(subset=[col]).copy()
        if Chi2_significance(no_nan, col):
            result_cols.append(col)
        
    for col in categorical_features:
        no_nan = df.dropna(subset=[col]).copy()
        no_nan[col] = Ordinal_Encoding(no_nan, col)
        if Chi2_significance(no_nan, col):
            result_cols.append(col)

    if bootstrap:
        for col in numeric_features:
            no_nan = df.dropna(subset=[col]).copy()
            cidiff = bootstraping_mean(no_nan, no_nan['TARGET'], feat_name=col)
            if verdict(cidiff) == 1:
                result_cols.append(col)
    else: 
        for col in numeric_features:
            no_nan = df.dropna(subset=[col]).copy()
            if Mann_Whitney_significance(no_nan, col):
                result_cols.append(col)
    
    return result_cols



def Chi2_significance(df, col):
    # Проверка значимости по критерию Хи-квадрат
    # Теперь функция возвращает название колонки, если она важна
    cross_tab = pd.concat([
        pd.crosstab(df[col], df['TARGET'], margins=False),
        df.groupby(col)['TARGET'].agg(['count', 'mean']).round(4)
    ], axis=1).rename(columns={0: f"target=0", 1: f"target=1", "mean": 'probability_of_default'})

    cross_tab['probability_of_default'] = np.round(cross_tab['probability_of_default'] * 100, 2)

    chi2_stat, p, dof, expected = chi2_contingency(cross_tab.values)

    prob = 0.95
    critical = chi2.ppf(prob, dof)
    if critical > abs(chi2_stat):
        return None
    else:
        return col
    
    
def Mann_Whitney_significance(df, col):
    # Проверка значимости по критерию Манна-Уитни
    # Теперь функция возвращает название колонки, если она важна
    _, p_mw = mannwhitneyu(df[df['TARGET'] == 0][col], df[df['TARGET'] == 1][col])
    if p_mw >= 0.05:
        return None
    else:
        return col

def Ordinal_Encoding(df, col):
    #Кодирование признаков (нам важно чтобы не увеличивалось число признаков,
    # поэтому использую этот энкодер)

    df = df.copy()
    categories = list(df.groupby(col)['TARGET'].mean().sort_values().index.values)
    OE = OrdinalEncoder(categories=[categories])
    df[col] = OE.fit_transform(df[[col]]).astype('int8')
    return df[col]


def bootstrap(
        data1,
        data2,
        n=1000,
        func=np.mean,
        subtr=np.subtract,
        alpha=0.05
):
    """
    Бутстрап средних значений для двух групп

    data1 - выборка 1 группы
    data2 - выборка 2 группы
    n=10000 - сколько раз моделировать
    func=np.mean - функция отвыборки, например, среднее
    subtr=np.subtract,
    alpha=0.05 - 95% доверительный интервал

    return:
    ci_diff - доверительный интервал разницы средних для двух групп
    s1 - распределение средних для 1 группы
    s2 - распределение средних для 2 группы
    confidence_interval(s1, s2, n, 1 - alpha) - доверительные интервалы для двух групп
    """
    s1, s2 = [], []
    s1_size = len(data1)
    s2_size = len(data2)

    for i in range(n):
        itersample1 = np.random.choice(data1, size=s1_size, replace=True)
        s1.append(func(itersample1))
        itersample2 = np.random.choice(data2, size=s2_size, replace=True)
        s2.append(func(itersample2))
    s1.sort()
    s2.sort()

    # доверительный интервал разницы
    bootdiff = subtr(s2, s1)
    bootdiff.sort()

    ci_diff = (np.round(bootdiff[np.round(n*alpha/2).astype(int)], 3),
               np.round(bootdiff[np.round(n*(1-alpha/2)).astype(int)], 3))

    return ci_diff, s1, s2


def bootstraping_mean(
        data,
        y,
        feat_name=None,
        val=[0, 1]
):
    """
    Бутстрап средних значений для любого признака

    data - датафрейм с данными
    y - таргет
    feat_name - название признака, строка

    return:
    cidiff - доверительный интервал разницы в средних значениях для двух групп
    """
    data1 = data[(y == val[0])][feat_name]
    data2 = data[(y == val[1])][feat_name]
    s1_mean_init = np.mean(data1)
    s2_mean_init = np.mean(data2)

    cidiff, s1, s2 = bootstrap(data1, data2)

    return cidiff


def verdict(ci_diff):
    # Убрал вывод в консоль
    cidiff_min = 0.001  # ,близкое к 0
    ci_diff_abs = [abs(ele) for ele in ci_diff]
    if (min(ci_diff) <= cidiff_min <= max(ci_diff)):
        # print(ci_diff, 'Различия в средних статистически незначимы.')
        pass
    elif (cidiff_min >= max(ci_diff_abs) >= 0) or (cidiff_min >= min(ci_diff_abs) >= 0):
        # print(ci_diff, 'Различия в средних статистически незначимы.')
        pass
    else:
        # print(ci_diff, 'Различия в средних статистически значимы.')
        return 1
