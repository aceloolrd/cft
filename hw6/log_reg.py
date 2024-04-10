import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
import pickle


def main(data_path: str, model_save_path: str) -> None:
    """
    Основная функция для обучения модели логистической регрессии, выполнения настройки гиперпараметров,
    оценки ее производительности и сохранения лучшей модели.

    Параметры:
        data_path (str): Путь к файлу обработанных данных.
        model_save_path (str): Путь для сохранения обученной модели.

    Возвращает:
        None
    """
    # Загрузка данных
    data = pd.read_csv(data_path)
    
    if 'SK_ID_CURR' in data.columns:
        X = data.drop(columns=['SK_ID_CURR', 'TARGET'])
    else:
        X = data.drop(columns=['TARGET'])
    y = data['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Создание и обучение модели логистической регрессии с настройкой гиперпараметров с использованием кросс-валидации
    pipeline = Pipeline([
        ('classifier', LogisticRegression(random_state=42))
    ])

    param_grid = {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l2']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Вывод лучших параметров модели
    print("Лучшие параметры:", grid_search.best_params_)

    # Получение окончательной модели с лучшими параметрами
    best_model = grid_search.best_estimator_

    # Предсказание вероятностей на тестовом наборе
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Вычисление ROC-AUC для данных
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print("Итоговый показатель ROC-AUC:", roc_auc)

    # Сохранение модели в файл pickle
    with open(model_save_path, 'wb') as f:
        pickle.dump(best_model, f)
    print("Модель сохранена в", model_save_path)

# Вызов основной функции с указанием путей к обработанным данным и месту сохранения модели
if __name__ == "__main__":
    data_path = 'processed_data.csv'  # Замените на путь к вашему файлу обработанных данных
    model_save_path = 'log_reg.pkl'  # Замените на путь для сохранения модели
    main(data_path, model_save_path)