import numpy as np
import pandas as pd
import os


DATA_FULL_PATH = "E:\\home-credit-default-risk\\"

data_path = os.path.join(DATA_FULL_PATH, 'application.csv')
features_path = os.path.join(DATA_FULL_PATH, 'application_features.csv')

def main(data_path, features_path):
    print("[INFO] Starting to create application_features")
    
    application = pd.read_csv(data_path)
    application_features = pd.DataFrame()
    application_features['SK_ID_CURR'] = application['SK_ID_CURR']

    #Кол-во документов\
    application_features['DOCS_COUNT'] = application.filter(like='FLAG_DOCUMENT_').sum(axis=1)

    #Есть ли полная информация о доме. Найдите все колонки, которые описывают характеристики дома и посчитайте кол-во непустых характеристик. Если кол-во пропусков меньше 30, то значение признака 1. Иначе 0\
    house_characteristics_columns = [
        'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
        'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
        'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG',
        'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG',
        'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE',
        'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE',
        'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE',
        'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE',
        'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',
        'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI',
        'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
        'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
        'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE', 'WALLSMATERIAL_MODE',
        'EMERGENCYSTATE_MODE'
    ]

    application_features['HOUSE_INFO'] = np.where(application[house_characteristics_columns].isnull().sum(axis=1) < 30, 1, 0)

    #Кол-во полных лет\
    application_features['AGE_Y'] = application['DAYS_BIRTH'] //-365
        
    #Сколько лет назад был сменён документ \\ Год смены документа\
    application_features['DOC_CHANGE_Y'] = (application['DAYS_BIRTH'] - application['DAYS_ID_PUBLISH']) //-365
        
    #В каком возрасте клиент сменил документ\
    application_features['AGE_AT_PUBLISH'] = (application['DAYS_BIRTH'] // -365) - application_features['DOC_CHANGE_Y']
    
    #Признак задержки смены документа. Документ выдается или меняется в 14, 20 и 45 лет\
    application_features['FLAG_DELAY_DOC_CHANGE']  = np.where((application_features['DOC_CHANGE_Y']==14)|
                                                            (application_features['DOC_CHANGE_Y']==20)|
                                                            (application_features['DOC_CHANGE_Y']==45), 0, 1)
        
    #Доля денег которые клиент отдает на займ за год\
    application_features['ANNUITY_INCOME_RATIO'] = application['AMT_ANNUITY']/application['AMT_INCOME_TOTAL']
        
    #Среднее кол-во детей в семье на одного взрослого\
    application_features['CHILDREN_PER_ADULT'] =  application['CNT_CHILDREN']/(application['CNT_FAM_MEMBERS'] - application['CNT_CHILDREN'])

        
    #Средний доход на ребенка\
    # NaN, если детей 0
    application_features['INCOME_PER_CHILD'] = np.where(application['CNT_CHILDREN'] == 0, np.nan, application['AMT_INCOME_TOTAL'] / application['CNT_CHILDREN'])

        
    #Средний доход на взрослого\
    application_features['INCOME_PER_ADULT'] = application['AMT_INCOME_TOTAL']/(application['CNT_FAM_MEMBERS']-application['CNT_CHILDREN'])

    #Процентная ставка\
    application_features['INTEREST_RATE'] = (1 - (application['AMT_GOODS_PRICE']/application['AMT_CREDIT']))*100

        
    #Взвешенный скор внешних источников. Подумайте какие веса им задать.\
    # Не совсем понял, что должно было выйти. Посчитал по формуле средневзвешенного значения
    # Задаем веса внешним источникам
    weights = np.array([1/3, 1/3, 1/3]) # w1, w2, w3

    application_features['WEIGHTED_AVERAGE_SCORE'] = np.dot(application[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], weights) / np.sum(weights)
        
    #Поделим людей на группы в зависимости от пола и образования. В каждой группе посчитаем средний доход. Сделаем признак разница между средним доходом в группе и доходом заявителя
    # Учитывая, что для тестовых данных никто не будет пересчитывать статистики, берем только train часть
    application_features['DIFF_BTW_INCOME_AND_MEAN_GROUP_INCOME'] = application.loc[application['TARGET'].notna()].groupby(['CODE_GENDER','NAME_EDUCATION_TYPE'])['AMT_INCOME_TOTAL'].transform('mean') - application['AMT_INCOME_TOTAL']


    application_features.set_index('SK_ID_CURR', inplace=True)
    application_features.to_csv(features_path, sep=',')
    
    print("[INFO] application_features successfully created")


if __name__ == "__main__":
    main(data_path, features_path)