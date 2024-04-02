import psycopg2
import pandas as pd

class DatabaseConnector:
    """
    Класс для работы с базой данных PostgreSQL.

    Args:
        args (dict): Параметры подключения к базе данных.

    Attributes:
        args (dict): Параметры подключения к базе данных.
    """

    def __init__(self, args: dict) -> None:
        self.args: dict = args
        
    
    def send_sql_query(self, query: str) -> None:
        """
        Выполняет SQL-запрос к базе данных.

        :param query: Строка с SQL-запросом.
        """
        try:
            with psycopg2.connect(**self.args) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    conn.commit()
        except psycopg2.Error as error:
            print("Error while executing SQL query:", error)
    
    
    def get_df_from_query(self, query: str) -> pd.DataFrame:
        """
        Выполняет SQL-запрос к базе данных и возвращает результат в виде датафрейма.

        :param query: Строка с SQL-запросом.

        :return: Датафрейм с результатом запроса.
        """
        try:
            with psycopg2.connect(**self.args) as conn:
                df = pd.read_sql(query, conn)
            return df
        except psycopg2.Error as error:
            print("Error while fetching data from PostgreSQL:", error)
            return None
