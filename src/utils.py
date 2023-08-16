import pandas as pd
import ast

def load_df(path:str='data/customer_complaints.csv'):
## Загрузка df
    df = pd.read_csv(path)
    for col in ['Список товаров в заказе', 'Количество каждого товара', 'Цена каждого товара', 'Название проблемного продукта']:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x))
    return df

def random_id(df):
## Choose random ID and message
    row = df.copy().sample(1)
    id = row['Уникальный айди обращения'].values[0]
    message = row['Обращение'].values[0]
    return id, message, row


if __name__ == '__main__':
    df = load_df('data/customer_complaints.csv')
    id, message, row = random_id(df)
    print(id)
    print(message)
    print(row)
