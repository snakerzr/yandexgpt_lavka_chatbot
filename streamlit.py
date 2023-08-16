import streamlit as st
import pandas as pd
import re
import os
import random
from src.YandexGPT import YandexGPT
from src.Chatbot import Chatbot


# default session variables initialization
default_session_state_dict = {"n_row": 0, "env_loaded": False}

for key, value in default_session_state_dict.items():
    if key not in st.session_state:
        st.session_state[key] = value


button_style = """
        <style>
        .stButton > button {
            color: black;
            background: white;
            width: 200px;
            height: 50px;
        }
        </style>
        """

st.markdown(button_style, unsafe_allow_html=True)
st.title("AGI Fireworks")
st.caption("Царь костыль и велосипеды")
st.header("Прототип чатбота для службы поддержки")


st.subheader("Загрузите .env файл")
uploaded_file = st.file_uploader(
    "Choose an .env file ", label_visibility="hidden", type=".env"
)


df = pd.read_csv("data/customer_complaints.csv")
df_view = df[
    [
        "Тон обращения",
        "Тип жалобы-обращения",
        "Наличие грамматических ошибок",
        "Отсутствие знаков препинания",
        "Список товаров в заказе",
        "Количество каждого товара",
        "Цена каждого товара",
        "Общая стоимость товаров",
        "Название проблемного продукта",
        "Количество товара с проблемой",
    ]
]

if uploaded_file is not None:
    st.session_state["env_loaded"] = True
    env = str(uploaded_file.read())
    service_account_id = re.findall("SERVICE_ACCOUNT_ID=([A-Za-z0-9-]+)", env)[0]
    key_id = re.findall("ID=([A-Za-z0-9-]+)", env)[0]
    private_key = (
        re.findall("PRIVATE_KEY='(.+)'", env)[0].replace("\\r", "").replace("\\n", "\n")
    )

    os.environ["SERVICE_ACCOUNT_ID"] = service_account_id
    os.environ["ID"] = key_id
    os.environ["PRIVATE_KEY"] = private_key

    st.text("Загрузка файла произошла успешно")


if st.session_state["env_loaded"]:
    llm = YandexGPT()
    chatbot = Chatbot(llm)


# Виджет кнопки для показа случайной строки
show_random_row_button = st.button("Show Random Row")


def get_random_row(data):
    random_index = random.randint(0, len(data) - 1)
    return random_index


if show_random_row_button:
    st.session_state["n_row"] = get_random_row(df_view)

st.table(df_view.iloc[st.session_state["n_row"]])

st.subheader("Поддержка Яндекс.Лавки")

widgets_container = st.empty()

# Text input for user
user_input = widgets_container.text_input(
    "You: ",
    value=df.iloc[st.session_state["n_row"]]["Обращение"],
    placeholder="Ask me anything ...",
    key="user_input",
)

row = df_view.iloc[st.session_state["n_row"]].to_frame().T

col1, col2, col3 = st.columns(3)

button_send = col3.button('Send')
with col2:
    st.checkbox("Show thoughts:", key="debug")


if button_send:
    answer = chatbot(user_input, "", row, debug=st.session_state["debug"])
    st.text_area("Answer", value=answer, label_visibility="hidden")

try:
    if 'c1_class' in answer:
        st.text('''Класс 1 - В сообщении говорится о том, что заказ вообще не привезли или вопросы к курьеру - т.е. нет упоминания о товаре.
                Класс 2 - В сообщении говорится о том, что заказ привезли, но проблема с каким-то товаром - т.е. есть упоминание о товаре.
                Класс 3 - Сообщение не относится к классу 1 и 2.''')
        st.text('Класс выбранный моделью: ' + answer['c1_class'])
except:
    pass

try:
    if 'sub_class' in answer:
        st.text('''ПодКласс 1 - В сообщении говорится, что вообще не привезли какой-то товар.
            ПодКласс 2 - В сообщении говорится, что привезли меньше товара, чем заказано.
            ПодКласс 3 - В сообщении говорится, что товар испорчен.
            ПодКласс 4 - Сообщение не относится к классу 1, 2 и 3.''')
        st.text('ПодКласс выбранный моделью: ' + answer['sub_class'])
    
except:
    pass

try:
    if 'coupon_dict' in answer:
        st.text('Это инфа о купоне' + str(answer['coupon_dict']))
except:
    pass

try:
    if 'response' in answer:
        st.text('Ответ:' + answer['response'])
except:
    pass
