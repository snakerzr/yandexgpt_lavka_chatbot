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
st.title("Chat Hueta")
st.header("От создателей Сашка и Петушка")


st.header("Загрузите файл с env")
uploaded_file = st.file_uploader(
    "Choose a file env", label_visibility="hidden", type=".env"
)


def return_text(var):
    return var


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
    st.write(os.environ["ID"])
    llm = YandexGPT()
    st.write(llm.key_id)
    st.write(llm.service_account_id)
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

st.text_area("", value="""Добрый день! Вас беспокоит техническая поддержка""")


# Text input for user
user_input = st.text_input("You: ", placeholder="Ask me anything ...", key="input")

row = df_view.iloc[st.session_state["n_row"]].to_frame().T

col1, col2, col3 = st.columns(3)
if col1.button("Example"):
    # Если он жмёт пример, то выдаём определенное обращение и запускаем Yandex GPT, после выдает ответ.
    st.text_area("", value="""Добрый день! Вас беспокоит техническая поддержка""")

with col2:
    st.checkbox("Show thoughts:", key="debug")

if col3.button("Send"):
    answer = chatbot(user_input, "", row, debug=st.session_state["debug"])
    st.text_area("", value=answer)


#
# if title:
#     text_gpt = YandexGPT()
#     st.text_area('', value=text_gpt)
