import random
import pandas as pd
from src.utils import load_df, random_id
from src.YandexGPT import YandexGPT
from langchain import PromptTemplate, LLMChain


class Chatbot:
    transfer_to_operator: str = "Я либо не понял, что вы имеете ввиду, либо пока не умею обрабатывать такие жалобы. По-этому перевожу вас на оператора. Вам точно помогут!"
    greetings: str = """Здравствуйте! 
    Я экспериментальный чатбот службы поддержки Яндекс.Лавки. Пока я умею обрабатывать только жалобы на товары в заказе.
    В будущем научусь многому другому.
    Скажите, чем я могу Вам помочь?"""
    response: str = "Да"

    def __init__(self, llm):
        self.llm = llm

    def _send_message_to_client(self, str_: str):
        # Send message to client function
        print(str_)

    def _await_response(self, str_: str):
        self.response = str_

    def _send_greetings(self):
        ## Greetings prompt
        self._send_message_to_client(self.greetings)

    def _classification_1(self, message: str) -> str:
        classification_1_prompt = """Сообщение можно отнести только к 3-м классам.
        
        Класс 1 - В сообщении говорится о том, что заказ вообще не привезли или вопросы к курьеру - т.е. нет упоминания о товаре.
        Класс 2 - В сообщении говорится о том, что заказ привезли, но проблема с каким-то товаром - т.е. есть упоминание о товаре.
        Класс 3 - Сообщение не относится к классу 1 и 2.
        
        Тебе нужно написать только номер класса, к которому относится сообщение.
        
        СООБЩЕНИЕ:<<<{message}>>>"""

        с1_chain = LLMChain(
            llm=self.llm, prompt=PromptTemplate.from_template(classification_1_prompt)
        )
        for _ in range(3):
            try:
                c1_message_class = с1_chain(message)["text"]
                return c1_message_class
            except:
                continue

    def _summary_generation(self, message: str) -> str:
        ## Summary
        summary_prompt = """СООБЩЕНИЕ:<<{message}>> О чем речь кратко?"""

        summary_chain = LLMChain(
            llm=self.llm, prompt=PromptTemplate.from_template(summary_prompt)
        )
        for _ in range(3):
            try:
                message_summary = summary_chain(message)["text"]
                return message_summary
            except:
                continue

    def _ner(self, message: str) -> str:
        # NER
        ner_prompt = """СООБЩЕНИЕ:<<{message}>> С каким товаром или товарами проблемы в этом сообщении? 
        Напиши только название товара, с которым произошли проблемы. Нужно написать одно слово, если говорится об одном товаре и несколько слов, если говорится о нескольких товарах.
        """
        two_ner_promt = """СООБЩЕНИЕ: <<{troubled_product}>> Вытащи название товара, о котором идёт речь в сообщении. На выходе хочу получить исключительно название товара, это 1 или 2 слова. Пиши не в кавычках
        """

        ner_chain = LLMChain(
            llm=self.llm, prompt=PromptTemplate.from_template(ner_prompt)
        )

        ner_chain_two = LLMChain(
            llm=self.llm, prompt=PromptTemplate.from_template(two_ner_promt)
        )
        for _ in range(3):
            try:
                troubled_product = ner_chain(message)["text"]
                if len(troubled_product.split(" ")) > 2:
                    for _ in range(3):
                        try:
                            troubled_product = ner_chain_two(troubled_product)["text"]
                            return troubled_product
                        except:
                            continue

                return troubled_product
            except:
                continue

    def _sentiment_recognition(self, message: str) -> str:
        ## Sentiment recognition

        sentiment_recognition_prompt = """ИНСТРУКЦИЯ:<<Напиши сентимент сообщения одним словом.>>
        
        СООБЩЕНИЕ:<<{message}>>"""

        sentiment_recognition_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(sentiment_recognition_prompt),
        )
        for _ in range(3):
            try:
                sentiment = sentiment_recognition_chain(message)["text"]
                return sentiment
            except:
                continue

    def _classification_product(self, message: str) -> str:
        ## Подкатегории проблем с товарами (Классификатор)

        classification_product_prompt = """Сообщение можно отнести только одному из классов.
        
        Класс 1 - В сообщении говорится, что вообще не привезли какой-то товар.
        Класс 2 - В сообщении говорится, что привезли меньше товара, чем заказано.
        Класс 3 - В сообщении говорится, что товар испорчен.
        Класс 4 - Сообщение не относится к классу 1, 2 и 3.
        
        Тебе нужно написать только номер класса, к которому относится сообщение.
        {message}"""

        classification_product_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(classification_product_prompt),
        )
        for _ in range(3):
            try:
                message_subclass = classification_product_chain(message)["text"]
                return message_subclass
            except:
                continue

    def _inclusion_check(self, order_list, troubled_product) -> str:
        ## Inclusion check func
        inclusion_check_prompt = """Есть ли в списке товаров {order_list} упоминание в каком-либо виде о {troubled_product}.
        Если есть напиши 1, если нет 0."""
        inclusion_check_chain = LLMChain(
            llm=self.llm, prompt=PromptTemplate.from_template(inclusion_check_prompt)
        )

        for _ in range(3):
            try:
                inclusion_check = inclusion_check_chain(
                    {"order_list": order_list, "troubled_product": troubled_product}
                )["text"]
                return inclusion_check
            except:
                continue

    def _find_index_different_names(self, troubled_product, order_list):
        """Это сложное колдовство, вы не понимаете.

        Args:
            troubled_product (str): _description_
            order_list (list): _description_

        Returns:
            int: _description_
        """
        troubled_product_index = 0

        check_product_prompt = (
            '''{troubled_product} и {product} это одно и то же? Ответь "Да" или "Нет"'''
        )

        for i, product in enumerate(order_list):
            check_product_chain = LLMChain(
                llm=self.llm, prompt=PromptTemplate.from_template(check_product_prompt)
            )

            try:
                check_product_message = check_product_chain(
                    {"troubled_product": troubled_product, "product": product}
                )["text"].lower()
                if "да" in check_product_message[0:4]:
                    troubled_product_index += i
                    break

            except:
                check_product_message = check_product_chain(
                    {"troubled_product": troubled_product, "product": product}
                )["text"].lower()
                if "да" in check_product_message[0:4]:
                    troubled_product_index += i
                    break

        return troubled_product_index

    def _get_troubled_product_total_price(
        self, troubled_product, order_list, product_quantities, product_prices
    ):
        troubled_product_index = self._find_index_different_names(
            troubled_product, order_list
        )
        troubled_price = product_prices[troubled_product_index]
        troubled_amount = product_quantities[troubled_product_index]
        return troubled_price * troubled_amount

    def _generate_coupon(self, troubled_total_cost):
        ## Generate coupon func
        coupon_name = "".join(
            random.choices(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=12
            )
        )
        coupon_exp_date = "2024-01-01"
        return coupon_name, coupon_exp_date, troubled_total_cost

    def _subclass_1_check(
        self, troubled_product, order_list, product_quantities, product_prices
    ) -> str:
        ## Subclass 1 func

        inclusion_check_val = self._inclusion_check(order_list, troubled_product)

        if inclusion_check_val == "1":
            complaint_valid = True
        else:
            complaint_valid = False

        if not complaint_valid:
            # Нет товара в заказе.
            invalid_complaint_message = """Не могу найти такой товар в заказе. Видимо, произошла какая-то ошибка. Перевожу на оператора."""
            self._send_message_to_client(invalid_complaint_message)
            return None
        elif complaint_valid:
            # Есть товар в заказе.
            troubled_total_cost = self._get_troubled_product_total_price(
                troubled_product, order_list, product_quantities, product_prices
            )
            coupon_name, coupon_exp_date, troubled_total_cost = self._generate_coupon(
                troubled_total_cost
            )
            return {
                "coupon_name": coupon_name,
                "coupon_exp_date": coupon_exp_date,
                "troubled_total_cost": troubled_total_cost,
            }

    def _product_not_delivered_message_generation(self, dict_) -> str:
        product_not_delivered_message_prompt = """Тебе клиент написал сообщение: вот его суть [{summary}].
        Извинись за предоставленные неудобства и дай информацию о купоне, который компенсирует стоимость недоставленных товаров, 
        НАЗВАНИЕ КУПОНА [{coupon_name}], и СРОК ДЕЙСТВИЯ КУПОНА [{coupon_exp_date}].
        В конце скажи про РАЗМЕР СКИДКИ: {troubled_total_cost}
        """
        high_temp_llm = YandexGPT(temperature=0.001)
        product_not_delivered_message_chain = LLMChain(
            llm=high_temp_llm,
            prompt=PromptTemplate.from_template(product_not_delivered_message_prompt),
        )

        for _ in range(3):
            try:
                product_not_delivered_message = product_not_delivered_message_chain(
                    dict_
                )["text"]
                return product_not_delivered_message
            except:
                continue

    def _amount_check(self, message: str) -> str:
        ## Amount check func

        amount_check_prompt = """В сообщении описана жалоба о том, что привезли какого-то товара меньше, чем заказал клиент.
        Тебе необходимо написать только число, относящееся к тому, сколько товара привезли по факту. Только число, без обозначений.
        СООБЩЕНИЕ:<{message}>
        """
        amount_check_chain = LLMChain(
            llm=self.llm, prompt=PromptTemplate.from_template(amount_check_prompt)
        )

        for _ in range(3):
            try:
                amount_check = amount_check_chain(message)["text"]
            except:
                continue

        return amount_check

    def _get_troubled_product_price(
        self,
        troubled_product: str,
        troubled_amount: int,
        order_list: list,
        product_prices: list,
        product_quantities: list,
    ) -> int:
        """A function that returns the total cost of the product that the problems occurred with."""

        troubled_product_index = self._find_index_different_names(
            troubled_product, order_list
        )
        troubled_price = product_prices[troubled_product_index]
        troubled_amount = (product_quantities[troubled_product_index]) - troubled_amount

        return troubled_price * troubled_amount

    def _subclass_2_check(
        self, troubled_product, order_list, product_quantities, product_prices, message
    ) -> str:
        ## Подкласс 2 - привезли меньше товара, чем заказано
        inclusion_check_val = self._inclusion_check(order_list, troubled_product)
        check_1 = inclusion_check_val == "1"

        troubled_product_amount = int(self._amount_check(message))

        check_2 = (
            troubled_product_amount
            < product_quantities[
                self._find_index_different_names(troubled_product, order_list)
            ]
        )

        if check_1 and check_2:
            complaint_valid = True
        else:
            complaint_valid = False

        if not complaint_valid:
            # Нет товара в заказе или че-то не то с кол-вом
            invalid_complaint_message = """Не могу найти такой товар в заказе или что-то не то с количеством. Видимо, произошла какая-то ошибка. Перевожу на оператора."""
            self._send_message_to_client(invalid_complaint_message)
            return None
        elif complaint_valid:
            # Есть товар в заказе.
            troubled_total_cost = self._get_troubled_product_price(
                troubled_product,
                troubled_product_amount,
                order_list,
                product_prices,
                product_quantities,
            )
            coupon_name, coupon_exp_date, troubled_total_cost = self._generate_coupon(
                troubled_total_cost
            )
            return {
                "coupon_name": coupon_name,
                "coupon_exp_date": coupon_exp_date,
                "troubled_total_cost": troubled_total_cost,
            }

    def _product_partly_delivered_message_generation(self, dict_) -> str:
        product_not_delivered_message_prompt = """Тебе клиент написал сообщение: вот его суть [{summary}].
        Извинись за предоставленные неудобства и дай информацию о купоне, который компенсирует стоимость недоставленных товаров, 
        НАЗВАНИЕ КУПОНА [{coupon_name}], и СРОК ДЕЙСТВИЯ КУПОНА [{coupon_exp_date}].
        В конце скажи про РАЗМЕР СКИДКИ: {troubled_total_cost}
        """
        high_temp_llm = YandexGPT(temperature=0.001)
        product_not_delivered_message_chain = LLMChain(
            llm=high_temp_llm,
            prompt=PromptTemplate.from_template(product_not_delivered_message_prompt),
        )

        for _ in range(3):
            try:
                product_not_delivered_message = product_not_delivered_message_chain(
                    dict_
                )["text"]
                return product_not_delivered_message
            except:
                continue

    def _yes_no_check(self, message):
        yes_no_prompt = """В сообщении ответ на вопрос: "Можете прислать фотографию испорченного товара?"
        Если ответ положительный - напиши 1
        Если отрицательный - напиши 0
        
        СООБЩЕНИЕ:<{message}>
        """
        yes_no_check_chain = LLMChain(
            llm=self.llm, prompt=PromptTemplate.from_template(yes_no_prompt)
        )
        for _ in range(3):
            try:
                result = yes_no_check_chain(message)["text"]
                return result
            except:
                continue

    def _subclass_3_check(self, troubled_product, order_list, answer=None) -> str:
        ## Подкласс 3 - товар испорчен

        inclusion_check_val = self._inclusion_check(order_list, troubled_product)
        check_1 = inclusion_check_val == "1"

        if check_1:
            complaint_valid = True
        else:
            complaint_valid = False

        if not complaint_valid:
            # Нет товара в заказе или че-то не то с кол-вом
            invalid_complaint_message = """Не могу найти такой товар в заказе. Видимо, произошла какая-то ошибка. Перевожу на оператора."""
            self._send_message_to_client(invalid_complaint_message)
            return invalid_complaint_message
        
        elif complaint_valid:
            # Есть товар в заказе.
            photo_request_message = (
                """Можете прислать фотографию испорченного товара?"""
            )
            self._send_message_to_client(photo_request_message)
            if answer is None:
                # self._await_response()
                answer = self.response
            answer = self._yes_no_check(answer)
            if answer == "1":
                self._send_message_to_client(
                    "Нам нужно убедиться, что товар испорчен, поэтому понадобится фотография, но я пока не умею распознавать фотографии, поэтому переведу вас на оператора."
                )
                return "Нам нужно убедиться, что товар испорчен, поэтому понадобится фотография, но я пока не умею распознавать фотографии, поэтому переведу вас на оператора."
            elif answer == "0":
                cannot_send_a_photo_message = """Извините, без фото мы не можем убедиться, что товар действительно испорчен."""
                self._send_message_to_client(cannot_send_a_photo_message)
                return cannot_send_a_photo_message
            else:
                return "Что-то пошло не так. Перевожу вас на оператора."

    def __call__(self, message, id, row, debug=False):
        # Greetings
        self._send_greetings()

        c1_class = self._classification_1(message)
        print(c1_class)
        if debug:
            self._send_message_to_client(c1_class)

        # Если сообщение про доставку или что-то другое
        if c1_class in ["1", "3"]:
            self._send_message_to_client(self.transfer_to_operator)

            if debug:
                return {"c1_class": c1_class, "response": self.transfer_to_operator}
            else:
                return {"response": self.transfer_to_operator}

        # Если сообщение про проблему с товаром
        elif c1_class == "2":
            # Сбор инфы о заказе
            order_list = row["Список товаров в заказе"].values[0]
            product_quantities = row["Количество каждого товара"].values[0]
            product_prices = row["Цена каждого товара"].values[0]

            summary = self._summary_generation(message)
            troubled_product = self._ner(message)
            emotion = self._sentiment_recognition(message)
            if debug:
                print(summary)
                print(troubled_product)
                print(emotion)

            # Определяем подкласс
            sub_class = self._classification_product(message)
            if debug:
                print(sub_class)

            if sub_class == "1":
                coupon_dict = self._subclass_1_check(
                    troubled_product, order_list, product_quantities, product_prices
                )
                if debug:
                    print(coupon_dict)

                if coupon_dict is None:
                    self._send_message_to_client(self.transfer_to_operator)
                    if debug:
                        return {
                            "c1_class": c1_class,
                            "sub_class": sub_class,
                            "response": self.transfer_to_operator,
                        }
                    else:
                        return {"response": self.transfer_to_operator}

                coupon_dict["summary"] = summary

                if debug:
                    print(coupon_dict)

                response = self._product_not_delivered_message_generation(coupon_dict)
                self._send_message_to_client(response)

                if debug:
                    return {
                        "c1_class": c1_class,
                        "sub_class": sub_class,
                        "coupon_dict": coupon_dict,
                        "response": response,
                    }
                else:
                    return {"response": response}

            elif sub_class == "2":
                coupon_dict = self._subclass_2_check(
                    troubled_product,
                    order_list,
                    product_quantities,
                    product_prices,
                    message,
                )

                if debug:
                    print(coupon_dict)

                if coupon_dict is None:
                    self._send_message_to_client(self.transfer_to_operator)
                    if debug:
                        return {
                            "c1_class": c1_class,
                            "sub_class": sub_class,
                            "response": self.transfer_to_operator,
                        }
                    else:
                        return {"response": self.transfer_to_operator}

                coupon_dict["summary"] = summary

                if debug:
                    print(coupon_dict)

                response = self._product_partly_delivered_message_generation(
                    coupon_dict
                )
                self._send_message_to_client(response)
                if debug:
                    return {
                        "c1_class": c1_class,
                        "sub_class": sub_class,
                        "coupon_dict": coupon_dict,
                        "response": response,
                    }
                else:
                    return {"response": response}
            elif sub_class == "3":
                response = self._subclass_3_check(troubled_product, order_list)

                if debug:
                    return {
                        "c1_class": c1_class,
                        "sub_class": sub_class,
                        "response": response,
                    }
                else:
                    return {"response": response}

    def gather_answers(self, message, id, row):
        result_df = pd.DataFrame(
            index=[id],
            columns=[
                "c1_class",
                "summary",
                "troubled_product",
                "emotion",
                "sub_class",
                "inclusion_check_val",
                "troubled_product_index",
                "troubled_total_cost",
                "troubled_amount",
                "response",
            ],
        )

        c1_class = self._classification_1(message)
        summary = self._summary_generation(message)
        troubled_product = self._ner(message)
        emotion = self._sentiment_recognition(message)
        sub_class = self._classification_product(message)

        order_list = row["Список товаров в заказе"].values[0]
        product_quantities = row["Количество каждого товара"].values[0]
        product_prices = row["Цена каждого товара"].values[0]

        inclusion_check_val = self._inclusion_check(order_list, troubled_product)

        if inclusion_check_val == "1":
            troubled_product_index = self._find_index_different_names(
                troubled_product, order_list
            )
            troubled_total_cost = self._get_troubled_product_total_price(
                troubled_product, order_list, product_quantities, product_prices
            )
            troubled_amount = self._amount_check(message)
        else:
            troubled_product_index = -1
            troubled_total_cost = -1
            troubled_amount = -1

        response = "None1"

        if sub_class == "1":
            coupon_dict = self._subclass_1_check(
                troubled_product, order_list, product_quantities, product_prices
            )

            if coupon_dict is not None:
                coupon_dict["summary"] = summary

            response = self._product_not_delivered_message_generation(coupon_dict)

        elif sub_class == "2":
            coupon_dict = self._subclass_2_check(
                troubled_product,
                order_list,
                product_quantities,
                product_prices,
                message,
            )

            if coupon_dict is not None:
                coupon_dict["summary"] = summary

            response = self._product_partly_delivered_message_generation(coupon_dict)

        elif sub_class == "3":
            coupon_dict = self._subclass_3_check(troubled_product, order_list, "Да")
            response = "None2"

        result_df["c1_class"] = c1_class
        result_df["summary"] = summary
        result_df["troubled_product"] = troubled_product
        result_df["emotion"] = emotion
        result_df["sub_class"] = sub_class
        result_df["inclusion_check_val"] = inclusion_check_val
        result_df["troubled_product_index"] = troubled_product_index
        result_df["troubled_total_cost"] = troubled_total_cost
        result_df["troubled_amount"] = troubled_amount
        result_df["response"] = response
        return result_df


def gather_data_for_metrics(llmclass):
    df = load_df()
    result = pd.DataFrame()

    llm = llmclass()

    chatbot = Chatbot(llm)

    for i in range(20, len(df)):
        row = df.iloc[i].to_frame().T
        id = row["Уникальный айди обращения"].values[0]
        message = row["Обращение"].values[0]
        print(i)
        print(id)
        print(message)
        print("=" * 10)

        answers_df = chatbot.gather_answers(message, id, row)
        result = pd.concat([result, answers_df])
        result.to_csv("lol_ok.csv")

    # result.to_csv('lol_ok.csv')


if __name__ == "__main__":
    gather_data_for_metrics(YandexGPT)
    # df = load_df()
    # # id, message, row = random_id(df)
    # llm = YandexGPT()
    # # print(llm('Привет, как дела?'))
    # chatbot = Chatbot(llm)
    # for i in range(20,len(df)):
    #     row = df.iloc[i].to_frame().T
    #     id = row['Уникальный айди обращения'].values[0]
    #     message = row['Обращение'].values[0]
    #     print(row)
    #     print('='*10)
    #     print(id)
    #     print('='*10)
    #     print(message)
    #     # print('='*10)
    #     # print(row)
    #     print('='*10)
    #     print('='*10)
    #     print('='*10)
    #     chatbot(message,id,row,debug=True)
