# AGI Fireworks Яндекс Лавка чатбот поддержки на YandexGPT
## install
```pip install -r requirements.txt```

OR USE https://lavkagpt.streamlit.app/

## .env

To access the YandexGPT API, you will need a service account with the `ai.languageModels.user` role in the cloud with access to YandexGPT.
1. Create a service account by following [this guide](https://cloud.yandex.ru/docs/iam/operations/sa/create).
2. [Specify](https://cloud.yandex.ru/docs/datasphere/operations/projects/update) this service account in the DataSphere project settings.
3. Enter the service account ID in the cell below.
4. [Create](https://cloud.yandex.ru/docs/iam/operations/authorized-key/create) an authorized key for the service account. 
5. Save the key value in the `private-key` secret. To learn how you can create a secret, see [this guide](https://cloud.yandex.ru/docs/datasphere/operations/data/secrets).
6. Create .env with following fields

```
ID=
SERVICE_ACCOUNT_ID=
CREATED_AT=2023-08-07T16:24:04.603231529Z
KEY_ALGORITHM=RSA_2048
PUBLIC_KEY='-----BEGIN PUBLIC KEY-----
-----END PUBLIC KEY-----
'
PRIVATE_KEY='-----BEGIN PRIVATE KEY-----
-----END PUBLIC KEY-----
'
```

## Run

```bash
streamlit run ./streamlit.py
```

OR

https://lavkagpt.streamlit.app/

## Upload .env file
![Alt text](image.png)

## Use

![Alt text](image-1.png)


Информация о заказе

![Alt text](image-2.png)

Введите запрос, нажмите `Send`

If you check the show_thoughts box, then internal requests and responses will appear.
