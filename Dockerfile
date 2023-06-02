FROM python:3.9.16-buster

COPY /model /model
COPY requirements.txt requirements.txt
COPY main.py main.py
COPY repurchase.py repurchase.py

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn main:app --host 0.0.0.0 --port $PORT