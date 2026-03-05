FROM python:3.11.9-slim-buster

WORKDIR /app

copy . /app

RUN pip install -r reqirements.txt

CMD [ "python3", "app.py" ]
