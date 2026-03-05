FROM python:3.11-slim-bookworm

WORKDIR /app

copy . /app

RUN pip install -r reqirements.txt

CMD [ "python3", "app.py" ]
