FROM python:3.11-slim-bookworm

WORKDIR /app

copy . /app

RUN pip install -r requirements.txt

CMD [ "python3", "flask_app/app.py" ]
