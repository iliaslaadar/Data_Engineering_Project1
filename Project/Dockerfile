FROM python:3.8

WORKDIR /home

ENV FLASK_APP=app.py

COPY requirements .

RUN pip install -r requirements

COPY . .

EXPOSE 5000

CMD ["python","app.py"]
