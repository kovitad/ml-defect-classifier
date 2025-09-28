FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY config /app/config
COPY data /app/data
COPY models /app/models
COPY src /app/src
COPY main.py /app/main.py

ENV TZ=Asia/Bangkok
ENV MLFLOW_TRACKING_URI=https://dta-mlflow.dta-test.kube.kasikornbank.com

CMD ["python", "/app/main.py"]
