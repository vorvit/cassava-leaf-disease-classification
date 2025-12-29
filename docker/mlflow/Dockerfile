FROM python:3.12-slim

RUN python -m pip install --no-cache-dir --upgrade pip \
  && python -m pip install --no-cache-dir "mlflow==3.8.1" "gunicorn==23.0.0"

WORKDIR /mlflow

EXPOSE 8080

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "8080", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlflow/mlruns"]
