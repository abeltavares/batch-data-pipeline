FROM apache/airflow:2.9.0-python3.11

USER root

COPY requirements.txt /requirements.txt
RUN chmod 644 /requirements.txt

# Use default Airflow user to install dependencies
USER 50000

RUN pip install --no-cache-dir apache-airflow==2.9.0 -r /requirements.txt
