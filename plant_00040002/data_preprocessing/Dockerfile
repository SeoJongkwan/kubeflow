FROM python:3.7.3
COPY . ./data_preprocessing

WORKDIR /data_preprocessing

RUN pip install -r preprocessing_requirements.txt

CMD ["python", "data_preprocessing.py"]
