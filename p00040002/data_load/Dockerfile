FROM python

COPY . ./data_load

WORKDIR /data_load

RUN pip install -r load_requirements.txt

CMD ["python", "data_load.py"]
