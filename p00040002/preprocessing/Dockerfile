FROM python

COPY . ./preprocessing

WORKDIR /preprocessing

RUN pip install -r preprocessing_requirements.txt

CMD ["python", "preprocessing.py"]
