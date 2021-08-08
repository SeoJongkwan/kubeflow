FROM python

COPY . ./model_train

WORKDIR /model_train

RUN pip install -r train_requirements.txt

CMD ["python", "model_train.py"]
