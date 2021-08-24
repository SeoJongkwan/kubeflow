FROM python

COPY . ./model_train

WORKDIR /model_train

RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre

RUN pip install -r train_requirements.txt

CMD ["python", "model_train.py"]
