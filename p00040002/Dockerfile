FROM python:3.7.3

COPY . ./p00040002

WORKDIR /p00040002

RUN apt-get update && apt-get install -y default-jre

RUN pip install --upgrade pip 

RUN pip install -r train_requirements.txt

CMD ["python", "p00040002.py"]
