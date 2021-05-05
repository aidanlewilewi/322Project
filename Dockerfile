# a Dockerfile specifies how to build a Docker image
FROM continuumio/anaconda3:2020.11

ADD . /code
WORKDIR /code

ENTRYPOINT ["python", "airline_app.py"]