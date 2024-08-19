# FROM python:3.8-alpine
FROM python:3.8-slim-buster

LABEL author="Marc Duby, Broad Institute"
LABEL description="The Flannick's Lab's Gene Set Grouping Application"

# update the image
# RUN apk add --update git
# RUN apk add --update bash
RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install -y bash

# # update for the python libs
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Pull the code from git
RUN mkdir /home/CodeAplication       
RUN cd /home/CodeAplication 
# RUN git clone -b md_mysql_batch https://github.com/broadinstitute/genetics-kp-dev /home/CodeTest/GeneticsPro
RUN git clone https://github.com/broadinstitute/dcc_pigean_flask.git /home/CodeAplication/GeneSetNmf
RUN cd /home/CodeAplication/GeneSetNmf

# install python libraries
RUN pip3 install scikit-learn==1.3.2
RUN pip3 install flask==3.0.3
RUN  pip3 install gunicorn==22.0.0

#create the logs directory
RUN mkdir /home/CodeAplication/GeneSetNmf/python-flask-server/logs

# set working directory
# WORKDIR /home/CodeTest/GeneticsPro/python-flask-server
WORKDIR /home/CodeAplication/GeneSetNmf/python-flask-server

# CMD cat /proc/version
CMD gunicorn -w 4 --bind 0.0.0.0:8082 app:app --timeout 3600
# CMD . /home/CodeTest/GeneticsPro/Test/echo_env.txt

