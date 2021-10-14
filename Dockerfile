FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

COPY requirements.txt /.

RUN apt-get update

RUN apt-get install -y python3-pip

RUN apt-get install -y build-essential libssl-dev libffi-dev python3-dev git

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt --user
