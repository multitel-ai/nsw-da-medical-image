FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# To fix GPG key error when running apt-get update

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean 

COPY ./requirements.txt /

RUN pip install -r /requirements.txt

RUN pip install --upgrade diffusers[torch]

WORKDIR /App
