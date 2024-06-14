# create a docker image for the application of flask with python3.10.12

FROM python:3.10.12

# set the working directory in the container
WORKDIR /app

# copy the dependencies file to the working directory
COPY requirements.txt .

# # upgrade apt-get
RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt install libgl1-mesa-glx -y

# install poppler
RUN apt-get update && apt-get install poppler-utils -y

RUN apt update && apt install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1

# install dependencies
RUN pip install -r requirements.txt

# install mim
RUN pip install openmim -q
RUN mim install "mmengine>=0.6.0" -q
RUN mim install "mmcv>=2.0.0rc4,<2.1.0" -q
RUN mim install "mmdet>=3.0.0,<4.0.0" -q

# copy the content of the local src directory to the working directory
COPY . .

# install prisma
RUN pip install -U prisma

# run prisma generate for python
RUN prisma generate

# command to run on container start
CMD ["uwsgi", "--socket", "0.0.0.0:5000", "--protocol=http", "-w", "wsgi:app", "--master", "--processes", "4", "--threads", "2"]

# build the docker image, with .env file, and name it asdr-flask
# docker build -t asdr-flask .

# run the docker image, expose port 5000 as 5666, with name asdr-flask, and run in background
# docker run -d -p 5666:5000 --name asdr-flask asdr-flask

