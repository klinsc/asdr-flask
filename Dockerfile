# create a docker image for the application of flask with python3.11.4

FROM python:3.11.4

# set the working directory in the container
WORKDIR /asdr-flask

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY . .

# run prisma generate for python
RUN prisma generate

# command to run on container start
CMD [ "python", "./main.py" ]

# build the docker image
# docker build -t asdr-flask .
