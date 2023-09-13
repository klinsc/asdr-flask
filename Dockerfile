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


# creat worker from ubuntu and do the following:
####################
# 1) Install Poppler
# wget https://poppler.freedesktop.org/poppler-21.09.0.tar.xz
# tar -xvf poppler-21.09.0.tar.xz
# sudo apt-get install libnss3 libnss3-dev
# sudo apt-get install libcairo2-dev libjpeg-dev libgif-dev
# sudo apt-get install cmake libblkid-dev e2fslibs-dev libboost-all-dev libaudit-dev
# cd poppler-21.09.0/
# mkdir build
# cd build/
# cmake  -DCMAKE_BUILD_TYPE=Release   \
#        -DCMAKE_INSTALL_PREFIX=/usr  \
#        -DTESTDATADIR=$PWD/testfiles \
#        -DENABLE_UNSTABLE_API_ABI_HEADERS=ON \
#        ..
# make 
# sudo make install
####################
# 1.1) OR by official documenation
# conda install poppler
# apt-get install poppler-utils
####################
# 2) Fix "ImportError: libGL.so.1: cannot open shared object file: No such file or directory"
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
