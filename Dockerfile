FROM civisanalytics/datascience-python

RUN apt-get update && \
    apt-get install -y libgmp-dev libmpfr-dev libmpc-dev && \
    apt-get install -y python3-pyaudio

WORKDIR /docker-share
COPY requirements.txt ./

RUN pip install -r requirements.txt

