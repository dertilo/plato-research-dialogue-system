FROM ufoym/deepo:tensorflow-py36-cu100

RUN apt-get update && \
    apt-get install -y libgmp-dev libmpfr-dev libmpc-dev libsndfile1 && \
    apt-get install -y python3-pyaudio

WORKDIR /docker-share
COPY requirements.txt ./

RUN pip install -r requirements.txt

