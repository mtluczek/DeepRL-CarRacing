# Use python3.8
FROM python:3.8

LABEL maintainer="OPENAI GYM"
# The parts of the scripts are from https://github.com/TTitcombe/docker_openai_gym

# Working directory is / by default. We explictly state it here for posterity
WORKDIR /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       apt-utils \
       build-essential \
       curl \
       xvfb \
       ffmpeg \
       xorg-dev \
       libsdl2-dev \
       swig \
       cmake

# Upgrade pip3
RUN pip3 install --upgrade pip

# Install the python requirements on the image
RUN pip3 install --trusted-host pypi.python.org numpy stable_baselines3 gym box2d-py pyopengl torch matplotlib seaborn pandas

RUN pip3 install --trusted-host pypi.python.org gym[box2d]

RUN pip3 install --trusted-host pypi.python.org opencv-python

# Create a directory in which we can do our work
RUN mkdir /home/my_rl/

# Set it as the working directory
WORKDIR /home/my_rl/

COPY ./ ./

# Copy over the start-up script
# ADD scripts/startup_script.sh /usr/local/bin/startup_script.sh

# Give permissions to execute
RUN chmod 777 scripts/startup_script.sh

# Set the display when we run the container. This allows us to record without the user needing to type anything explicitly
# This code snippet was taken from https://github.com/duckietown/gym-duckietown/issues/123
ENTRYPOINT ["scripts/startup_script.sh"]