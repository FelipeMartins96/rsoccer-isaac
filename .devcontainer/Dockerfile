FROM isaacgym:latest

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    xvfb \
    ffmpeg

WORKDIR /opt

RUN pip install --upgrade pip & \
    pip install black moviepy pandas

RUN git clone \
    https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git && \
    cd IsaacGymEnvs && \
    git checkout ca7a4fb

RUN cd IsaacGymEnvs &&\
    pip install -e .

COPY --chown=gymuser . .
WORKDIR /opt/rsoccer-isaac

USER gymuser