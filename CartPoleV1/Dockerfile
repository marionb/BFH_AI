# FROM ubuntu:22.04
FROM python:3.10-buster

RUN apt-get update && \
    # apt-get install -y locales && \
    # apt-get install -y software-properties-common && \
    # add-apt-repository ppa:deadsnakes/ppa && \
    # apt install -y python3.10 python3-pip && \
	# localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
    python3-pip \
    python3-venv \
    virtualenv \
    vim \
    swig

COPY solution /scripts/
COPY requirements.txt /scripts/.

# installation of pytorch with download
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --disable-pip-version-check --no-cache-dir -r /scripts/requirements.txt

ENTRYPOINT ["tail", "-f", "/dev/null"]