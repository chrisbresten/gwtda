FROM nvcr.io/nvidia/tensorflow:21.11-tf2-py3 
RUN apt-get update 
ENV DEBIAN_FRONTEND=noninteractive
RUN set -e \
&&  apt update && DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends --yes \
python3-psycopg2
RUN pip3 install tenacity
RUN pip3 install gwtools
RUN pip3 install pyastronomy
RUN pip3 install gudhi

RUN rm --recursive --force \
    /usr/share/doc/* \
    /usr/share/man/* \
    /var/cache/apt/archives/*.deb \
    /var/cache/apt/archives/partial/*.deb \
    /var/cache/apt/*.bin \
    /var/cache/debconf/*.old \
    /var/log/apt \
    /var/log/*.log 


