FROM python:3.7

WORKDIR /app

RUN wget https://justinsvegliato.com/cplex/cplex_studio1210.linux-x86-64.bin
RUN chmod +x cplex_studio1210.linux-x86-64.bin

COPY . .

RUN apt-get update
RUN apt-get -y install vim
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# DOCKER COMMANDS
# Build Image: docker build -t metareasoning-for-state-abstractions metareasoning-for-state-abstractions
# Run Container: docker run -i -t metareasoning-for-state-abstractions /bin/bash
# Delete Image: docker image rm -f metareasoning-for-state-abstractions