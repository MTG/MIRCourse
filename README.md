# MIRCourse

This repository contains a docker-compose file to run an ipython notebook server and the notebooks used during Spring trimester of the MIR course at MTG (still in development).

## Install docker

### Windows
https://docs.docker.com/docker-for-windows/install/

### Mac
https://docs.docker.com/docker-for-mac/install/

### Ubuntu
https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-docker-ce

## Run 
First run the following command:
```
docker-compose up
```
Then accesss localhost:8888 on your browser and when asked for a password use _mir_

Then, you can access the notebooks of the course from the browser and run them. All data used in the notebooks are not included in this repository due to size concerns. Some of the notebooks require downloading data from Freesound and Dunya using user specific tokens (hence would require that you get a user token and use that). Please refer to notebooks "DownloadDataFrom*" for more info.
