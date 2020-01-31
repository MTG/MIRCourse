# MIRCourse

This repository contains a docker-compose file to run a Jupyter server and the notebooks used during Spring trimester of the MIR course at MTG/UPF. To run the notebooks, you need to first install docker and run the Jupyter server available in the docker image.

## Install docker

### Windows
https://docs.docker.com/docker-for-windows/install/

### Mac
https://docs.docker.com/docker-for-mac/install/

### Ubuntu
https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-docker-ce

## Running the Jupyter server 
In a terminal/console window, change to this directory

On MacOS or Windows, run:

    docker-compose up

On Linux, run the following (this command ensures that any files you create are owned by your own user):

    JUPYTER_USER_ID=$(id -u) docker-compose up

The first time you run this command it will download the required docker images (about 2GB in size). If you have previously downloaded the images and would like to update them with the last version, run:

    docker-compose pull

Then accesss http://localhost:8888 with your browser and when asked for a
password use the default password ***mir***

Then, you can access the notebooks of the course from the browser and run them. All data used in the notebooks are not included in this repository due to size concerns. Some of the notebooks require downloading data from Freesound and Dunya using user specific tokens (hence would require that you get a user token and use that). Please refer to notebooks "DownloadDataFrom*" for more info.

## Notebooks:
For half of the tasks/examples, there are two notebook versions: 'LectureX.ipynb' and 'LectureX_solution.ipynb'. The first one is a student version where part of the code is missing (marked with: "Your code starts here" ..."Your code ends here"). The expected output is also involved in the notebook but since part of code is missing, re-running it will not re-produce the same output (unless you fill the expected parts).
The second version contains a solution (not the solution) and is complete to produce the expected outcome.

## Other help pages:	
Installation notes on the course web site:
https://sites.google.com/site/mirspring2018/installation

The course largely uses Essentia algorithms (available in the docker image) for feature extraction. For quick tutorials on how to import and call Essentia algorithms, you can refer to https://essentia.upf.edu/documentation/essentia_python_tutorial.html
