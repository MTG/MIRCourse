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
In terminal, cd to your local folder for this repository and run the following command:
```
sudo docker-compose up
```

This would install a docker image (first time, it would download the image of size ~2Gb) and provide a web link (http://0.0.0.0:8888). Clicking or copy-pasting this link to a browser, one can access ipython notebooks (Windows users: http://{YourIPaddress}:8888). The password required is mir.

Then, you can access the notebooks of the course from the browser and run them. All data used in the notebooks are not included in this repository due to size concerns. Some of the notebooks require downloading data from Freesound and Dunya using user specific tokens (hence would require that you get a user token and use that). Please refer to notebooks "DownloadDataFrom*" for more info.

## Notebooks:
For half of the tasks/examples, there are two notebook versions: 'LectureX.ipynb' and 'LectureX_solution.ipynb'. The first one is a student version where part of the code is missing (marked with: "Your code starts here" ..."Your code ends here"). The expected output is also involved in the notebook but since part of code is missing, re-running it will not re-produce the same output (unless you fill the expected parts).
The second version contains a solution (not the solution) and is complete to produce the expected outcome.

Installation notes on the course web site:
https://sites.google.com/site/mirspring2018/installation
