# Topic Emergence

Creating a measure for topic emergence in the Astrophysics Data Sytem (ADS).  

See all options by going to root of this repository and running `make`.

- [Installation](#installation)
- [Data Pipeline](#data-pipeline)
- [App](#app)


## Installation
**Requirements**:
 - [GNU make](https://www.gnu.org/software/make/). Tested on [this](#make-version) version.
 - Python 3.7. Tested on Python 3.7.6.
 - [Docker](https://www.docker.com/). Tested on `Docker version 19.03.5, build 633a0ea`.

First, you will want to be working in a virtual environement. For example, you could make once using python's [built-in venv module](https://docs.python.org/3/library/venv.html).
```bash
python -m venv my_env
```
You can then activate the environment with `source venv/bin/activate`.

While in the virtual environment, you can now install the python requirements with `make requirements`.

If you plan to use the `docker-build-app` and `docker-run-app` commands, you will also need to install docker.

## Data Pipeline
The commands listed after each step in the pipeline can all be found in the [Makefile](Makefile).

The pipeline takes the input corpus, extracts potential keywords from the abstracts and titles, and counts frequencies for those keywords for each year (`join-and-clean`, `docs-to-keywords-df`, `get-filtered-keywords`).  
  
It then normalizes these frequency counts by the total count of papers for each year. The ratio of each year's frequency as compared to the first non-zero keyword count is also computed for each keyword (`normalize-keyword-freqs`).
 
From these keyword frequencies time series, a number of features are extracted and saved (`slope-complexity`).
 
Also from the normalized keyword frequencies, dynamic time warps are calculated pairwise between all keywords. Then kmeans clustering is performed on this matrix of dynamic time warps (`dtw`, `cluster-tests`, `dtw-viz`).
 
Finally, a flask app can run which displays a scatter plot with each keyword time series as a point. The axes are each one of the extracted time series features. The bubbles are sized by the log of their counts and colored by their dynamic time warp kmeans cluster (`link-data-to-app`, `app`).

See list of all options with descriptions by running `make`.
 
## App
To just run the app from a docker image, first pull or build the keyword-emergence-visualizer docker image (Dockerfile [here](app/Dockerfile)). You can see available image tags [here](https://storage.analytics.nasa.gov/repository/datasquad/keyword-emergence-visualizer). For example, you can pull the `latest` image with:
 ```
 docker pull storage.analytics.nasa.gov/datasquad/keyword-emergence-visualizer:latest
```
Once you have the image on your machine, you can run it, being sure to correctly configure your ports and docker volume mappings. For example:
```bash
cd app
export IMAGE=storage.analytics.nasa.gov/datasquad/keyword-emergence-visualizer:latest
docker run -it -p 5002:5000 -v $(pwd)/data:/home/data $IMAGE
```

For development purposes, you can run the app with the make commands `app` or `docker-run-app`. The first command will depend upon having the local environment properly configured with all requirements installed. The second requires only the docker container and some information from git about commits, remotes, and tags. The `docker-build-app` and `docker-run-app` commands service to automatically supply the image with some information about the git repo. 

## Index

### make version
```txt
GNU Make 3.81
Copyright (C) 2006  Free Software Foundation, Inc.
This is free software; see the source for copying conditions.
There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.

This program built for i386-apple-darwin11.3.0
```