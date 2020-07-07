# Topic Emergence ADS

Creating a measure for topic emergence in the Astrophysics Data Sytem (ADS).  

See all options by going to root of this repository and running `make`.

- [Installation](#installation)
- [Running](#running)
- [App](#app)


## Installation
**Requirements**:
 - [GNU make](https://www.gnu.org/software/make/). Tested on [this](#make-version) version.
 - Python 3.7. Tested on Python 3.7.6.

First, you will want to be working in a virtual environment. For example, you could make one using python's [built-in venv module](https://docs.python.org/3/library/venv.html).
```bash
python -m venv my_env
```
You can then activate the environment with `source my_env/bin/activate`.

While in the virtual environment, you can now install the python requirements with `make requirements`.

If you plan to use the `docker-build-app` and `docker-run-app` commands, you will also need to install docker.

## Running
The commands listed after each step in the pipeline can all be found in the [Makefile](Makefile). You can see all of the options by simple running `make` or `make help`.

Here is how to simply run an example experiment:
1) Run `make db lda CONFIG_FILE=config/example_config.mk`
2) Decide which topic model to use for the TEV. See `reports/example_experiment/` for coherence plot and data. For this example let's say you choose the topic model with 10 topics.
3) Run `make viz CONFIG_FILE=config/example_config.mk N_TOPICS=30`.
4) First you will need to make an [ADS account](https://ui.adsabs.harvard.edu/user/account/register) and obtain an API token [here](https://ui.adsabs.harvard.edu/user/settings/token). Then, run:
    ```bash
    make link-data-to-app app-dev \ 
        CONFIG_FILE=config/example_experiment.mk \
        N_TOPICS=30 \
        ADS_TOKEN=YOUR-ADS-TOKEN
     ```
5) Visit `localhost:5000` to see the Topic Emergence Visualizer for your experiment.

## Configuration

In order to tailor the experiment to your use case, you need to understand the configuration files.

There are three configuration files:
1) *Main makefile config* -- Make configuration for naming experiment, parameters which change performance of pipeline, and references to other configuration files. Example [here](config/example_experiment.mk).
2) *parameter config* -- YAML configuration that changes how the experiment is conducted. Example [here](config/example_config.yaml).
3) *drop features* -- a txt file with one keyword per line. It is a blacklist of keywords which should be ignored. Example [here](config/drop_features.txt).

More details about these options are contained within the example configuration files.

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