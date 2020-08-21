# Topic Emergence ADS
# Create topic models for Astrophysics Data System and visualize trends over time.
# author: Anthony Buonomo
# email: arb246@georgetown.edu

TIMESTAMP=$$(date +%Y-%m-%d_%H:%M:%S)

.PHONY: requirements \
		sync-raw-data-from-s3 \
		0-join-and-clean \
		1-write-ads-to-db \
		2-get-keywords-from-texts \
		3-add-missed-locations \
		4-prepare-for-lda \
		5-make-topic-models \
		6-prepare-for-topic-model-viz \
		7-get-time-chars \
		clean-experiment \
		check_clean

# Default Config Variables
EXPERIMENT_NAME=example_experiment

N_TOPICS=500
BATCH_SIZE=1000
PARAM_YAML=config/example_config.yaml
PORT=5000
INFO="This is an example experiment."

BUCKET=s3://datasquad-low/home/DataSquad/topic-emergence-ADS
PROFILE=default

# Experiment specific variables contained in CONFIG_FILE
# Values in CONFIG_FILE will overwrite the variables above
include $(CONFIG_FILE)

# Internal variables, not to be altered by CONFIG_FILE.
raw_dir=data/raw
data_dir=data/$(EXPERIMENT_NAME)
model_dir=models/$(EXPERIMENT_NAME)
reports_dir=reports/$(EXPERIMENT_NAME)
viz_dir=reports/viz/$(EXPERIMENT_NAME)
inf_dir=models/$(EXPERIMENT_NAME)/inferences

records_loc=$(data_dir)/kwds.jsonl
db_loc=$(data_dir)/ads_metadata.sqlite
lda_prep_data_dir=$(data_dir)/lda_prep_data
lda_models_dir=$(model_dir)/topic_models
lda_model_viz_data_dir=$(model_dir)/topic_model$(N_TOPICS)

raw_files=$(shell find $(raw_dir) -type f -name '*')

# Commands
## Runs 0 through 3
db: 0-join-and-clean \
	1-write-ads-to-db \
	2-get-keywords-from-texts \
	3-add-missed-locations

## Runs 4 and 5
lda: 4-prepare-for-lda \
	 5-make-topic-models

## Runs 6 and 7
viz: 6-prepare-for-topic-model-viz \
	 7-get-time-chars \

## Install packages to current environment with pip (venv recommended)
requirements:
	pip install -r requirements.txt; \
	pip install git+https://github.com/abuonomo/pyLDAvis.git; \
	python -m spacy download en_core_web_sm; \
	pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz; \

## Sync raw ADS metadata to raw data dir.
sync-raw-data-from-s3:
	aws s3 sync s3://datasquad-low/data/ADS/2020_03_15 $(raw_dir)

## 0. Join all years and and use rake to extract keywords.
0-join-and-clean: $(records_loc)
$(records_loc): $(raw_files)
	mkdir -p $(data_dir); \
	mkdir -p $(model_dir); \
	mkdir -p $(viz_dir); \
	mkdir -p $(reports_dir); \
	python src/join_and_clean.py \
		$(raw_dir) \
		$(records_loc) \
		--config_loc $(PARAM_YAML)

## 1. Write kwds.jsonl files to sqlite database
1-write-ads-to-db:
	python src/db.py write-ads-to-db \
		--infile $(records_loc) \
		--db_loc $(db_loc)

## 2. Extract keywords from papers using SingleRank and insert into database
2-get-keywords-from-texts:
	python src/db.py get-keywords-from-texts --db_loc $(db_loc) \
		--config_loc $(PARAM_YAML) \
		--batch_size $(BATCH_SIZE)

## 3. Find missed keyword locations
3-add-missed-locations:
	python src/db.py add-missed-locations --db_loc $(db_loc) --config_loc $(PARAM_YAML)

## 4. Transform data into gensim corpus and dictionary for LDA training
4-prepare-for-lda:
	python src/db.py prepare-for-lda \
		--db_loc $(db_loc) \
		--config_loc $(PARAM_YAML) \
		--prepared_data_dir $(lda_prep_data_dir)

## 5. Train topic models
5-make-topic-models:
	python src/model.py make-topic-models \
		--prepared_data_dir $(lda_prep_data_dir) \
		--config_loc $(PARAM_YAML) \
		--out_models_dir $(lda_models_dir) \
		--reports_dir $(reports_dir)

## 6. Prepare visualization data
6-prepare-for-topic-model-viz:
	python src/model.py prepare-for-topic-model-viz \
		--db_loc $(db_loc) \
		--prepared_data_dir  $(lda_prep_data_dir) \
		--tmodel_loc $(lda_models_dir)/topic_model$(N_TOPICS) \
		--viz_data_dir $(lda_model_viz_data_dir)

## 7. Get time and characteristics
7-get-time-chars:
	python src/model.py get-time-chars \
		--viz_data_dir $(lda_model_viz_data_dir) \
		--config_loc $(PARAM_YAML)

## Link experiment data to app directory
link-data-to-app:
	ln -f $(lda_model_viz_data_dir)/* app/data
	ln -f $(PARAM_YAML) app/templates/config.yaml

## sync experiment viz data to s3
sync-viz-to-s3:
ifeq (default,$(PROFILE))
	aws s3 cp $(CONFIG_FILE) $(BUCKET)/$(CONFIG_FILE)
	aws s3 cp $(PARAM_YAML) $(BUCKET)/$(PARAM_YAML)
	aws s3 sync $(lda_model_viz_data_dir) $(BUCKET)/$(lda_model_viz_data_dir)
else
	aws s3 cp $(CONFIG_FILE) $(BUCKET)/$(CONFIG_FILE) --profile $(PROFILE)
	aws s3 cp $(PARAM_YAML) $(BUCKET)/$(PARAM_YAML) --profile $(PROFILE)
	aws s3 sync $(lda_model_viz_data_dir) $(BUCKET)/$(lda_model_viz_data_dir) --profile $(PROFILE)
endif

## sync experiment viz from s3
sync-viz-from-s3:
ifeq (default,$(PROFILE))
	aws s3 sync $(BUCKET)/$(lda_model_viz_data_dir) $(lda_model_viz_data_dir)
else
	aws s3 sync $(BUCKET)/$(lda_model_viz_data_dir) $(lda_model_viz_data_dir) --profile $(PROFILE)
endif

## sync all experiment data to s3
sync-experiment-to-s3:
ifeq (default,$(PROFILE))
	aws s3 cp $(CONFIG_FILE) $(BUCKET)/$(CONFIG_FILE)
	aws s3 cp $(PARAM_YAML) $(BUCKET)/$(PARAM_YAML)
	aws s3 sync $(data_dir) $(BUCKET)/$(data_dir)
	aws s3 sync $(model_dir) $(BUCKET)/$(model_dir)
	aws s3 sync $(viz_dir) $(BUCKET)/$(viz_dir)
	aws s3 sync $(reports_dir) $(BUCKET)/$(reports_dir)
else
	aws s3 cp $(CONFIG_FILE) $(BUCKET)/$(CONFIG_FILE) --profile $(PROFILE)
	aws s3 cp $(PARAM_YAML) $(BUCKET)/$(PARAM_YAML) --profile $(PROFILE)
	aws s3 sync $(data_dir) $(BUCKET)/$(data_dir)  --profile $(PROFILE)
	aws s3 sync $(model_dir) $(BUCKET)/$(model_dir)  --profile $(PROFILE)
	aws s3 sync $(viz_dir) $(BUCKET)/$(viz_dir)  --profile $(PROFILE)
	aws s3 sync $(reports_dir) $(BUCKET)/$(reports_dir)  --profile $(PROFILE)
endif

## sync all experiment data from s3
sync-experiment-from-s3:
ifeq (default,$(PROFILE))
	aws s3 sync $(BUCKET)/$(data_dir) $(data_dir)
	aws s3 sync $(BUCKET)/$(model_dir) $(model_dir)
	aws s3 sync $(BUCKET)/$(viz_dir) $(viz_dir)
	aws s3 sync $(BUCKET)/$(reports_dir) $(reports_dir)
else
	aws s3 sync $(BUCKET)/$(data_dir) $(data_dir)  --profile $(PROFILE)
	aws s3 sync $(BUCKET)/$(model_dir) $(model_dir)  --profile $(PROFILE)
	aws s3 sync $(BUCKET)/$(viz_dir) $(viz_dir)  --profile $(PROFILE)
	aws s3 sync $(BUCKET)/$(reports_dir) $(reports_dir)  --profile $(PROFILE)
endif

## Run app for visualize results in development mode
app-dev:
	export VERSION=$$(python version.py); \
	export FLASK_ENV=development; \
	export APP_DATA_DIR=data; \
	export VERSION=$$(python version.py); \
	cd app; \
	export PYTHONPATH=$${PYTHONPATH}:../src/; \
	flask run

## Run app for visualize results in production mode
app-prod: | $(APP_DATA_FILES)
	export VERSION=$$(python version.py); \
	cd app && APP_DATA_DIR=data gunicorn app:app -b :$(PORT) --timeout 1200

check_clean:
	@echo -n "These folders will be deleted:\n$(data_dir)\n$(model_dir)\n$(viz_dir)\nProceed? [y/N] " && read ans && [ $${ans:-N} = y ]

## Delete all files for the given experiment
clean-experiment: check_clean
	@echo "Manually remove the files."

## Get short description of this experiment
info:
	@echo $(INFO);

## Get descriptions for all experiments
all-info:
	@for CONFIG_FILE in $(wildcard config/*.mk); \
	do \
		source $${CONFIG_FILE}; \
		echo "$${EXPERIMENT_NAME}: $${INFO}"; \
		export EXPERIMENT_NAME=; \
		export INFO=; \
	done;

sync-astro2010-whitepapers-from-s3:
	aws s3 sync "s3://datasquad-low/home/DataSquad/topic-emergence-ADS/Astro 2010 Whitepapers/" data/astro2010_whitepapers

get-inference-from-dir:
	mkdir $(inf_dir); \
	python src/get_paper_topic_distribs.py \
		--lda_model $(lda_models_dir)/topic_model$(N_TOPICS) \
		--dir_of_txts data/astro2010_whitepapers \
		--output_embedding_csv $(inf_dir)/astro2010_topic_distributions.csv

export-keywords:
	python src/exports.py keywords \
		--db_loc $(db_loc) \
		--config_loc $(PARAM_YAML) \
		--kwd_export_loc $(reports_dir)/keywords.csv

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
