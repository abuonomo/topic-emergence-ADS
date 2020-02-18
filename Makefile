BUCKET = hq-ocio-ci-bigdata/home/DataSquad/topic-emergence-ADS/
PROFILE = moderate
RECIPES=python recipes.py

# Set parameters depending on whether running test or full data
ifeq ($(MODE), full)
	EXP_NAME=full_02_14_2020
    RECORDS_LOC=data/$(EXP_NAME)/rake_kwds.jsonl
    LIMIT=0
    MIN_THRESH=100
    FREQ=250
    SCORE=1.5
    HARD=10000
else
	EXP_NAME=test_02_14_2020
    RECORDS_LOC=data/$(EXP_NAME)/rake_kwds_small.jsonl
    LIMIT=500
    MIN_THRESH=500
    FREQ=20
    SCORE=1.5
    HARD=10000
    MODE=test
endif

DATA_DIR=data/$(EXP_NAME)
VIZ_DIR=reports/viz/$(EXP_NAME)
MODEL_DIR=models/$(EXP_NAME)

.PHONY: join-and-clean docs-to-keywords-df get-filtered-kwds normalize-keyword-freqs \
		slope-complexity dtw cluster-tests dtw-viz \
		make-topic-models visualize-topic-models app
all: join-and-clean docs-to-keywords-df get-filtered-kwds normalize-keyword-freqs \
	 slope-complexity dtw cluster-tests dtw-viz \
	 make-topic-models visualize-topic-models

$(DATA_DIR) $(MODEL_DIR) $(VIZ_DIR):
	mkdir -p $(DATA_DIR) $(MODEL_DIR) $(VIZ_DIR)

## Install packages to current environment with pip (venv recommended)
install:
	pip install -r requirements.txt

RAW_DIR='data/raw'
RAW_FILES=$(shell find $(RAW_DIR) -type f -name '*')
## Join all years and and use rake to extract keywords.
join-and-clean: $(RECORDS_LOC)
$(RECORDS_LOC): $(RAW_FILES)
	mkdir -p $(DATA_DIR); \
	mkdir -p $(MODEL_DIR); \
	mkdir -p $(VIZ_DIR); \
	python src/join_and_clean.py \
		$(RAW_DIR) \
		$(RECORDS_LOC) \
		--limit $(LIMIT)

ALL_KWDS_LOC=$(DATA_DIR)/all_keywords.jsonl
YEAR_COUNT_LOC=$(DATA_DIR)/year_counts.csv
## Get dataframe of keyword frequencies over the years
docs-to-keywords-df: $(ALL_KWDS_LOC) $(YEAR_COUNT_LOC)
$(ALL_KWDS_LOC) $(YEAR_COUNT_LOC): $(RECORDS_LOC)
	$(RECIPES) docs-to-keywords-df \
		--infile $(RECORDS_LOC) \
		--outfile $(ALL_KWDS_LOC) \
		--out_years $(YEAR_COUNT_LOC)

FILT_KWDS_LOC=$(DATA_DIR)/all_keywords_threshold_$(FREQ)_$(SCORE)_$(HARD).jsonl
## Filter keywords by total frequency and rake score. Also provide hard limit.
get-filtered-kwds: $(FILT_KWDS_LOC)
$(FILT_KWDS_LOC): $(ALL_KWDS_LOC)
	$(RECIPES) get-filtered-kwds \
		--infile $(ALL_KWDS_LOC) \
		--out_loc $(FILT_KWDS_LOC) \
		--threshold $(FREQ) --score_thresh=$(SCORE) --hard_limit $(HARD)

NORM_KWDS_LOC=$(DATA_DIR)/all_keywords_norm_threshold_$(FREQ)_$(SCORE)_$(HARD).jsonl
## Normalize keyword frequencies by year totals and percent of baselines.
normalize-keyword-freqs: $(NORM_KWDS_LOC)
$(NORM_KWDS_LOC): $(FILT_KWDS_LOC) $(YEAR_COUNT_LOC)
	$(RECIPES) normalize-keyword-freqs \
		--kwds_loc $(FILT_KWDS_LOC) \
		--in_years $(YEAR_COUNT_LOC) \
		--out_norm $(NORM_KWDS_LOC)

TS_FEATURES_LOC=$(DATA_DIR)/slope_complex.csv
## Get various measures for keyword time series
slope-complexity: $(TS_FEATURES_LOC)
$(TS_FEATURES_LOC): $(NORM_KWDS_LOC)
	$(RECIPES) slope-complexity \
		--norm_loc $(NORM_KWDS_LOC) \
		--out_df $(TS_FEATURES_LOC)

DTW_DISTS_LOC=$(DATA_DIR)/dynamic_time_warp_distances.csv
## Compute pairwise dynamic time warp between keywords
dtw: $(DTW_DISTS_LOC)
$(DTW_DISTS_LOC): $(NORM_KWDS_LOC)
	$(RECIPES) dtw \
		--norm_loc $(NORM_KWDS_LOC) \
		--dtw_loc $(DTW_DISTS_LOC)

ELBOW_PLT_LOC=$(VIZ_DIR)/elbow.png
## Try various numbers of clusters for kmeans, produce plots
cluster-tests: $(ELBOW_PLT_LOC)
$(ELBOW_PLT_LOC): $(DTW_DISTS_LOC)
	$(RECIPES) cluster-tests \
		--dtw_loc $(DTW_DISTS_LOC) \
		--out_elbow_plot $(ELBOW_PLT_LOC)

KM_MODEL_LOC=$(MODEL_DIR)/kmeans.jbl
MANIF_PLT_LOC=$(VIZ_DIR)/manifold.png
MANIF_POINTS_LOC=$(MODEL_DIR)/dtw_manifold_proj.jbl
## Cluster keywords by dynamic time warp values and plot in tensorboard.
dtw-viz: $(MANIF_PLT_LOC) $(MANIF_POINTS_LOC) $(KM_MODEL_LOC)
$(MANIF_PLT_LOC) $(MANIF_POINTS_LOC) $(KM_MODEL_LOC): $(NORM_KWDS_LOC) $(DTW_DISTS_LOC)
	$(RECIPES) dtw-viz \
		--norm_loc $(NORM_KWDS_LOC) \
		--dtw_loc $(DTW_DISTS_LOC) \
		--kmeans_loc $(KM_MODEL_LOC) \
		--out_man_plot $(MANIF_PLT_LOC) \
		--out_man_points $(MANIF_POINTS_LOC)

APP_DATA_FILES=$(shell find $$(pwd)/app/data/ -type f -name '*')
## link data files to app's data dir
link-data-to-app: $(APP_DATA_FILES)
$(APP_DATA_FILES): $(MANIF_POINTS_LOC) $(NORM_KWDS_LOC) $(TS_FEATURES_LOC) $(YEAR_COUNT_LOC)
	ln -f $(YEAR_COUNT_LOC) app/data
	ln -f $(FILT_KWDS_LOC) app/data
	ln -f $(MANIF_POINTS_LOC) app/data
	ln -f $(KM_MODEL_LOC) app/data
	ln -f $(TS_FEATURES_LOC) app/data

## Run app for visualize results
app: $(APP_DATA_FILES)
	cd app && flask run

#========= Topic Modeling =========#

COH_PLT_LOC=$(VIZ_DIR)/coherence.png
DOC_FEAT_MAT_LOC=$(DATA_DIR)/doc_feature_matrix.mm
MULT_LAB_BIN_LOC=$(MODEL_DIR)/mlb.jbl
MAP_LOC=$(MODEL_DIR)/mat_doc_mapping.csv
TMODEL_DIR=$(MODEL_DIR)/topic_models
## Create document term matrix, topic model, and write to tensorboard
make-topic-models: $(COH_PLT_LOC) $(DOC_FEAT_MAT_LOC) $(MULT_LAB_BIN_LOC) $(MAP_LOC)
$(COH_PLT_LOC) $(DOC_FEAT_MAT_LOC) $(MULT_LAB_BIN_LOC) $(MAP_LOC): $(NORM_KWDS_LOC)
	mkdir -p $(TMODEL_DIR); \
	$(RECIPES) make-topic-models \
		--norm_loc $(NORM_KWDS_LOC) \
		--plot_loc $(COH_PLT_LOC) \
		--mat_loc $(DOC_FEAT_MAT_LOC) \
		--mlb_loc $(MULT_LAB_BIN_LOC) \
		--map_loc $(MAP_LOC) \
		--tmodels_dir $(TMODEL_DIR)

TMODEL_VIZ_LOC=$(VIZ_DIR)/topic_model_viz.html
TMODELS=$(shell find $(TMODEL_DIR) -type f -name '*')
N_TOPICS=7
# Above line collects all files in dir for command prerequisite
## Visualize topic models with pyLDAviz
visualize-topic-models: $(TMODEL_VIZ_LOC)
$(TMODEL_VIZ_LOC): $(TMODELS)
	$(RECIPES) visualize-topic-models \
		--tmodel_dir $(TMODEL_DIR) \
		--n $(N_TOPICS) \
		--mlb_loc $(MULT_LAB_BIN_LOC) \
		--map_loc $(MAP_LOC) \
		--tmodel_viz_loc $(TMODEL_VIZ_LOC)

#========= Docker =========#

IMAGE_NAME=keyword-emergence-visualizer
GIT_REMOTE=origin
## Build flask app docker container
docker-build-app:
	export COMMIT=$$(git log -1 --format=%H); \
	export REPO_URL=$$(git remote get-url $(GIT_REMOTE)); \
	export REPO_DIR=$$(dirname $$REPO_URL); \
	export BASE_NAME=$$(basename $$REPO_URL .git); \
	export GIT_LOC=$$REPO_DIR/$$BASE_NAME/tree/$$COMMIT; \
	export VERSION=$$(python version.py); \
	echo $$GIT_LOC; \
	cd app; \
	docker build -t $(IMAGE_NAME):$$VERSION \
		--build-arg GIT_URL=$$GIT_LOC \
		--build-arg VERSION=$$VERSION .

## Run flask app using gunicorn through docker container
docker-run-app: $(APP_DATA_FILES)
	export VERSION=$$(python version.py); \
	cd app; \
	docker run -it \
		-p 5001:5000 \
		-v $$(pwd)/data:/home/data/ \
		$(IMAGE_NAME):$$VERSION

#===== S3 Bucket Syncing =====#

## sync data from s3 bucket
sync-from-s3:
	aws s3 sync s3://$(BUCKET)models/$(EXP_NAME) models/$(EXP_NAME) --profile $(PROFILE)
	aws s3 sync s3://$(BUCKET)data/$(EXP_NAME) data/$(EXP_NAME) --profile $(PROFILE)
	aws s3 sync s3://$(BUCKET)reports/viz/$(EXP_NAME) reports/viz/$(EXP_NAME) --profile $(PROFILE)

## sync data and models to s3 bucket
sync-to-s3:
	aws s3 sync models/$(EXP_NAME) s3://$(BUCKET)models/$(EXP_NAME) --profile $(PROFILE)
	aws s3 sync data/$(EXP_NAME) s3://$(BUCKET)data/$(EXP_NAME) --profile $(PROFILE)
	aws s3 sync reports/viz/$(EXP_NAME) s3://$(BUCKET)reports/viz/$(EXP_NAME) --profile $(PROFILE)

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