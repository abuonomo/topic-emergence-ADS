BUCKET=datasquad-low/home/DataSquad/topic-emergence-ADS/
PROFILE=moderate
RECIPES=python recipes.py
IN_ADS_TOKEN=
export LOGLEVEL=INFO
export ADS_TOKEN=$(IN_ADS_TOKEN)

# Set parameters depending on whether running test or full data
BATCH_SIZE=1000
N_PROCESS=1
MIN_THRESH=100
MAX_THRESH=1
NO_BELOW=300
NO_ABOVE=0.25
RAW_DIR=data/raw
JOURNAL_LIMIT=--no_only_nature_and_sci
TOPIC_RANGE_FILE=config/topic_range.json
CONFIG_FILE=config/example_small.mk
TIMESTAMP=$$(date +%Y-%m-%d_%H:%M:%S)
include $(CONFIG_FILE) # This file may overwrite some defaults variables above

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
requirements:
	pip install -r requirements.txt && python -m spacy download en_core_web_sm
	@echo "You need to install the rust compiler on your system to use contextualized embedding topic modeling."
	@echo "If you are on a mac see here: https://sourabhbajaj.com/mac-setup/Rust/"

## Install the requirements for the app
requirements-app:
	pip install -r app/requirements.txt

RAW_FILES=$(shell find $(RAW_DIR) -type f -name '*')
RECORDS_LOC=$(DATA_DIR)/kwds.jsonl
## Join all years and and use rake to extract keywords.
join-and-clean: $(RECORDS_LOC)
$(RECORDS_LOC): $(RAW_FILES)
	mkdir -p $(DATA_DIR); \
	mkdir -p $(MODEL_DIR); \
	mkdir -p $(VIZ_DIR); \
	python src/join_and_clean.py \
		$(RAW_DIR) \
		$(RECORDS_LOC) \
		--limit $(LIMIT) \
		$(JOURNAL_LIMIT) \
		$(YEAR_LIMIT)

ALL_KWDS_LOC=$(DATA_DIR)/all_keywords.jsonl
YEAR_COUNT_LOC=$(DATA_DIR)/year_counts.csv
KWD_TOKENS_LOC=$(DATA_DIR)/kwd_tokens.jsonl
## Get dataframe of keyword frequencies over the years
docs-to-keywords-df: $(ALL_KWDS_LOC) $(YEAR_COUNT_LOC)
$(ALL_KWDS_LOC) $(YEAR_COUNT_LOC): $(RECORDS_LOC)
	python src/extract_keywords.py main \
		--infile $(RECORDS_LOC) \
		--out_years $(YEAR_COUNT_LOC) \
		--out_tokens $(KWD_TOKENS_LOC) \
		--strategy $(STRATEGY) \
		--n_process $(N_PROCESS) \
		--batch_size $(BATCH_SIZE)


filter-kwds: $(KWD_TOKENS_LOC)
	python src/extract_keywords.py filter-kwds \
		--infile $(RECORDS_LOC) \
		--in_kwd_lists $(KWD_TOKENS_LOC) \
		--outfile $(ALL_KWDS_LOC) \
		--min_thresh $(MIN_THRESH) \
		--max_thresh $(MAX_THRESH) \

TOKENIZED_CORPUS_LOC=$(DATA_DIR)/tokens.jsonl
## Tokenize the corpus using spacy and textacy
tokenize:
	python src/extract_keywords.py tokenize \
		--infile $(RECORDS_LOC) \
		--outfile $(TOKENIZED_CORPUS_LOC) \
		--n_process $(N_PROCESS) \
		--batch_size $(BATCH_SIZE)

OUT_AFFIL=$(DATA_DIR)/nasa_affiliation.csv
## Get overall nasa affiliation
affil: $(OUT_AFFIL)
$(OUT_AFFIL): $(RECORDS_LOC) src/get_overall_nasa_affil.py
	python src/get_overall_nasa_affil.py $(RECORDS_LOC) $(OUT_AFFIL)

bootstrap: $(RECORDS_LOC)
	python src/bootstrapping.py $(RECORDS_LOC)

FILT_KWDS_LOC=$(DATA_DIR)/all_keywords_threshold_$(FREQ)_$(SCORE)_$(HARD).jsonl
## Filter keywords by total frequency and rake score. Also provide hard limit.
get-filtered-kwds: $(FILT_KWDS_LOC)
$(FILT_KWDS_LOC): $(ALL_KWDS_LOC)
	@echo "See notebook/arb-10-look-at-textacy-kwds.ipynb to see code for selecting filter params."
	python src/extract_keywords.py filter-kwds \
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
$(TS_FEATURES_LOC): $(NORM_KWDS_LOC) $(OUT_AFFIL)
	$(RECIPES) slope-complexity \
		--norm_loc $(NORM_KWDS_LOC) \
		--affil_loc $(OUT_AFFIL) \
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

APP_DATA_DIR=$$(pwd)/app/data
APP_DATA_FILES=$(shell find $(APP_DATA_DIR) -type f -name '*')
## link data files to app's data dir
link-data-to-app:
	ln -f $(YEAR_COUNT_LOC) app/data
	ln -f $(FILT_KWDS_LOC) app/data/all_keywords_threshold.jsonl
	ln -f $(MANIF_POINTS_LOC) app/data
	ln -f $(KM_MODEL_LOC) app/data
	ln -f $(TS_FEATURES_LOC) app/data
	ln -f $(TMODEL_VIZ_GEN_LOC) app/static/html/topic_model_viz.html
	ln -f $(TOPIC_TO_BIBCODES_LOC) app/data/topic_distribs_to_bibcodes.csv
	ln -f $(TOPIC_TO_YEARS_LOC) app/data/topic_years.csv


## Run app for visualize results
app-dev: | $(APP_DATA_FILES)
	export APP_DATA_DIR=data && cd app && flask run FLASK_ENV=development

app-prod: | $(APP_DATA_FILES)
	cd app && APP_DATA_DIR=data gunicorn app:app -b :5000 --timeout 1200
#========= Topic Modeling =========#

DOC_FEAT_MAT_LOC=$(DATA_DIR)/doc_feature_matrix.mtx
MULT_LAB_BIN_LOC=$(MODEL_DIR)/mlb.jbl
MAP_LOC=$(MODEL_DIR)/mat_doc_mapping.csv
DCT_LOC=$(MODEL_DIR)/gensim_dct.mm
CORP_LOC=$(MODEL_DIR)/gensim_corpus.mm
## Create document term matrix
prepare-features: $(DOC_FEAT_MAT_LOC) $(MULT_LAB_BIN_LOC) $(MAP_LOC)
$(DOC_FEAT_MAT_LOC) $(MULT_LAB_BIN_LOC) $(MAP_LOC): $(NORM_KWDS_LOC)
	python src/topic_modeling.py prepare-features \
		--norm_loc $(NORM_KWDS_LOC) \
		--mat_loc $(DOC_FEAT_MAT_LOC) \
		--mlb_loc $(MULT_LAB_BIN_LOC) \
		--map_loc $(MAP_LOC) \
		--dct_loc $(DCT_LOC) \
		--corp_loc $(CORP_LOC)

TOKENS_LOC=$(MODEL_DIR)/gensim_tokens.jsonl
## Prepare corpus, dictionary, and matrix id to doc id mapping for gensim topic modeling
prepare-gensim-features: $(CORP_TOK_LOC) $(DCT_TOK_LOC) $(MAP_LOC)
$(CORP_TOK_LOC) $(DCT_TOK_LOC) $(MAP_LOC): $(RECORDS_LOC)
	python src/topic_modeling.py prepare-gensim-features \
		--docs_loc $(RECORDS_LOC) \
		--dct_loc $(DCT_TOK_LOC) \
		--corp_loc $(CORP_TOK_LOC) \
		--map_loc $(MAP_LOC) \
		--token_loc $(TOKENS_LOC) \
		--no_below $(NO_BELOW) \
		--no_above $(NO_ABOVE)

COH_PLT_LOC=$(VIZ_DIR)/coherence.png
TMODEL_DIR=$(MODEL_DIR)/topic_models
ALG='lda'
## Create test topic models of varying sizes
run-topic-models: $(COH_PLT_LOC)
$(COH_PLT_LOC): $(DOC_FEAT_MAT_LOC) $(MULT_LAB_BIN_LOC) $(MAP_LOC)
	mkdir -p $(TMODEL_DIR); \
	python src/topic_modeling.py run-topic-models \
		--plot_loc $(COH_PLT_LOC) \
		--mat_loc $(DOC_FEAT_MAT_LOC) \
		--mlb_loc $(MULT_LAB_BIN_LOC) \
		--map_loc $(MAP_LOC) \
		--tmodels_dir $(TMODEL_DIR) \
		--alg $(ALG)

COHERENCE_LOC=$(VIZ_DIR)/coherence$(TIMESTAMP).csv
DCT_TOK_LOC=$(MODEL_DIR)/gensim_tok_dct.mm
CORP_TOK_LOC=$(MODEL_DIR)/gensim_tok_corpus.mm
## Make topic models using gensim's LdaMulticore
run-gensim-lda-mult: $(COH_PLT_LOC)
$(COH_PLT_LOC): $(DCT_LOC) $(CORP_LOC) $(MAP_LOC)
	mkdir -p $(TMODEL_DIR); \
	python src/topic_modeling.py run-gensim-lda-mult \
		--plot_loc $(COH_PLT_LOC) \
		--topic_range_loc $(TOPIC_RANGE_FILE) \
		--tmodels_dir $(TMODEL_DIR) \
		--coherence_loc $(COHERENCE_LOC) \
		--dct_loc $(DCT_LOC) \
		--corp_loc $(CORP_LOC) \
		--tokens_loc $(KWD_TOKENS_LOC)

## Get coherences for gensim topic models
get-gensim-coherences: $(COH_PLT_LOC)
	python src/topic_modeling.py get-gensim-coherences \
		--plot_loc $(COH_PLT_LOC) \
		--corp_loc $(CORP_LOC) \
		--tmodels_dir $(TMODEL_DIR)

TMODELS=$(shell find $(TMODEL_DIR) -type f -name '*')
N_TOPICS=50
TMODEL_VIZ_LOC=$(VIZ_DIR)/topic_model_viz$(N_TOPICS).html
TOPIC_COHS_LOC=$(VIZ_DIR)/topic_coherences$(N_TOPICS).csv
# Above line collects all files in dir for command prerequisite
## Visualize topic models with pyLDAviz
visualize-topic-models: $(TMODEL_VIZ_LOC)
$(TMODEL_VIZ_LOC): $(TMODELS)
	python src/topic_modeling.py visualize-topic-models \
		--infile $(RECORDS_LOC) \
		--tmodel_dir $(TMODEL_DIR) \
		--n $(N_TOPICS) \
		--mlb_loc $(MULT_LAB_BIN_LOC) \
		--map_loc $(MAP_LOC) \
		--tmodel_viz_loc $(TMODEL_VIZ_LOC) \
		--topic_cohs_loc $(TOPIC_COHS_LOC)

TMODEL_VIZ_GEN_LOC=$(VIZ_DIR)/gensim_topic_model_viz$(N_TOPICS).html
## Visualize gensim topic models with pyLDAvis
#visualize-gensim-topic-models: $(TMODEL_VIZ_GEN_LOC)
#$(TMODEL_VIZ_GEN_LOC): $(TMODELS) $(MAP_LOC)
visualize-gensim-topic-models:
	python src/topic_modeling.py visualize-gensim-topic-models \
		--infile $(RECORDS_LOC) \
		--tmodel_dir $(TMODEL_DIR) \
		--n $(N_TOPICS) \
		--in_corpus $(CORP_LOC) \
		--dct_loc $(DCT_LOC) \
		--map_loc $(MAP_LOC) \
		--tmodel_viz_loc $(TMODEL_VIZ_GEN_LOC) \
		--topic_to_bibcodes_loc $(TOPIC_TO_BIBCODES_LOC)

TOPIC_TO_BIBCODES_LOC=$(VIZ_DIR)/topic_distribs_to_bibcodes$(N_TOPICS).hdf5
## Explore topic models and how they connect to original dataset
explore-topic-models: $(TOPIC_TO_BIBCODES_LOC)
$(TOPIC_TO_BIBCODES_LOC):  $(TMODELS)
	python src/topic_modeling.py explore-topic-models \
		--infile $(RECORDS_LOC) \
		--tmodel_dir $(TMODEL_DIR) \
		--n $(N_TOPICS) \
		--mlb_loc $(MULT_LAB_BIN_LOC) \
		--map_loc $(MAP_LOC) \
		--topic_to_bibcodes_loc $(TOPIC_TO_BIBCODES_LOC)

TOPIC_TO_YEARS_LOC=$(VIZ_DIR)/topic_years$(N_TOPICS).jsonl
## Get year time series for topics
get-topic-years: $(TOPIC_TO_YEARS_LOC)
$(TOPIC_TO_YEARS_LOC): $(TOPIC_TO_BIBCODES_LOC) $(RECORDS_LOC)
	python src/topic_modeling.py get-topic-years \
		--records_loc $(RECORDS_LOC) \
		--in_bib $(TOPIC_TO_BIBCODES_LOC) \
		--topic_cohs_loc $(TOPIC_COHS_LOC) \
		--map_loc $(MAP_LOC) \
		--out_years $(TOPIC_TO_YEARS_LOC)

NORM_TOPICS_LOC=$(VIZ_DIR)/topic_years_norm$(N_TOPICS).jsonl
## Normalize topic frequencies by year totals and percent of baselines.
normalize-topic-freqs: $(TOPIC_TO_YEARS_LOC)
$(NORM_TOPICS_LOC): $(TOPIC_TO_YEARS_LOC) $(YEAR_COUNT_LOC)
	$(RECIPES) normalize-keyword-freqs \
		--kwds_loc $(TOPIC_TO_YEARS_LOC) \
		--in_years $(YEAR_COUNT_LOC) \
		--out_norm $(NORM_TOPICS_LOC)

TOPIC_TS_FEATURES_LOC=$(DATA_DIR)/topic_time_series_measures$(N_TOPICS).csv
## Get various measures for topics time series
topics-time-series-measures: $(TOPIC_TS_FEATURES_LOC)
$(TOPIC_TS_FEATURES_LOC): $(NORM_TOPICS_LOC) $(OUT_AFFIL)
	$(RECIPES) slope-complexity \
		--norm_loc $(NORM_TOPICS_LOC) \
		--affil_loc $(OUT_AFFIL) \
		--out_df $(TOPIC_TS_FEATURES_LOC)

TOPIC_DTW_DISTS_LOC=$(DATA_DIR)/topic_dynamic_time_warp_distances.csv
## Compute pairwise dynamic time warp between topics
topic-dtw: $(TOPIC_DTW_DISTS_LOC)
$(TOPIC_DTW_DISTS_LOC): $(NORM_TOPICS_LOC)
	$(RECIPES) dtw \
		--norm_loc $(NORM_TOPICS_LOC) \
		--dtw_loc $(TOPIC_DTW_DISTS_LOC)

ELBOW_PLT_LOC=$(VIZ_DIR)/topic_elbow.png
## Try various numbers of clusters for kmeans topic time series clustering
topic-cluster-tests: $(ELBOW_PLT_LOC)
$(ELBOW_PLT_LOC): $(TOPIC_DTW_DISTS_LOC)
	$(RECIPES) cluster-tests \
		--dtw_loc $(TOPIC_DTW_DISTS_LOC) \
		--out_elbow_plot $(ELBOW_PLT_LOC)

TOPIC_KM_MODEL_LOC=$(MODEL_DIR)/topic_kmeans.jbl
TOPIC_MANIF_PLT_LOC=$(VIZ_DIR)/topic_manifold.png
TOPIC_MANIF_POINTS_LOC=$(MODEL_DIR)/topic_dtw_manifold_proj.jbl
## Cluster topics by dynamic time warp values and plot in tensorboard.
topic-dtw-viz: $(TOPIC_MANIF_PLT_LOC) $(TOPIC_MANIF_POINTS_LOC) $(TOPIC_KM_MODEL_LOC)
$(TOPIC_MANIF_PLT_LOC) $(TOPIC_MANIF_POINTS_LOC) $(TOPIC_KM_MODEL_LOC): $(NORM_TOPICS_LOC) $(TOPIC_DTW_DISTS_LOC)
	$(RECIPES) dtw-viz \
		--norm_loc $(NORM_TOPICS_LOC) \
		--dtw_loc $(TOPIC_DTW_DISTS_LOC) \
		--kmeans_loc $(TOPIC_KM_MODEL_LOC) \
		--out_man_plot $(TOPIC_MANIF_PLT_LOC) \
		--out_man_points $(TOPIC_MANIF_POINTS_LOC)

link-topic-data-to-app:
	ln -f $(YEAR_COUNT_LOC) app/data/year_counts.csv
	ln -f $(FILT_KWDS_LOC) app/data/kwd_all_keywords_threshold.jsonl
	ln -f $(TS_FEATURES_LOC) app/data/kwd_slope_complex.csv
	ln -f $(TOPIC_TO_YEARS_LOC) app/data/all_keywords_threshold.jsonl
	ln -f $(TOPIC_MANIF_POINTS_LOC) app/data/dtw_manifold_proj.jbl
	ln -f $(TOPIC_KM_MODEL_LOC) app/data/kmeans.jbl
	ln -f $(KM_MODEL_LOC) app/data/kwd_kmeans.jbl
	ln -f $(TOPIC_TS_FEATURES_LOC) app/data/slope_complex.csv
	ln -f $(TMODEL_VIZ_GEN_LOC) app/static/html/topic_model_viz.html
	ln -f $(TOPIC_TO_YEARS_LOC) app/data/topic_years.jsonl
	ln -f $(TOPIC_TO_BIBCODES_LOC) app/data/topic_distribs_to_bibcodes.hdf5


DOC_TXTS=$(DATA_DIR)/documents.txt
## Prepare data for neural LDA
prepare-for-neural-lda: $(DOC_TXTS)
$(DOC_TXTS): $(RECORDS_LOC)
	python src/topic_modeling.py prepare-for-neural-lda \
		--infile $(RECORDS_LOC) \
		--outfile $(DOC_TXTS)

NEURAL_LDA_MODEL_DIR=$(MODEL_DIR)/neural_lda
N_EPOCHS=10
## Run contextual neural lda with bert embeddings
run-neural-lda: $(NEURAL_LDA_MODEL_LOC)
	python src/topic_modeling.py run-neural-lda \
		--in_docs $(DOC_TXTS) \
		--dct_loc $(DCT_LOC) \
		--corp_loc $(CORP_LOC) \
		--lda_model_dir $(NEURAL_LDA_MODEL_DIR) \
		--n_topics $(N_TOPICS) \
		--num_epochs $(N_EPOCHS)

#$(NEURAL_LDA_MODEL_LOC): $(DOC_TXTS)

#========= Docker =========#

PIPELINE_IMAGE_NAME=keyword-emergence-pipeline
## Build docker image for service, automatically labeling image with link to most recent commit
build:
	export COMMIT=$$(git log -1 --format=%H); \
	export REPO_URL=$$(git remote get-url $(GIT_REMOTE)); \
	export REPO_DIR=$$(dirname $$REPO_URL); \
	export BASE_NAME=$$(basename $$REPO_URL .git); \
	export GIT_LOC=$$REPO_DIR/$$BASE_NAME/tree/$$COMMIT; \
	export VERSION=$$(python version.py); \
	echo $$GIT_LOC; \
	docker build -t $(PIPELINE_IMAGE_NAME):$$VERSION \
		--build-arg GIT_URL=$$GIT_LOC \
		--build-arg VERSION=$$VERSION .; \
	docker tag $(PIPELINE_IMAGE_NAME):$$VERSION $(PIPELINE_IMAGE_NAME):latest; \
	docker tag $(PIPELINE_IMAGE_NAME):$$VERSION storage.analytics.nasa.gov/datasquad/$(PIPELINE_IMAGE_NAME):$$VERSION; \
	docker tag $(PIPELINE_IMAGE_NAME):$$VERSION storage.analytics.nasa.gov/datasquad/$(PIPELINE_IMAGE_NAME):latest; \

## Push the docker image to storage.analytics.nasa.gov
push:
	export VERSION=$$(python version.py); \
	docker push storage.analytics.nasa.gov/datasquad/$(PIPELINE_IMAGE_NAME):$$VERSION; \
	docker push storage.analytics.nasa.gov/datasquad/$(PIPELINE_IMAGE_NAME):latest

## Push docker image to storage.analytics.nasa.gov as stable version
push-stable:
	export VERSION=$$(python version.py); \
	docker tag $(PIPELINE_IMAGE_NAME):$$VERSION storage.analytics.nasa.gov/datasquad/$(PIPELINE_IMAGE_NAME):stable; \
	docker push storage.analytics.nasa.gov/datasquad/$(PIPELINE_IMAGE_NAME):latest

## Save the docker image locally
save:
	docker save $(PIPELINE_IMAGE_NAME):latest | gzip > $(PIPELINE_IMAGE_NAME)_latest.tar.gz

# Server name here references an entry in the ~/.ssh/config file
SERVER_NAME=compute-ml
## Upload docker image to server
upload:
	scp $(PIPELINE_IMAGE_NAME)_latest.tar.gz \
		compute-ml:/home/ubuntu/projects/keyword-emergence/$(PIPELINE_IMAGE_NAME)_latest.tar.gz

## Load docker image on remote server from local file which was uploaded
load:
	ssh compute-ml "cd /home/ubuntu/projects/keyword-emergence/ && docker load < $(PIPELINE_IMAGE_NAME)_latest.tar.gz"

## Run docker image remotely
run-remote:
	ssh compute-ml << HERE
		cd /home/ubuntu/projects/keyword-emergence/
		docker run -d --shm-size 4g \
			--env NB_WORKERS=12 \
			-v $(pwd)/config:/home/config \
			-v $(pwd)/data:/home/data \
			-v $(pwd)/models:/home/models \
			-v $(pwd)/reports:/home/reports \
			keyword-emergence-pipeline:latest \
			dtw-viz CONFIG_FILE=config/full_new_data.mk"
	HERE

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
docker-run-app: | $(APP_DATA_FILES)
	export VERSION=$$(python version.py); \
	cd app; \
	docker run -it \
		-p 5001:5000 \
		-v $$(pwd)/static/html:/home/static/html/ \
		-v $$(pwd)/data:/home/data/ \
		$(IMAGE_NAME):$$VERSION

#===== S3 Bucket Syncing =====#

## sync data from s3 bucket
sync-from-s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)models/$(EXP_NAME) models/$(EXP_NAME)
	aws s3 sync s3://$(BUCKET)data/$(EXP_NAME) data/$(EXP_NAME)
	aws s3 sync s3://$(BUCKET)reports/viz/$(EXP_NAME) reports/viz/$(EXP_NAME)
else
	aws s3 sync s3://$(BUCKET)models/$(EXP_NAME) models/$(EXP_NAME) --profile $(PROFILE)
	aws s3 sync s3://$(BUCKET)data/$(EXP_NAME) data/$(EXP_NAME) --profile $(PROFILE)
	aws s3 sync s3://$(BUCKET)reports/viz/$(EXP_NAME) reports/viz/$(EXP_NAME) --profile $(PROFILE)
endif

## sync data and models to s3 bucket
sync-to-s3:
ifeq (default,$(PROFILE))
	aws s3 sync models/$(EXP_NAME) s3://$(BUCKET)models/$(EXP_NAME)
	aws s3 sync data/$(EXP_NAME) s3://$(BUCKET)data/$(EXP_NAME)
	aws s3 sync reports/viz/$(EXP_NAME) s3://$(BUCKET)reports/viz/$(EXP_NAME)
else
	aws s3 sync models/$(EXP_NAME) s3://$(BUCKET)models/$(EXP_NAME) --profile $(PROFILE)
	aws s3 sync data/$(EXP_NAME) s3://$(BUCKET)data/$(EXP_NAME) --profile $(PROFILE)
	aws s3 sync reports/viz/$(EXP_NAME) s3://$(BUCKET)reports/viz/$(EXP_NAME) --profile $(PROFILE)
endif

## sync app data and models from s3 bucket. WARNING: This will overwrite existing files for current experiment.
sync-app-data-from-s3:
	for file in $(YEAR_COUNT_LOC) $(FILT_KWDS_LOC) $(MANIF_POINTS_LOC) $(KM_MODEL_LOC) $(TS_FEATURES_LOC); do \
		aws s3 cp s3://$(BUCKET)$$file $$file --profile $(PROFILE); \
	done

## sync raw ADS files from s3 bucket
sync-raw-data-from-s3:
	aws s3 sync s3://datasquad-low/data/ADS/2020_03_15 data/raw

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
