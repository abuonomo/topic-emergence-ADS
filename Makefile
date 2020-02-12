RECIPES=python recipes.py

# Set parameters depending on whether running test or full data
ifeq ($(MODE), full)
	EXP_NAME=kwd_analysis_full_perc2
    RECORDS_LOC=data/rake_kwds.jsonl
    MIN_THRESH=100
    FREQ=250
    SCORE=1.5
    HARD=10000
else
	EXP_NAME=kwd_analysis_perc
    RECORDS_LOC=data/rake_kwds_small.jsonl
    MIN_THRESH=500
    FREQ=20
    SCORE=1.5
    HARD=10000
    MODE=test
endif

DATA_DIR=data/$(EXP_NAME)
VIZ_DIR=reports/viz/$(EXP_NAME)
MODEL_DIR=models/$(EXP_NAME)

## Join all years and and use rake to extract keywords.
#data/rake_kwds.jsonl:
#	python src/join_and_clean.py \
#		data/raw \
#		data/rake_kwds.jsonl


## Get keyword frequencies over time and normalize
#data/kwd_analysis_full: data/rake_kwds.jsonl
#	python src/create_keyword_and_syn_lists.py \
#		data/rake_kwds.jsonl \
#		data/kwd_analysis_full

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

NORM_KWDS_LOC=data/all_keywords_norm_threshold_$(FREQ)_$(SCORE)_$(HARD).jsonl
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

$(VIZ_DIR)/slope_complex_count.html:

#NORM_LOC=$(DATA_DIR)/all_keywords_norm_threshold_{FREQ}_{SCORE}_{HARD}.jsonl
NORM_LOC:

$(DATA_DIR)/all_keywords_norm_threshold_$(FREQ)_$(SCORE)_$(HARD).jsonl:

#DTW = data/kwd_analysis_full/dynamic_time_warp_distances.csv
#NORMED = data/kwd_analysis_full/lim_normed_keyword_stems.jsonl
#$(DTW): $(NORMED)
#	python src/dtw_time_analysis.py data/kwd_analysis_full
### make pairwise dynamic time warp
#warp: $(DTW)

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