#!/bin/bash
#python src/db.py write-ads-to-db --infile scratch/kwds.jsonl --db_loc scratch/ads.sqlite
#python src/db.py get-keywords-from-texts --db_loc scratch/ads.sqlite
#python src/db.py add-missed-locations --db_loc scratch/ads.sqlite  --config_loc config/config.yaml
python src/db.py make-topic-models --db_loc scratch/ads.sqlite --config_loc config/config.yaml --out_models_dir scratch/tmodels
