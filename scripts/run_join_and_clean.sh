#!/bin/bash
python -m ipdb src/join_and_clean.py \
        data/raw \
        data/arb-astro2020-09-10-2020/kwds.jsonl \
        --config_loc config/arb-astro2020-09-10-2020.yaml
