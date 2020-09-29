#!/bin/sh

# run python to get topic distribs
source venv/bin/activate
echo `which python`
cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/arb-blacklist-08-03-2020/topic_models/topic_model500 --output_embedding_csv data/example_experiment/astro2010_topic_distributions_whitepapers_vis11.csv"
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/summary --lda_model models/arb-blacklist-08-03-2020/topic_models/topic_model500 --output_embedding_csv data/example_experiment/astro2010_topic_distributions_whitepaper_sum_vis11.csv"
echo $cmd
$cmd
