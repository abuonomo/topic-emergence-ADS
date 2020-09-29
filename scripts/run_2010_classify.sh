#!/bin/sh

# run python to get topic distribs
source venv/bin/activate
echo `which python`
# for paras in 2010 report
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/2010_Decadal_Text/ --lda_model models/arb-blacklist-08-03-2020/topic_models/topic_model500 --output_embedding_csv data/example_experiment/astro2010_topic_distributions_decadal2010_viz11.csv"
# for targetted selectins of 2010 report
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_txt/ --lda_model models/arb-blacklist-08-03-2020/topic_models/topic_model500 --output_embedding_csv data/example_experiment/astro2010_topic_distributions_selected_decadal2010_viz11.csv"
cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/arb-blacklist-08-03-2020/topic_models/topic_model500 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_weighted_viz11.csv"
echo $cmd
$cmd
