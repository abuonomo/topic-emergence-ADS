#!/bin/sh

# run python to get topic distribs
source venv/bin/activate
echo `which python`
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/arb-blacklist-08-03-2020/topic_models/topic_model500 --output_embedding_csv data/example_experiment/astro2010_topic_distributions_whitepapers_vis11.csv"
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/summary --lda_model models/arb-blacklist-08-03-2020/topic_models/topic_model500 --output_embedding_csv data/example_experiment/astro2010_topic_distributions_whitepaper_sum_vis11.csv"

#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/astro2010-1998to2010_rec1/topic_models/topic_model350 --output_embedding_csv data/example_experiment/astro2010_topic_distributions_whitepapers_16_350_rec1.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/astro2010-1998to2010_rec2/topic_models/topic_model350 --output_embedding_csv data/example_experiment/astro2010_topic_distributions_whitepapers_16_350_rec2.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/astro2010-1998to2010_rec3/topic_models/topic_model350 --output_embedding_csv data/example_experiment/astro2010_topic_distributions_whitepapers_16_350_rec3.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/astro2010-1998to2010_rec4/topic_models/topic_model365 --output_embedding_csv data/example_experiment/astro2010_topic_distributions_whitepapers_16_365_rec4.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/astro2010-1998to2010_rec5/topic_models/topic_model350 --output_embedding_csv data/example_experiment/astro2010_topic_distributions_whitepapers_16_350_rec5.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/astro2010-1998to2010_rec6/topic_models/topic_model350 --output_embedding_csv data/example_experiment/astro2010_topic_distributions_whitepapers_16_350_rec6.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/astro2010-1998to2010_rec7/topic_models/topic_model350 --output_embedding_csv data/example_experiment/astro2010_topic_distributions_whitepapers_16_350_rec7.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/astro2010-1998to2010_rec8/topic_models/topic_model350 --output_embedding_csv data/example_experiment/astro2010_topic_distributions_whitepapers_16_350_rec8.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/astro2010-1998to2010_rec9/topic_models/topic_model350 --output_embedding_csv data/example_experiment/astro2010_topic_distributions_whitepapers_16_350_rec9.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/astro2010-1998to2010_rec10/topic_models/topic_model350 --output_embedding_csv data/example_experiment/astro2010_topic_distributions_whitepapers_16_350_rec10.csv"
#echo $cmd; $cmd


cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/astro2020-2007to2019_rec2/topic_models/topic_model365 --output_embedding_csv data/example_experiment/astro2020_topic_distributions_2010_whitepapers_15_365_rec2.csv"
echo $cmd; $cmd
cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/astro2020-2007to2019_rec3/topic_models/topic_model350 --output_embedding_csv data/example_experiment/astro2020_topic_distributions_2010_whitepapers_15_350_rec3.csv"
echo $cmd; $cmd
cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/astro2020-2007to2019_rec4/topic_models/topic_model350 --output_embedding_csv data/example_experiment/astro2020_topic_distributions_2010_whitepapers_15_350_rec4.csv"
echo $cmd; $cmd
cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/astro2020-2007to2019_rec5/topic_models/topic_model350 --output_embedding_csv data/example_experiment/astro2020_topic_distributions_2010_whitepapers_15_350_rec5.csv"
echo $cmd; $cmd
cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/astro2020-2007to2019_rec6/topic_models/topic_model350 --output_embedding_csv data/example_experiment/astro2020_topic_distributions_2010_whitepapers_15_350_rec6.csv"
echo $cmd; $cmd
cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/astro2020-2007to2019_rec7/topic_models/topic_model350 --output_embedding_csv data/example_experiment/astro2020_topic_distributions_2010_whitepapers_15_350_rec7.csv"
echo $cmd; $cmd
cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/astro2020-2007to2019_rec8/topic_models/topic_model350 --output_embedding_csv data/example_experiment/astro2020_topic_distributions_2010_whitepapers_15_350_rec8.csv"
echo $cmd; $cmd
cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/astro2020-2007to2019_rec9/topic_models/topic_model365 --output_embedding_csv data/example_experiment/astro2020_topic_distributions_2010_whitepapers_15_365_rec9.csv"
echo $cmd; $cmd
cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_whitepapers/text/ --lda_model models/astro2020-2007to2019_rec10/topic_models/topic_model350 --output_embedding_csv data/example_experiment/astro2020_topic_distributions_2010_whitepapers_15_350_rec10.csv"
echo $cmd; $cmd
