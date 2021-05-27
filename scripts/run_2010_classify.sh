#!/bin/sh

# run python to get topic distribs
source venv/bin/activate
echo `which python`
# for paras in 2010 report
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/arb-blacklist-08-03-2020/topic_models/topic_model500 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_11_500.csv"
# for targetted selectins of 2010 report
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/astro2010_txt/ --lda_model models/arb-blacklist-08-03-2020/topic_models/topic_model500 --output_embedding_csv data/example_experiment/astro2010_topic_distributions_selected_decadal2010_viz11.csv"
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/arb-blacklist-08-03-2020/topic_models/topic_model500 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_weighted_viz11.csv"
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/arb-1997to2010_thr_0.075/topic_models/topic_model425 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_12_425_0.075.csv"

# try vs 2020 run (2007-2019) just to see what happens..
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2020-2006to2019/topic_models/topic_model350 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_15_350_rec1.csv"
#echo $cmd; $cmd

#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2020-2007to2019_rec2/topic_models/topic_model365 --output_embedding_csv data/example_experiment/topic_distributions_decadal2020_panel_reports_15_365_rec2.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2020-2007to2019_rec3/topic_models/topic_model350 --output_embedding_csv data/example_experiment/topic_distributions_decadal2020_panel_reports_15_350_rec3.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2020-2007to2019_rec4/topic_models/topic_model350 --output_embedding_csv data/example_experiment/topic_distributions_decadal2020_panel_reports_15_350_rec4.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2020-2007to2019_rec5/topic_models/topic_model350 --output_embedding_csv data/example_experiment/topic_distributions_decadal2020_panel_reports_15_350_rec5.csv"
#echo $cmd; $cmd

#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2020-2007to2019_rec6/topic_models/topic_model350 --output_embedding_csv data/example_experiment/topic_distributions_decadal2020_panel_reports_15_350_rec6.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2020-2007to2019_rec7/topic_models/topic_model350 --output_embedding_csv data/example_experiment/topic_distributions_decadal2020_panel_reports_15_350_rec7.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2020-2007to2019_rec8/topic_models/topic_model350 --output_embedding_csv data/example_experiment/topic_distributions_decadal2020_panel_reports_15_350_rec8.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2020-2007to2019_rec9/topic_models/topic_model365 --output_embedding_csv data/example_experiment/topic_distributions_decadal2020_panel_reports_15_365_rec9.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2020-2007to2019_rec10/topic_models/topic_model350 --output_embedding_csv data/example_experiment/topic_distributions_decadal2020_panel_reports_15_350_rec10.csv"
#echo $cmd; $cmd

#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2010-1998to2010_rec1/topic_models/topic_model350 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_16_350_rec1.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2010-1998to2010_rec2/topic_models/topic_model350 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_16_350_rec2.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2010-1998to2010_rec3/topic_models/topic_model350 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_16_350_rec3.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2010-1998to2010_rec4/topic_models/topic_model365 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_16_365_rec4.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2010-1998to2010_rec5/topic_models/topic_model350 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_16_350_rec5.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2010-1998to2010_rec6/topic_models/topic_model350 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_16_350_rec6.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2010-1998to2010_rec7/topic_models/topic_model350 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_16_350_rec7.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2010-1998to2010_rec8/topic_models/topic_model350 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_16_350_rec8.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2010-1998to2010_rec9/topic_models/topic_model365 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_16_365_rec9.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2010-1998to2010_rec10/topic_models/topic_model350 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_16_350_rec10.csv"
#echo $cmd; $cmd

#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2010-1998to2010_ao_lrg_rec1/topic_models/topic_model125 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_19_125_rec1.csv"
echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2010-1998to2010_ao_lrg_rec2/topic_models/topic_model125 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_19_125_rec2.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2010-1998to2010_ao_lrg_rec3/topic_models/topic_model125 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_19_125_rec3.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2010-1998to2010_ao_lrg_rec4/topic_models/topic_model125 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_19_125_rec4.csv"
#echo $cmd; $cmd
#cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2010-1998to2010_ao_lrg_rec5/topic_models/topic_model125 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_19_125_rec5.csv"
#echo $cmd; $cmd

cmd="python src/get_paper_topic_distribs.py --dir_of_txts data/decadal_panel_reports/ --lda_model models/astro2020-2007to2019_ao/topic_models/topic_model150 --output_embedding_csv data/example_experiment/topic_distributions_decadal2010_panel_reports_20_150.csv"
echo $cmd; $cmd
