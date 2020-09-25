EXPERIMENT_NAME=arb-blacklist-08-03-2020

# number of papers per year, 0 means no limt
LIMIT=0

# When putting the keywords in the database, this is the commit size
BATCH_SIZE=1000

# downstream parameter
PARAM_YAML=config/arb-blacklist-08-03-2020.yaml

# this is for running the app locally, not the port if deployed elsewhere
PORT=5000

INFO="Gensim corpus without keyword counts, just occurrence. Using updated blacklist as of 08/03/2020"
