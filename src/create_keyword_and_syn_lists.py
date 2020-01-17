import argparse
import logging
import pandas as pd
import spacy
from tqdm import tqdm
from fuzzyset import FuzzySet

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

NLP = spacy.load("en_core_web_sm")


def main(infile, outfile):
    LOG.info(f'Reading keywords from {infile}.')
    rake_kwds = pd.read_json(infile, orient='records', lines=True)
    rake_kwds.columns = ['keyword', 'score']
    LOG.info(f'Writing keywords to {outfile}.')
    with open(outfile, 'w') as f0:
        for kwd in tqdm(rake_kwds['keyword']):
            f0.write(kwd)
            f0.write('\n')
    fs = FuzzySet(rake_kwds['keyword'])
    # for s, k in fs:
    #
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('i', help='input json lines records')
    parser.add_argument('o', help='ouput text file with keywords')
    args = parser.parse_args()
    main(args.i, args.o)
