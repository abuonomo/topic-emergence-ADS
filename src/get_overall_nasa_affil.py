import argparse
import logging
import json

from tqdm import tqdm
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def main(infile, outfile):
    nasa_affils = []
    with open(infile, 'r') as f0:
        for cnt, line in tqdm(enumerate(f0)):
            d = json.loads(line)
            if 'astronomy' in d['database']:
                nasa_affils.append(d['nasa_afil'])
            else:
                continue
    nasa_affils = np.array(nasa_affils)
    uns = np.unique(nasa_affils, return_counts=True)
    overall_affil = uns[1][1] / nasa_affils.shape[0]
    LOG.info(f'Overall affiliation: {overall_affil}')
    out_df = pd.DataFrame({"nasa_affiliation": [overall_affil]})
    LOG.info(f"Writing to {outfile}")
    out_df.to_csv(outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('i', help='input txt file')
    parser.add_argument('o', help='input txt file')
    args = parser.parse_args()
    main(args.i, args.o)
