import argparse
import logging
import json

from tqdm import tqdm
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def main(infile, outfile, year_min=0):
    nasa_affils = []
    not_counted = 0
    counted = 0
    total = 0
    with open(infile, "r") as f0:
        for cnt, line in tqdm(enumerate(f0)):
            total += 1
            d = json.loads(line)
            if not np.isnan(int(d["year"])):
                if ("astronomy" in d["database"]) and (int(d["year"]) >= year_min):
                    nasa_affils.append(d["nasa_afil"])
                    counted += 1
                else:
                    not_counted += 1
                    continue
            else:
                not_counted += 1
                continue
    LOG.info(f"Did not count {not_counted}/{total} papers.")
    nasa_affils = np.array(nasa_affils)
    uns = np.unique(nasa_affils, return_counts=True)
    overall_affil = uns[1][1] / nasa_affils.shape[0]
    LOG.info(f"Overall affiliation: {overall_affil}")
    out_df = pd.DataFrame({"nasa_affiliation": [overall_affil]})
    LOG.info(f"Writing to {outfile}")
    out_df.to_csv(outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("i", help="input records file")
    parser.add_argument("o", help="output nasa affiliation csv")
    parser.add_argument(
        "--year_min",
        help="minimum year for affiliation calculation",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    main(args.i, args.o, args.year_min)
