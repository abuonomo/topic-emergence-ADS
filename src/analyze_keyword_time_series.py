import argparse
import logging
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tensorboardX import SummaryWriter
from tensorboardX.utils import figure_to_image
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def plot_time(v, show=False):
    years = v.index.str.replace("_sum", "").astype(int)
    vals = v.values
    plt.style.use('dark_background')
    plt.figure(figsize=(5, 5))  # Must be square for tensorboard
    plt.plot(years, vals, color="white")
    plt.title(v.name)
    fig = plt.gcf()
    if show:
        plt.show()
    return fig


def get_fig_images(df):
    images = []
    LOG.info('Getting images of time series.')
    for kwd in tqdm(df.index):
        fig = plot_time(df.loc[kwd])
        fig_img = figure_to_image(fig)
        images.append(fig_img)
    image_arr = np.stack(images)
    return image_arr


def main(in_dir: Path):
    dtw_loc = in_dir / "dynamic_time_warp_distances.csv"
    LOG.info(f"Reading dynamic time warps from {dtw_loc}.")
    dtw_df = pd.read_csv(in_dir / 'dynamic_time_warp_distances.csv', index_col=0)
    norm_loc = in_dir / "lim_normed_keyword_stems.jsonl"
    LOG.info(f"Reading normalized keywords years from {norm_loc}.")
    normed_kwd_years = pd.read_csv(norm_loc, index_col=0)

    writer = SummaryWriter()
    images = get_fig_images(normed_kwd_years)
    LOG.info('Writing to tensorboard.')
    writer.add_embedding(dtw_df.values, metadata=dtw_df.index, label_img=images)
    writer.close()
    LOG.info('Done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('i', help='input txt file', type=Path)
    args = parser.parse_args()
    main(args.i)
