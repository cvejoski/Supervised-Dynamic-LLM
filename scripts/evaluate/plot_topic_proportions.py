import csv
import glob
import logging
import math
import os
import pathlib
import pickle
import sys
from collections import defaultdict
import pandas as pd
import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from wordcloud import WordCloud

sns.set(color_codes=True)
sns.set_palette("colorblind")

matplotlib.rcParams['text.usetex'] = True

params = {
    'legend.fontsize': 'x-large',
    'figure.autolayout': True,
    'figure.figsize': (25, 5),
    'axes.labelsize': '26',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'x-large',
    'pdf.fonttype': 3,
    'ps.fonttype': 3,
    'ytick.labelsize': 'x-large'
}
plt.rcParams.update(params)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
min_dates = {
    'wallstreetbets': '2019-12-31 23:18:00',
    'politics': '2019-01-01 06:50:55',
    'the_donald': '2017-12-31 23:00:14',
    'askscience': '2017-12-31 23:36:04'
}


@click.command()
@click.option('--models-root-dir', type=click.Path(exists=True), required=True, help='Path to the trained model.')
@click.option('--out-dir', type=click.Path(exists=False), required=True, help='Path to a location for storing the plots.')
@click.option('--data-dir', type=click.Path(exists=True), required=True, help='Path to the data directory.')
@click.option('--filter-models', type=click.File('r'), default=None, required=False, help='Path csv file containing filtered models.')
def main(models_root_dir: click.Path, out_dir: click.Path, data_dir: click.Path, filter_models: click.File):
    important_models = defaultdict(list)
    if filter_models is not None:
        reader = csv.reader(filter_models)
        next(reader)
        for row in reader:
            important_models[row[0]].append(row[-2])
    dataset = data_dir.split(os.path.sep)[-4]
    dataset_key = data_dir.split(os.path.sep)[-4] + '-' + data_dir.split(os.path.sep)[-3]
    filter_models = important_models[dataset_key]
    for model_path in tqdm(sorted(glob.glob(os.path.join(models_root_dir, '**', 'best_model.p'), recursive=True), key=os.path.getmtime)):
        model_path_splitted = model_path.split(os.sep)
        model_name = model_path_splitted[-3]
        if (filter_models and not model_name in filter_models):
            continue
        model_path = os.path.split(model_path)[0]

        beta_path = pathlib.Path(model_path) / 'beta.pkl'
        theta_path = pathlib.Path(model_path) / 'theta.pkl'
        vocab_path = pathlib.Path(data_dir) / 'vocabulary.pkl'

        if not beta_path.exists():
            continue
        beta = pickle.load(open(beta_path, 'rb'))
        theta = pickle.load(open(theta_path, 'rb'))
        num_time_steps = theta[0].shape[0]
        vocabulary = pickle.load(open(vocab_path, 'rb'))['vocab']

        num_words = 50
        num_topics = beta.shape[0]

        fig, axes = plt.subplots(nrows=math.ceil(num_topics / 3),
                                 ncols=6,
                                 facecolor='w',
                                 edgecolor='k',
                                 figsize=(40, math.ceil(num_topics / 3) * 5),
                                 dpi=200,
                                 sharex=False)
        topics_ordered_ix = np.argsort(theta[0].max(0))[::-1]
        axes = axes.flatten()
        T = num_time_steps
        min_date = pd.to_datetime(min_dates[dataset])
        for i, k in enumerate(topics_ordered_ix):
            upper_ = theta[0][:, k] + 1.96 * theta[1][:, k] / np.sqrt(theta[2])
            lower_ = theta[0][:, k] - 1.96 * theta[1][:, k] / np.sqrt(theta[2])
            axes[i * 2].plot(range(T), theta[0][:, k], lw=2, linestyle='-')
            axes[i * 2].fill_between(range(T), lower_, upper_, alpha=0.2)

            tick_label = [(min_date + np.timedelta64(int(t), 'W')).strftime("%b %Y") for t in axes[i * 2].get_xticks()]
            axes[i * 2].set_xticklabels(tick_label, rotation=25)

            if beta.ndim == 2:
                important_words = beta.argsort(axis=-1)
                tokens = important_words[k, -num_words:][::-1]
                words = [vocabulary[token] for token in tokens]
                words_importance = beta[k, tokens]
                word_freq = dict(zip(words, words_importance))
                wordcloud = WordCloud(min_font_size=12, max_font_size=80, max_words=40, background_color="white").generate_from_frequencies(word_freq)

                axes[i * 2 + 1].imshow(wordcloud, interpolation="bilinear")
                axes[i * 2 + 1].axis("off")
        for i in range(num_topics * 2, len(axes)):
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
            axes[i].grid(False)
            axes[i].set_facecolor('white')

        axes[0].get_shared_x_axes().join(*axes[::2])
        # fig.suptitle(r"Topic Evolution Dataset: \textbf{" + dataset + r"} Model: \textbf{" + model_name + "} (mean and 2 std deviations)",
        #              y=1.,
        #              fontsize=32)

        out_path = pathlib.Path(out_dir) / "figures" / "topic_proportions" / "all"
        out_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path / f"topic_evolution_{dataset}_{model_name}_{num_topics}.pdf", bbox_inches='tight')

        time_steps = [(min_date + np.timedelta64(int(t), 'W')).strftime("%W %b %Y") for t in range(num_time_steps)]

        export = {'theta_m': theta[0], 'theta_std': theta[1], 'theta_n_docs': theta[2], 'time': time_steps, 'beta': beta, 'vocabulary': vocabulary}

        pickle.dump(export, open(out_path / f"topic_evolution_{dataset}_{model_name}_{num_topics}.pkl", 'wb'))


if __name__ == '__main__':
    main()
