import click

import pandas as pd
from functools import partial
from multiprocessing import Pool
import os
import tqdm

from pandas.io import pickle
import numpy as np
from scipy.sparse import csr_matrix


def group_by(value: int, period: str, data: pd.DataFrame, bow_transformer):
    text = data.query(f"date_{period} == {value}").text.values
    bows = bow_transformer.transform(text)
    return (value, np.asarray(bows.sum(axis=0)).flatten())


@click.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Path to the downloaded Reddit data.')
@click.option('--num-workers', type=int, default=1, help='Number of workers')
def main(input: click.Path(exists=True),  num_workers: int):
    data_path = os.path.join(input, "data_all_preprocessed.zip")
    data = pd.read_pickle(data_path)
    bow_transformer = pickle.read_pickle(open(os.path.join(input, 'bow_model.pkl'), 'rb'))
    vocab_size = len(bow_transformer.get_feature_names())
    for period_ in ['day', 'week', 'month']:
        pool = Pool(num_workers)
        unique_time_periods = data[f'date_{period_}'].unique()
        bows = np.zeros((unique_time_periods.max()+1, vocab_size), dtype=np.int32)
        for r in tqdm.tqdm(pool.imap_unordered(partial(group_by, data=data, period=period_, bow_transformer=bow_transformer), unique_time_periods), total=len(unique_time_periods)):
            bows[r[0]] += r[1]
        pool.close()
        pool.join()

        pickle.to_pickle(csr_matrix(bows), os.path.join(input, f'data_bow_ranking_{period_}.zip'))


if __name__ == '__main__':
    main()
