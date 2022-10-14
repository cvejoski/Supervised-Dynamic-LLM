import os
from functools import partial
from multiprocessing import Pool
from typing import Tuple
from collections import defaultdict
import click
import numpy as np
import pandas as pd
import tqdm
from numpy.random import choice
from pandas.io import pickle
from scipy.sparse import csr_matrix


def group_by_author(author_name: str, data: pd.DataFrame, total_periods: Tuple[int], bow_transformer):
    data = data.loc[np.in1d(data['author'], [author_name])]

    vocab_size = len(bow_transformer.get_feature_names())
    day_ix = data.columns.get_loc(f'date_day')
    week_ix = data.columns.get_loc(f'date_week')
    month_ix = data.columns.get_loc(f'date_month')
    text_ix = data.columns.get_loc('text')
    ts_day = defaultdict(list)
    ts_week = defaultdict(list)
    ts_month = defaultdict(list)
    for row in tqdm.tqdm(data.values, desc=f"Author Name: {author_name}"):
        ts_day[row[day_ix]].append(row[text_ix])
        ts_week[row[week_ix]].append(row[text_ix])
        ts_month[row[month_ix]].append(row[text_ix])

    day_bows = np.zeros((total_periods[0], vocab_size), dtype=np.int32)
    for key, val in ts_day.items():
        day_bows[int(key)] += np.asarray(bow_transformer.transform(val).sum(axis=0)).flatten()

    week_bows = np.zeros((total_periods[1], vocab_size), dtype=np.int32)
    for key, val in ts_week.items():
        week_bows[int(key)] += np.asarray(bow_transformer.transform(val).sum(axis=0)).flatten()

    month_bows = np.zeros((total_periods[2], vocab_size), dtype=np.int32)
    for key, val in ts_month.items():
        month_bows[int(key)] += np.asarray(bow_transformer.transform(val).sum(axis=0)).flatten()
    return (author_name, csr_matrix(day_bows)),  (author_name, csr_matrix(week_bows)),  (author_name, csr_matrix(month_bows))


@click.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Path to the root dir containing the bow model and preprocessed documents')
@click.option('--num-top-authors', required=True, type=int, help='Number of top authors.')
@click.option('--num-noise-authors', required=True, type=int, default=1, help='Number of noise authors')
@click.option('--num-workers', type=int, default=1, help='Number of workers')
@click.option('--criteria', type=click.Choice(['regularity', 'score', 'consistancy'], case_sensitive=False), default='criteria', help='Criteria for choosing the top authors')
def main(input: click.Path(exists=True), num_top_authors: int, num_noise_authors: int, num_workers: int, criteria: str):

    data_path = os.path.join(input, "data_all_preprocessed.zip")
    data = pd.read_pickle(data_path)
    bow_transformer = pickle.read_pickle(open(os.path.join(input, 'bow_model.pkl'), 'rb'))

    top_authors = get_top_authors(num_top_authors, data, criteria)

    rest_authors = data[~data.author.isin(top_authors)].author.unique().tolist()
    agg_authors = choice(rest_authors, (num_noise_authors, len(rest_authors) // num_noise_authors), replace=False)

    for i in range(num_noise_authors):
        new_author = f'AGGREGATED-BY-US-{i}'
        data.loc[data.author.isin(agg_authors[i]), 'author'] = new_author
        top_authors.append(new_author)
    result_day, result_week, result_month = [], [], []

    pool = Pool(num_workers)
    unique_days = data[f'date_day'].unique()
    unique_weeks = data[f'date_week'].unique()
    unique_months = data[f'date_month'].unique()
    for r in tqdm.tqdm(pool.imap_unordered(partial(group_by_author, data=data, total_periods=(unique_days.max()+1, unique_weeks.max()+1, unique_months.max()+1), bow_transformer=bow_transformer), top_authors), total=len(top_authors)):
        result_day.append(r[0])
        result_week.append(r[1])
        result_month.append(r[2])
    pool.close()
    pool.join()

    pickle.to_pickle(result_day, os.path.join(input, f'data_bow_{criteria}_top_{num_top_authors}_noise_{num_noise_authors}_user_ranking_day.zip'))
    pickle.to_pickle(result_week, os.path.join(input, f'data_bow_{criteria}_top_{num_top_authors}_noise_{num_noise_authors}_user_ranking_week.zip'))
    pickle.to_pickle(result_month, os.path.join(input, f'data_bow_{criteria}_top_{num_top_authors}_noise_{num_noise_authors}_user_ranking_month.zip'))
    with open(os.path.join(input, 'vocab.csv'), 'w', newline='\n', encoding='utf-8') as f:
        for word in bow_transformer.get_feature_names():
            f.write(word)
            f.write('\n')


def get_top_authors(num_top_authors, data, criteria):
    if criteria == 'regularity':
        posts_per_author_per_day = data[['id', 'date_day', 'author']].groupby(['author', 'date_day']).count()
        authors_day_posts = posts_per_author_per_day.groupby('author').count().sort_values('id', ascending=False)
        top_authors = authors_day_posts[:num_top_authors]
        return top_authors.index.tolist()
    elif criteria == 'score':
        data['score_abs'] = data.score.abs()
        posts_per_author_per_day = data[['id', 'date_day', 'author', 'score_abs']].groupby(['author', 'date_day']).aggregate({'id': "count", 'score_abs': 'mean'})
        authors_day_posts = posts_per_author_per_day.groupby('author').aggregate({'id': "count", 'score_abs': 'mean'}).sort_values(['id', 'score_abs'], ascending=False)

        top_authors = authors_day_posts[:num_top_authors]
        return top_authors.index.tolist()


if __name__ == '__main__':
    main()
