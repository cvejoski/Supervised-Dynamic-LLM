from email.policy import default
from operator import index
import click


import pandas as pd
import glob
import os
import re
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def preprocess(data):
    data.sort_values(by=['created_utc'], inplace=True, axis=0)
    data.loc[:, 'created_datetime'] = pd.to_datetime(data.created_utc, unit='s')
    min_date = data.created_datetime.min()
    date_dt = (data.created_datetime - min_date)
    data.loc[:, 'date_hour'] = (date_dt / np.timedelta64(1, 'h')).astype('int32', copy=False)
    data.loc[:, 'date_day'] = (date_dt / np.timedelta64(1, 'D')).astype('int32', copy=False)
    data.loc[:, 'date_week'] = (date_dt / np.timedelta64(1, 'W')).astype('int32', copy=False)
    data.loc[:, 'date_month'] = (date_dt / np.timedelta64(1, 'M')).astype('int32', copy=False)
    data.loc[:, 'author'] = data.author.astype('category')
    data.loc[:, 'author_code'] = data.author.cat.codes

    data.rename(columns={'selftext': 'text'}, inplace=True)
    return data


@click.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Path to the downloaded Reddit data.', default='C:\\Projects\\CommonWork\\deep_random_fields\\data\\raw\\reddit\\tifu\\submissions')
@click.option('--sample-ratio', '-s', type=float, default=None, help='Sub-sample ratio per dataset.')
@click.option('--min-comments', type=int, default=0, help="Minimum number of comments for the given post.")
@click.option('--min-score', type=int, default=0, help="Minimum score value for the given post.")
def main(input: click.Path(exists=True),  sample_ratio=None, min_comments=0, min_score=0):

    files = glob.glob(os.path.join(input, '*.csv'))
    comments, score = [], []
    pb = tqdm(files, desc="Merging files")

    def filter_docs(doc):
        if not isinstance(doc, str):
            print(doc)
            doc = ''
        return [w.lower() for w in word_tokenize(doc) if w.lower() not in stop_words and w.isalpha()]

    for filename in pb:
        pb.set_postfix_str(f"Merging Files: {filename}")
        data = pd.read_csv(filename, index_col='id', low_memory=False, encoding='utf-8',  lineterminator='\n')
        data.replace({'[removed]': np.NaN,  '[deleted]': np.NaN}, inplace=True)
        data.dropna(subset=['selftext', 'title'], how='all', inplace=True)
        data = data.query("author != 'AutoModerator'")
        data['selftext'] = data['selftext'].fillna("")
        data['text'] = data.title + " " + data.selftext
        data.drop(columns=data.columns.difference(['author', 'text', 'num_comments', 'score', 'full_link', 'created_utc']), inplace=True)
        data = data.replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n',  ' ', regex=True).replace(r' \n',  ' ', regex=True).replace(r'\n ',  ' ', regex=True)
        data.drop_duplicates(subset=['text'], inplace=True)
        mask = data.text.apply(filter_docs).str.len().gt(200)
        data = data[mask]
        data_comments = data.query(f'num_comments >= {min_comments}')
        data_score = data.query(f'score >= {min_score}')
        if sample_ratio:
            data_comments = data_comments.sample(frac=sample_ratio, random_state=1)
            data_score = data_score.sample(frac=sample_ratio, random_state=1)
        comments.append(data_comments)
        score.append(data_score)
        # pb.update()
    data_comments = pd.concat(comments)
    data_score = pd.concat(score)

    data_comments = preprocess(data_comments)
    data_score = preprocess(data_score)
    data_comments.drop(columns=['created_utc'], inplace=True)
    data_score.drop(columns=['created_utc'], inplace=True)

    print(f"Comments Size {len(data_comments)}")
    print(f"Score Size {len(data_score)}")

    os.makedirs(os.path.join(input, 'aggregated'), exist_ok=True)

    data_score.sort_values(by='created_datetime').to_csv(os.path.join(input, 'aggregated', f'data_{min_score}_score.csv'), index=False)
    data_comments.sort_values(by='created_datetime').to_csv(os.path.join(input, 'aggregated', f'data_{min_comments}_comments.csv'), index=False)


if __name__ == '__main__':
    main()
