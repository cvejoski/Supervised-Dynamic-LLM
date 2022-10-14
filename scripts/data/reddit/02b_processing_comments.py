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


def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"https?://\S+|www\.\S+", "", sample)


def remove_emoji(string):
    # https://stackoverflow.com/a/49146722/330558
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


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

    data.rename(columns={'body': 'text'}, inplace=True)
    return data


@click.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Path to the downloaded Reddit data.')
@click.option('--sample-ratio', '-s', type=float, default=None, help='Sub-sample ratio per dataset.')
@click.option('--all-documents', is_flag=True)
def main(input: click.Path(exists=True), sample_ratio=None, all_documents=False):

    files = glob.glob(os.path.join(input, '*.csv'))
    all_data = []
    for filename in tqdm(files, desc="Merging files"):
        data: pd.DataFrame = pd.read_csv(filename, index_col='id', low_memory=False)

        if not all_documents:
            data = data.query('score > 1')
        data.dropna(subset=['body'], inplace=True)
        data = data[(data.body != '[deleted]') & (data.body != '[removed]')]
        data = data.query("author != 'AutoModerator'")
        data = data.replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n',  ' ', regex=True)
        data['body'] = data.body.apply(lambda x: remove_URL(x))
        mask = data.body.apply(lambda doc: [w.lower() for w in word_tokenize(doc) if w.lower() not in stop_words and w.isalpha()]).str.len().gt(50)
        data = data[mask]
        data.drop_duplicates(subset=['body'], inplace=True)
        if sample_ratio:
            data = data.sample(frac=sample_ratio, random_state=1)
        all_data.append(data)
    all_data: pd.DataFrame = pd.concat(all_data)
    all_data = preprocess(all_data)
    all_data.drop(columns=['permalink', 'submission_id', 'created_utc'], inplace=True)
    print(f"Number of rows {len(all_data)}")
    os.makedirs(os.path.join(input, 'aggregated'), exist_ok=True)
    if all_documents:
        all_data.sort_values(by='created_datetime').to_csv(os.path.join(input, 'aggregated', f'data_all.csv'))
    else:
        all_data.sort_values(by='created_datetime').to_csv(os.path.join(input, 'aggregated', f'data.csv'))


if __name__ == '__main__':
    main()
