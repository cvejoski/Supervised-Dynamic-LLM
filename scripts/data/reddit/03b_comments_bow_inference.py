import click
import tqdm
import pandas as pd
import os
from multiprocessing import Pool
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from deep_fields.models.generative_models.text.utils import preprocess_text


def to_bow(doc, model):
    bow = model.transform([doc])
    return list(bow.toarray()[0])
    # return bow.nonzero()[1]


@click.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Path to the downloaded Reddit data.')
@click.option('--subsample', '-s', type=float, default=None, help='Random subsample of the data.')
def main(input: click.Path(exists=True), subsample: float):

    if subsample is not None:
        data = pd.read_csv(input, usecols=['id', 'author', 'score', 'created_utc', 'date_hour', 'date_day', 'date_week',
                           'date_month', 'text'], low_memory=False, lineterminator="\n").sample(frac=subsample)
    else:
        data = pd.read_csv(input, usecols=['id', 'author', 'score', 'created_utc', 'date_hour', 'date_day', 'date_week', 'date_month', 'text'], low_memory=False, lineterminator="\n")
    texts = data.text.values

    pool = Pool(14)
    text_pre = []
    for r in tqdm.tqdm(pool.imap_unordered(preprocess_text, texts), total=len(texts)):
        text_pre.append(r)
    pool.close()
    pool.join()

    cvectorizer = CountVectorizer(min_df=20, max_df=0.95, stop_words=None)
    cvectorizer.fit(text_pre)

    # bows = cvectorizer.transform(text_pre)
    # vocab = cvectorizer.get_feature_names()
    data['text'] = text_pre
    root, file = os.path.split(input)
    file_name = file.split('.')[0]
    os.makedirs(os.path.join(root, 'stats'), exist_ok=True)
    if subsample is not None:
        pickle.dump(cvectorizer, open(os.path.join(root, 'stats', f'bow_model_{int(subsample*100)}.pkl'), 'wb'))
        data.to_pickle(os.path.join(root, 'stats', f'{file_name}_preprocessed_{int(subsample*100)}.zip'))
    else:
        pickle.dump(cvectorizer, open(os.path.join(root, 'stats', f'bow_model.pkl'), 'wb'))
        data.to_pickle(os.path.join(root, 'stats', f'{file_name}_preprocessed.zip'))


if __name__ == '__main__':
    main()
