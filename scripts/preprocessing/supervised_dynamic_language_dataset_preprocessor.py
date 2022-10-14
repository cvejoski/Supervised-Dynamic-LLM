import csv
import glob
import os
import pickle
import math
import warnings
from multiprocessing import Pool
from operator import itemgetter
from functools import partial
from typing import List, Tuple
import click
import operator
import numpy as np
import tqdm
from deep_fields.models.generative_models.text.utils import (
    SPECIAL_TOKENS, tokenize_doc_transformers, count_lines_in_file,
    preprocess_text, tokenize_doc, nltk_word_tokenizer, basic_english_normalize, is_english)

from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split


parent_path = os.path.dirname(__file__)

warnings.filterwarnings("ignore")
# Read stopwords
with open(os.path.join(parent_path, 'stops.txt'), 'r') as f:
    STOPS = f.read().split('\n')


def fix_nulls(s):
    for line in s:
        yield line.replace('\0', ' ')


@click.command()
@click.option('-i', '--input', type=click.Path(exists=True), required=True, help="Input directory of the raw data.")
@click.option('-e', '--embeddings', 'embeddings_path', type=click.Path(exists=True), required=True, help="Input directory of the embeddings.")
@click.option('-o', '--output', type=click.Path(exists=False), required=True, help="Output directory of the pre-processed data.")
@click.option('-min-df-tp', '--min-df-tp', default=5, type=int, required=True, help="Minimum document frequency for the topic model.")
@click.option('-min-df-lm', '--min-df-lm', default=5, type=int, required=True, help="Minimum document frequency for the language model.")
@click.option('-max-df', '--max-df', default=1.0, type=float, required=True, help="Maximum document frequency.")
@click.option('-bow-vocab-size', default=None, type=int, help="Size of the BoW vocabulary. It takes the most frequent N words.")
@click.option('-psteps', '--prediction-steps', 'p_steps', default=1, type=int, required=True, help="Split the data along the time axis. Number of timesteps for prediction")
@click.option('-tt-ratio', '--train-test-ratio', 'tt_ratio', nargs=2, type=float, required=True, help="Split the train data into train/validate.")
@click.option('-max-doc-len', '--max-doc-len', default=None, type=int, help="Max len of a document")
@click.option('-min-sent-len', '--min-sent-len', default=1, type=int, help="Min len of a sentence")
@click.option('-max-sent-len', '--max-sent-len', default=None, type=int, help="Max len of a sentence")
@click.option('--date-field-name', default='date', type=str, help='Name of the date field')
@click.option('--reward-column', type=str, required=False, help='Name of the reward column.')
@click.option('--random-seed', default=87, type=int, required=True, help='Random seed for spliting the data')
@click.option('--num-workers', default=1, type=int, required=True, help='Number of workers for paralle preprocessing and tokenizatioin')
@click.option('--binarize-reward', '-br', multiple=True, default=[], type=int, help="Binarize the reward into bins (ex. [0, 1, 2, 11, 101, 1001, 5001, 10001, 50000])")
@click.option('--covariates', '-c', multiple=True, required=False, type=(str, str), help="Name of the dataset attributes that are used as covariates")
@click.option('--total-num-time-steps', default=None, type=int, help='Total number of timesteps to be used')
@click.option('--samples-per-timestep', default=None, type=int, help="Subsample the documents per each time step")
@click.option('--split-by-paragraph', is_flag=True, help='Take the abstracts as document text')
@click.option('--remove-persons-organizations', is_flag=True, help='Remove person and organizations from the text.')
@click.option('--max-reward', default=None, type=int, help='Maximum value of the reward.')
def preprocess(input: str, output: str, min_df_tp: int, min_df_lm: int,
               max_df: float, max_doc_len: int,  min_sent_len: int,
               max_sent_len: int, p_steps: int, tt_ratio: tuple, embeddings_path: str, reward_column: str, covariates: List[tuple],
               binarize_reward: list, random_seed: int, num_workers: int, date_field_name: str, total_num_time_steps: int,
               samples_per_timestep: int, split_by_paragraph: bool, remove_persons_organizations: bool, bow_vocab_size: int,
               max_reward: int):

    csv.field_size_limit(1310720)
    np.random.seed(random_seed)
    all_docs, all_timestamps, all_rewards, all_rewards_bin, all_covariates = _read_docs(input, False, reward_column, binarize_reward,
                                                                                        covariates, date_field_name, samples_per_timestep, total_num_time_steps, max_reward)

    print("Sorting Documents")

    all_timestamps_, all_docs_, all_rewards_, all_rewards_bin_, all_covariates_ = zip(
        *sorted(zip(all_timestamps, all_docs, all_rewards, all_rewards_bin, all_covariates), key=itemgetter(0)))
    if split_by_paragraph:
        print('splitting by paragraphs ...')
        all_docs = []
        all_timestamps = []
        all_rewards = []
        all_rewards_bin = []
        all_covariates = []
        for dd, doc in tqdm.tqdm(enumerate(all_docs_), total=len(all_docs_)):
            splitted_doc = doc.split('.\n')
            for ii in splitted_doc:
                all_docs.append(ii)
                all_timestamps.append(all_timestamps_[dd])
                if reward_column is not None:
                    all_rewards.append(all_rewards_[dd])
                    all_rewards_bin.append(all_rewards_bin_[dd])
                if covariates:
                    all_covariates.append(all_covariates_[dd])
    else:
        all_docs = list(all_docs_)
        all_timestamps = list(all_timestamps_)
        if reward_column is not None:
            all_rewards = list(all_rewards_)
            all_rewards_bin = list(all_rewards_bin_)
        if covariates:
            all_covariates = list(all_covariates_)

    print('SENTENCE: ')
    print('         tokenization and preprocessing ...')

    all_docs_sent_ = []
    pool = Pool(num_workers)
    f = partial(tokenize_doc, max_doc_len=max_doc_len)
    for r in tqdm.tqdm(pool.imap(f, all_docs, 100), total=len(all_docs), desc="Tokenize documents for GRU/LSTM LM"):
        all_docs_sent_.append(r)
    pool.close()
    pool.join()
    all_docs_sent_transformer = tokenize_doc_transformers(all_docs, max_doc_len)

    print('         counting document frequency of words ...')
    # all_docs_sent_flatten = [s for doc in all_docs_sent_ for s in doc]
    cvectorizer = CountVectorizer(min_df=min_df_lm, stop_words=None, tokenizer=nltk_word_tokenizer)
    cvectorizer.fit(all_docs_sent_)

    # Get vocabulary
    print('         building the vocabulary sentence ...')
    vocab_sent = cvectorizer.vocabulary_
    del cvectorizer
    print(f'         vocabulary size: {len(vocab_sent)}')

    # Remove punctuation
    print('BOW: \n'
          '         removing punctuation ...')

    pool = Pool(num_workers)
    all_docs_bow = []
    for r in tqdm.tqdm(pool.imap(preprocess_text, all_docs, 100), total=len(all_docs)):
        all_docs_bow.append(r)
    pool.close()
    pool.join()
    del all_docs

    print('         counting document frequency of words ...')
    cvectorizer = CountVectorizer(min_df=min_df_tp, max_df=max_df, stop_words=None)
    cvz = cvectorizer.fit_transform(all_docs_bow).sign()
    vocab_bow = cvectorizer.vocabulary_
    del cvectorizer

    sum_words = cvz.sum(0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vocab_bow.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    words_freq = [w for w in words_freq if w[0] not in STOPS]
    if bow_vocab_size is None:
        vocab_bow = [w[0] for w in words_freq]
    else:
        vocab_bow = [w[0] for w in words_freq[:bow_vocab_size]]

    # Filter out stopwords (if any)

    vocab_bow_size = len(vocab_bow)
    word2id = dict([(w, j) for j, w in enumerate(vocab_bow)])

    print(f'         vocabulary size after removing stopwords from list: {vocab_bow_size}')

    # Create mapping of timestamps
    all_times = sorted(set(all_timestamps))
    time2id = dict([(t, i) for i, t in enumerate(all_times)])
    id2time = dict([(i, t) for i, t in enumerate(all_times)])
    time_list = [id2time[i] for i in range(len(all_times))]

    # Split in train/test/valid
    print('tokenizing documents and splitting into train/test/valid/prediction...')

    num_time_points = len(time_list)
    tr_size = num_time_points - p_steps
    pr_size = num_time_points - tr_size
    print(f'total number of time steps: {num_time_points}')
    print(f'total number of train time steps: {tr_size}')
    del cvz

    max_reward = max(all_rewards)
    min_reward = min(all_rewards)
    tr_docs = list(filter(lambda x: x[3] in all_times[:tr_size], zip(all_docs_bow,  all_docs_sent_transformer, all_docs_sent_, all_timestamps, all_rewards, all_rewards_bin, all_covariates)))
    pr_docs = list(filter(lambda x: x[3] in all_times[tr_size:tr_size + pr_size], zip(all_docs_bow,
                   all_docs_sent_transformer, all_docs_sent_, all_timestamps, all_rewards, all_rewards_bin, all_covariates)))

    # ts_tr = [t for t in all_timestamps if t in all_times[:tr_size]]
    # ts_pr = [t for t in all_timestamps if t in all_times[tr_size:tr_size + pr_size]]

    tr_size = int(np.floor(len(tr_docs) * tt_ratio[0]))
    ts_size = int(np.floor(len(tr_docs) * tt_ratio[1]))
    va_size = int(len(tr_docs) - tr_size - ts_size)

    tr_docs, va_docs = train_test_split(tr_docs, train_size=tt_ratio[0], random_state=random_seed)
    va_docs, te_docs = train_test_split(va_docs, train_size=va_size, random_state=random_seed)

    print('  removing words from vocabulary not in training set ...')
    vocab_bow = list(set([w for doc in tr_docs for w in doc[0].split() if w in word2id]))
    vocab_bow_size = len(vocab_bow)
    print('         bow vocabulary after removing words not in train: {}'.format(vocab_bow_size))
    vocab_sent = list(set([w for doc in tr_docs for w in basic_english_normalize(doc[2]) if w in vocab_sent]))
    vocab_sent_size = len(vocab_sent)
    print('         sentence vocabulary after removing words not in train: {}'.format(vocab_sent_size))

    print(f'Train Size {len(tr_docs)}')
    print(f'Validation Size {len(va_docs)}')
    print(f'Test Size {len(te_docs)}')
    print(f'Prediction Size {len(pr_docs)}')

    vocab = vocab_bow + list(set(vocab_sent) - set(vocab_bow))
    vocab.extend(SPECIAL_TOKENS)
    print(f' total vocabulary size: {len(vocab)}')

    # Create dictionary and inverse dictionary
    print('  create dictionary and inverse dictionary ')
    word2id = dict([(w, j) for j, w in enumerate(vocab)])
    id2word = dict([(j, w) for j, w in enumerate(vocab)])

    print('tokenizing bow ...')
    docs_b_tr = tokenize_bow(tr_docs, vocab_bow_size, word2id)
    docs_b_va = tokenize_bow(va_docs, vocab_bow_size, word2id)
    docs_b_te = tokenize_bow(te_docs, vocab_bow_size, word2id)
    docs_b_pr = tokenize_bow(pr_docs, vocab_bow_size, word2id)
    del all_docs_bow

    print('tokenizing sentences ...')
    unk_ix = word2id['<unk>']
    docs_s_tr = tokenize_document(tr_docs, unk_ix, word2id)
    docs_s_va = tokenize_document(va_docs, unk_ix, word2id)
    docs_s_te = tokenize_document(te_docs, unk_ix, word2id)
    docs_s_pr = tokenize_document(pr_docs, unk_ix, word2id)
    del all_docs_sent_

    docs_r_tr = [float(doc[-3]) for doc in tr_docs]
    docs_r_va = [float(doc[-3]) for doc in va_docs]
    docs_r_te = [float(doc[-3]) for doc in te_docs]
    docs_r_pr = [float(doc[-3]) for doc in pr_docs]

    docs_r_bin_tr = [float(doc[-2]) for doc in tr_docs]
    docs_r_bin_va = [float(doc[-2]) for doc in va_docs]
    docs_r_bin_te = [float(doc[-2]) for doc in te_docs]
    docs_r_bin_pr = [float(doc[-2]) for doc in pr_docs]

    docs_covar_tr = [doc[-1] for doc in tr_docs]
    docs_covar_va = [doc[-1] for doc in va_docs]
    docs_covar_te = [doc[-1] for doc in te_docs]
    docs_covar_pr = [doc[-1] for doc in pr_docs]

    docs_t_tr = [doc[1] for doc in tr_docs]
    docs_t_va = [doc[1] for doc in va_docs]
    docs_t_te = [doc[1] for doc in te_docs]
    docs_t_pr = [doc[1] for doc in pr_docs]

    docs_ts_tr = [doc[3] for doc in tr_docs]
    docs_ts_va = [doc[3] for doc in va_docs]
    docs_ts_te = [doc[3] for doc in te_docs]
    docs_ts_pr = [doc[3] for doc in pr_docs]

    # Remove empty documents
    print('removing empty documents ...')

    def remove_empty(in_docs_b, in_docs_trans, in_docs_text, in_timestamps, in_reward, in_reward_bin, in_covar):
        out_docs_b = []
        out_docs_trans, out_docs_text = [], []
        out_timestamps = []

        out_reward_bin, out_reward = [], []
        out_covar = []
        for ii, doc in enumerate(in_docs_b):
            if doc:
                out_docs_b.append(doc)
                out_docs_trans.append(in_docs_trans[ii])
                out_docs_text.append(in_docs_text[ii])
                out_timestamps.append(in_timestamps[ii])
                if in_reward:
                    out_reward.append(in_reward[ii])
                    out_reward_bin.append(in_reward_bin[ii])
                if in_covar:
                    out_covar.append(in_covar[ii])

        return out_docs_b, out_docs_trans, out_docs_text, out_timestamps, out_reward, out_reward_bin, out_covar

    def remove_by_threshold(in_docs_b, in_docs_trans, in_docs_text, in_timestamps, in_docs_r, in_docs_r_bin, in_docs_c, thr):
        out_docs_b = []
        out_docs_trans, out_docs_text = [], []
        out_timestamps = []
        out_docs_r = []
        out_docs_r_bin = []
        out_docs_c = []
        for ii, doc in enumerate(in_docs_b):
            if len(doc) > thr:
                out_docs_b.append(doc)
                out_docs_trans.append(in_docs_trans[ii])
                out_docs_text.append(in_docs_text[ii])
                out_timestamps.append(in_timestamps[ii])
                if reward_column is not None:
                    out_docs_r.append(in_docs_r[ii])
                    out_docs_r_bin.append(in_docs_r_bin[ii])
                if covariates:
                    out_docs_c.append(in_docs_c[ii])
        return out_docs_b, out_docs_trans, out_docs_text, out_timestamps, out_docs_r, out_docs_r_bin, out_docs_c

    docs_b_tr, docs_t_tr, docs_s_tr, docs_ts_tr, docs_r_tr, docs_r_bin_tr, docs_covar_tr = remove_empty(docs_b_tr, docs_t_tr, docs_s_tr, docs_ts_tr,
                                                                                                        docs_r_tr, docs_r_bin_tr, docs_covar_tr)
    docs_b_va, docs_t_va, docs_s_va, docs_ts_va, docs_r_va, docs_r_bin_va, docs_covar_va = remove_empty(docs_b_va, docs_t_va, docs_s_va, docs_ts_va, docs_r_va,
                                                                                                        docs_r_bin_va, docs_covar_va)
    docs_b_te, docs_t_te, docs_s_te, docs_ts_te, docs_r_te, docs_r_bin_te, docs_covar_te = remove_empty(docs_b_te, docs_t_te, docs_s_te, docs_ts_te, docs_r_te,
                                                                                                        docs_r_bin_te, docs_covar_te)
    docs_b_pr, docs_t_pr, docs_s_pr, docs_ts_pr, docs_r_pr, docs_r_bin_pr, docs_covar_pr = remove_empty(docs_b_pr,  docs_t_pr, docs_s_pr, docs_ts_pr, docs_r_pr,
                                                                                                        docs_r_bin_pr, docs_covar_pr)

    # Remove prediction and test documents with length=1
    docs_b_pr, docs_t_pr, docs_s_pr, docs_ts_pr, docs_r_pr, docs_r_bin_pr, docs_covar_pr = remove_by_threshold(
        docs_b_pr, docs_t_pr, docs_s_pr, docs_ts_pr, docs_r_pr, docs_r_bin_pr, docs_covar_pr, 1)
    docs_b_te, docs_t_te, docs_s_te, docs_ts_te, docs_r_te, docs_r_bin_te, docs_covar_te = remove_by_threshold(
        docs_b_te, docs_t_te, docs_s_te, docs_ts_te, docs_r_te, docs_r_bin_te, docs_covar_te, 1)

    # Split test set in 2 halves
    print('splitting test documents in 2 halves...')
    docs_te_h1 = [[w for i, w in enumerate(doc) if i <= len(doc) / 2.0 - 1] for doc in docs_b_te]
    docs_te_h2 = [[w for i, w in enumerate(doc) if i > len(doc) / 2.0 - 1] for doc in docs_b_te]

    # Getting lists of words and doc_indices
    print('creating lists of words...')

    def create_list_words(in_docs):
        return [x for y in in_docs for x in y]

    words_tr = create_list_words(docs_b_tr)
    words_va = create_list_words(docs_b_va)
    words_te = create_list_words(docs_b_te)
    words_te_h1 = create_list_words(docs_te_h1)
    words_te_h2 = create_list_words(docs_te_h2)
    words_pr = create_list_words(docs_b_pr)

    print('  len(words_tr): ', len(words_tr))
    print('  len(words_va): ', len(words_va))
    print('  len(words_va): ', len(words_te))
    print('  len(words_te_h1): ', len(words_te_h1))
    print('  len(words_te_h2): ', len(words_te_h2))
    print('  len(words_pr): ', len(words_pr))

    # Get doc indices
    print('getting doc indices...')

    def create_doc_indices(in_docs):
        aux = [[j for _ in range(len(doc))] for j, doc in enumerate(in_docs)]
        return [int(x) for y in aux for x in y]

    doc_indices_tr = create_doc_indices(docs_b_tr)
    doc_indices_va = create_doc_indices(docs_b_va)
    doc_indices_te = create_doc_indices(docs_b_te)
    doc_indices_te_h1 = create_doc_indices(docs_te_h1)
    doc_indices_te_h2 = create_doc_indices(docs_te_h2)
    doc_indices_pr = create_doc_indices(docs_b_pr)

    print('  len(np.unique(doc_indices_tr)): {} [this should be {}]'.format(len(np.unique(doc_indices_tr)), len(docs_b_tr)))
    print('  len(np.unique(doc_indices_va)): {} [this should be {}]'.format(len(np.unique(doc_indices_va)), len(docs_b_va)))
    print('  len(np.unique(doc_indices_te)): {} [this should be {}]'.format(len(np.unique(doc_indices_te)), len(docs_b_te)))
    print('  len(np.unique(doc_indices_te_h1)): {} [this should be {}]'.format(len(np.unique(doc_indices_te_h1)), len(docs_te_h1)))
    print('  len(np.unique(doc_indices_te_h2)): {} [this should be {}]'.format(len(np.unique(doc_indices_te_h2)), len(docs_te_h2)))
    print('  len(np.unique(doc_indices_pr)): {} [this should be {}]'.format(len(np.unique(doc_indices_pr)), len(docs_b_pr)))

    # Number of documents in each set
    n_docs_tr = len(docs_b_tr)
    n_docs_va = len(docs_b_va)
    n_docs_te = len(docs_b_te)
    n_docs_te_h1 = len(docs_te_h1)
    n_docs_te_h2 = len(docs_te_h2)
    n_docs_pr = len(docs_b_pr)

    # Create bow representation
    print('creating bow representation...')

    def create_bow(doc_indices, words, n_docs, vocab_size):
        return sparse.coo_matrix(([1] * len(doc_indices), (doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()

    bow_tr = create_bow(doc_indices_tr, words_tr, n_docs_tr, vocab_bow_size)
    bow_va = create_bow(doc_indices_va, words_va, n_docs_va, vocab_bow_size)
    bow_te = create_bow(doc_indices_te, words_te, n_docs_te, vocab_bow_size)
    bow_te_h1 = create_bow(doc_indices_te_h1, words_te_h1, n_docs_te_h1, vocab_bow_size)
    bow_te_h2 = create_bow(doc_indices_te_h2, words_te_h2, n_docs_te_h2, vocab_bow_size)
    bow_pr = create_bow(doc_indices_pr, words_pr, n_docs_pr, vocab_bow_size)

    del words_tr
    del words_pr
    del words_va
    del words_te_h1
    del words_te_h2
    del doc_indices_tr
    del doc_indices_pr
    del doc_indices_va
    del doc_indices_te_h1
    del doc_indices_te_h2
    del doc_indices_te

    print('bow => tf idf')
    tfidf_trfm = TfidfTransformer(norm=None)
    tfidf_tr = tfidf_trfm.fit_transform(bow_tr)
    tfidf_va = tfidf_trfm.transform(bow_va)
    tfidf_te = tfidf_trfm.transform(bow_te)
    tfidf_te_h1 = tfidf_trfm.transform(bow_te_h1)
    tfidf_te_h2 = tfidf_trfm.transform(bow_te_h2)
    if pr_size != 0:
        tfidf_pr = tfidf_trfm.transform(bow_pr)
    else:
        tfidf_pr = None

    pad_ix = word2id['<pad>']

    if max_doc_len is None:
        max_doc_len = max(list(map(len, [d for d in docs_s_tr + docs_s_pr + docs_s_va + docs_s_te])))

    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    def pad_doc(doc: List[list], max_len: int) -> Tuple[List[list], List[list], List[int]]:
        l_ = len(doc)
        if l_ > max_len:
            doc = doc[:max_len]
            l_ = max_len

        padding = [pad_ix] * (max_len - l_)
        x = doc + padding

        return x, l_

    def pad_dataset(dataset: List[str]) -> Tuple[list, list, list]:
        _x, s_l = [], []
        for i in range(len(dataset)):
            x, sl = pad_doc(dataset[i], max_doc_len)
            _x.append(x)
            s_l.append(sl)
        return _x, s_l

    print('padding training dataset')
    docs_s_tr = pad_dataset(docs_s_tr)

    print('exporting training dataset...')
    ts_tr_id = [time2id[t] for t in docs_ts_tr]
    docs_r_tr_normalized = (np.asarray(docs_r_tr) - min_reward) / (max_reward - min_reward)
    train_dataset = {'text': docs_t_tr, 'seq2seq': docs_s_tr, 'time': ts_tr_id, 'bow': bow_tr, 'tfidf': tfidf_tr, 'reward': docs_r_tr,
                     'reward_normalized': docs_r_tr_normalized, 'reward_bin': docs_r_bin_tr, 'covariates': np.asarray(docs_covar_tr, dtype=np.float32)}

    with open(os.path.join(output, 'train.pkl'), 'wb') as out:
        pickle.dump(train_dataset, out, protocol=4)

    del train_dataset
    del docs_t_tr
    del docs_s_tr
    del ts_tr_id
    del docs_covar_tr
    del tfidf_tr
    del docs_r_tr
    del docs_r_bin_tr
    del docs_r_tr_normalized

    print('padding validation dataset')
    docs_s_va = pad_dataset(docs_s_va)

    print('exporting validation dataset...')
    ts_va_id = [time2id[t] for t in docs_ts_va]
    docs_r_va_normalized = (np.asarray(docs_r_va) - min_reward) / (max_reward - min_reward)
    validation_dataset = {'text': docs_t_va, 'seq2seq': docs_s_va, 'time': ts_va_id, 'bow': bow_va, 'tfidf': tfidf_va,
                          'reward_normalized': docs_r_va_normalized, 'reward': docs_r_va, 'reward_bin': docs_r_bin_va, 'covariates': np.asarray(docs_covar_va, dtype=np.float32)}

    with open(os.path.join(output, 'validation.pkl'), 'wb') as out:
        pickle.dump(validation_dataset, out, protocol=4)

    del validation_dataset
    del docs_t_va
    del docs_s_va
    del ts_va_id
    del bow_va
    del tfidf_va
    del docs_r_bin_va
    del docs_r_va
    del docs_r_va_normalized

    print('padding test dataset')
    docs_s_te = pad_dataset(docs_s_te)

    print('exporting test dataset...')
    ts_te_id = [time2id[t] for t in docs_ts_te]
    docs_r_te_normalized = (np.asarray(docs_r_te) - min_reward) / (max_reward - min_reward)
    test_dataset = {'text': docs_t_te, 'seq2seq': docs_s_te, 'time': ts_te_id, 'bow_h1': bow_te_h1, 'bow_h2': bow_te_h2, 'bow': bow_te, 'tfidf_h1': tfidf_te_h1,
                    'reward_normalized': docs_r_te_normalized, 'tfidf_h2': tfidf_te_h2, 'tfidf': tfidf_te, 'reward': docs_r_te, 'reward_bin': docs_r_bin_te,
                    'covariates': np.asarray(docs_covar_te, dtype=np.float32)}

    with open(os.path.join(output, 'test.pkl'), 'wb') as out:
        pickle.dump(test_dataset, out, protocol=4)

    del test_dataset
    del docs_t_te
    del docs_s_te
    del ts_te_id
    del bow_te_h1
    del bow_te_h2
    del bow_te
    del tfidf_te_h1
    del tfidf_te_h2
    del tfidf_te
    del docs_r_te
    del docs_r_bin_te
    del docs_r_te_normalized

    print('padding prediction dataset')
    docs_s_pr = pad_dataset(docs_s_pr)

    print('export prediction dataset...')
    ts_pr_id = [time2id[t] for t in docs_ts_pr]
    docs_r_pr_normalized = (np.asarray(docs_r_pr) - min_reward) / (max_reward - min_reward)
    prediction_dataset = {'text': docs_t_pr, 'seq2seq': docs_s_pr, 'time': ts_pr_id, 'bow': bow_pr, 'tfidf': tfidf_pr,
                          'reward_normalized': docs_r_pr_normalized, 'reward': docs_r_pr, 'reward_bin': docs_r_bin_pr,
                          'covariates': np.asarray(docs_covar_pr, dtype=np.float32)}

    with open(os.path.join(output, 'prediction.pkl'), 'wb') as out:
        pickle.dump(prediction_dataset, out, protocol=4)

    del prediction_dataset
    del docs_t_pr
    del docs_s_pr
    del ts_pr_id
    del bow_pr
    del tfidf_pr
    del docs_r_bin_pr
    del docs_r_pr_normalized

    time = {'all_time': all_times, 'time2id': time2id, 'id2time': id2time}
    with open(os.path.join(output, 'time.pkl'), 'wb') as out:
        pickle.dump(time, out)

    del time
    del all_times
    del time2id
    del id2time

    print('counting words ...')

    word_counts = np.squeeze(np.asarray((bow_tr > 0).sum(axis=0)))
    word_counts = dict(zip(range(len(word_counts)), word_counts.tolist()))

    embeddings = __load_embeddings(embeddings_path)
    e_size = embeddings['the'].shape[0]
    vectors = list(map(lambda x: embeddings.get(x, np.random.randn(e_size)), vocab))

    vocabulary = {'vocab': vocab, 'stoi': word2id, 'itos': id2word, 'word_count': dict(word_counts), 'vectors': np.asarray(vectors, dtype=np.float32)}

    with open(os.path.join(output, 'vocabulary.pkl'), 'wb') as out:
        pickle.dump(vocabulary, out)


def tokenize_bow(tr_docs, vocab_bow_size, word2id):
    docs_b_tr = [[word2id[w] for w in doc[0].split() if w in word2id and word2id[w] < vocab_bow_size] for doc in tr_docs]
    return docs_b_tr


def __load_embeddings(embeddings_path):
    print("load embeddings...")
    embeddings = dict()
    with open(embeddings_path, 'rb') as f:
        for row in f:
            line = row.decode().split()
            word = line[0]
            vect = np.array(line[1:]).astype(np.float)
            embeddings[word] = vect
    e_size = embeddings['the'].shape[0]
    for tok in SPECIAL_TOKENS:
        embeddings[tok] = np.zeros(e_size)
    return embeddings


def get_type_conversion_function(type: str):
    if type == 'str':
        return lambda x: str(x)
    elif type == 'bool':
        return lambda x: 1.0 if x == 'True' else 0.0
    elif type == 'int':
        return lambda x: -1.0 if x == '' else int(x)
    elif type == 'float':
        return lambda x: -1.0 if x == '' else float(x)
    else:
        raise TypeError(f"Unknown type {type}")


def _read_docs(input, is_abstract: bool, reward: str, binarize_reward: list, covariates: List[tuple], date_field_name: str,  n_docs_per_time: int,
               total_num_time_steps: int, max_reward: int):
    all_timestamps, all_docs, all_rewards, all_rewards_bin, all_covar = [], [], [], [], []
    if total_num_time_steps is None:
        total_num_time_steps = math.inf
    if os.path.isfile(input):
        files = [input]
    else:
        files = filter(os.path.isfile, glob.glob(os.path.join(input, "*")))
    for file_path in files:
        n_rows = count_lines_in_file(file_path)
        with open(file_path, 'r', encoding='utf-8', newline='') as out:
            csv_reader = csv.reader(fix_nulls(out), delimiter=',', quotechar='"')
            header: list = next(csv_reader)
            date_ix = header.index(date_field_name)
            text_ix = header.index('text')
            if covariates:
                cov_ix = [(header.index(n[0]), get_type_conversion_function(n[1])) for n in covariates]
            if reward is not None:
                reward_id = header.index(reward)
            if is_abstract:
                text_ix = header.index('abstract')

            print(f"Reading data with header: {header}")
            count = 0
            p_bar = tqdm.tqdm(desc=f"Reading documents: {file_path}", unit="document", total=n_rows)
            while True:
                try:
                    row = next(csv_reader)
                    date = row[date_ix]
                    text = row[text_ix]
                    if not is_english(text):
                        continue

                    if '-' in date:
                        year, month = date.split('-')
                        date = int(year) * 100 + int(month)
                    else:
                        date = int(date)
                    if date > total_num_time_steps:
                        continue
                    if reward is not None:
                        if float(row[reward_id]) > max_reward:
                            continue
                        all_rewards_bin.append(np.digitize(float(row[reward_id]), binarize_reward) - 1)
                        all_rewards.append(float(row[reward_id]))
                    all_timestamps.append(date)
                    all_docs.append(text)
                    if covariates:
                        try:
                            all_covar.append([convert_fn(row[ix]) for ix, convert_fn in cov_ix])
                        except Exception as e:
                            print(e)

                except StopIteration:
                    break
                except Exception as e:
                    print(f"Exception for row {count}: {e}")

                finally:
                    count += 1
                    p_bar.update()
                # if count > 3_000:
                #     break
    if n_docs_per_time is not None:
        all_docs, all_timestamps, all_rewards, all_rewards_bin, all_covar = subsample_per_time_period(
            n_docs_per_time, all_timestamps, all_docs, all_rewards, all_rewards_bin, all_covar)

    return all_docs, all_timestamps, all_rewards, all_rewards_bin, all_covar


def subsample_per_time_period(n_documents: int, time_ids: list, docs: list,  all_rewards: list, all_rewards_bin: list, all_covar: list):
    _ids,  _docs,  _covar, _reward, _reward_bin = [], [], [], [], []
    unique_time_ix = set(time_ids)
    time_ids = np.asarray(time_ids)

    all_rewards = np.asarray(all_rewards)
    all_rewards_bin = np.asarray(all_rewards_bin)
    all_covar = np.asarray(all_covar)
    for time_ix in unique_time_ix:
        ids = np.where(time_ids == time_ix)[0]
        if len(ids) <= n_documents:
            sample_ids = ids
            n = len(sample_ids)
        else:
            sample_ids = np.random.choice(ids, n_documents)
            n = n_documents
        _docs.extend(operator.itemgetter(*sample_ids)(docs))
        _ids.extend([time_ix] * n)
        if all_covar.size != 0:
            _covar.extend(all_covar[sample_ids].tolist())
        if all_rewards.size != 0:
            _reward.extend(all_rewards[sample_ids].tolist())
            _reward_bin.extend(all_rewards_bin[sample_ids].tolist())

    return _docs, _ids, _reward, _reward_bin, _covar


def tokenize_document(docs, unk_ix, word2id):
    docs = [[word2id.get(w) for w in basic_english_normalize(doc[2]) if w in word2id] for doc in docs]
    return docs


if __name__ == '__main__':
    preprocess()
