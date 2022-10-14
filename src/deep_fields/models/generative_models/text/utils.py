import io
import math
import re
import string
from collections import namedtuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nltk
import numpy as np
import PIL.Image
import seaborn as sns
import spacy
import torch
import tqdm
from spacy_langdetect import LanguageDetector
from torch import nn

# from torchvision.transforms import ToTensor

nlp = spacy.load("en_core_web_sm")
nlp_ld = spacy.load("en_core_web_sm")
# nlp_ld.add_pipe(LanguageDetector(), last=True)
Vocab = namedtuple('Vocab', 'vocab, stoi, itos, word_count, vectors')
Time = namedtuple('Time', 'all_time, time2id, id2time')
EPSILON = 1e-10
EULER_GAMMA = 0.5772156649015329


def count_lines_in_file(file_path):

    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b:
                break
            yield b

    with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
        return sum(bl.count("\n") for bl in blocks(f))


def reparameterize_kumaraswamy(a, b):
    u = (1e-4 - 0.9999) * torch.rand_like(a) + 0.9999

    return torch.pow(1.0 - torch.pow(u, 1. / (b + EPSILON)), 1. / (a + EPSILON))


def reparameterize_normal(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # Sampling:

    sigma = torch.exp(0.5 * logvar)
    z = torch.randn_like(mean)
    z.mul_(sigma).add_(mean)
    return z


def chunk_docs(n_docs: int, chunk_size: int):
    for i in range(0, n_docs, chunk_size):
        yield i, min(i + chunk_size, n_docs)


def sample(dist, mode=None, unk_idx=None):
    """
    Auxiliary sampling method.
    """
    if mode in ['sample-no-unk', 'greedy-no-unk'] and unk_idx is None:
        raise ValueError('Unknown index for the <unk> token!')
    if mode == 'greedy':
        _, _sample = torch.topk(dist, 1, dim=-1)
    elif mode == 'sample':
        sample_prob = torch.nn.functional.softmax(torch.nn.functional.softplus(dist), dim=-1).squeeze(1)
        sample_prob[torch.isinf(sample_prob) | torch.isnan(sample_prob)] = 0
        _sample = torch.multinomial(sample_prob, num_samples=1)
    elif mode == 'sample-no-unk':
        # reduce chances for <unk>
        dist[:, :, unk_idx] = dist.min()
        sample_prob = torch.nn.functional.softmax(dist, dim=-1).squeeze(1)
        _sample = torch.multinomial(sample_prob, num_samples=1)
    elif mode == 'greedy-no-unk':
        # prevent <unk>
        dist[:, :, unk_idx] = dist.min()
        _, _sample = torch.topk(dist, 1, dim=-1)
    else:
        raise ValueError(f'Unknown sampling mode = {mode}')

    _sample = _sample.squeeze()

    return _sample


def nearest_neighbors(word, embeddings, vocab, num_words):
    vectors = embeddings.cpu().numpy()
    index = vocab.index(word)
    query = embeddings[index].cpu().numpy()
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors = mostSimilar[:num_words]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors


SPECIAL_TOKENS = ['<sos>', '<eos>', '<pad>', '<unk>', '<num>']
_patterns = [r'\'', r'\"', r'\.', r'<br \/>', r',', r'\(', r'\)', r'\!', r'\?', r'\;', r'\:', r'\s+', r'[0-9]+']
_replacements = [' \'  ', '', ' . ', ' ', ' , ', ' ( ', ' ) ', ' ! ', ' ? ', ' ', ' ', ' ', '<num>']
_patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def basic_english_normalize(line):
    r"""
    Basic normalization for a line of text.
    Normalization includes
    - lowercasing
    - complete some basic text normalization for English words as follows:
        add spaces before and after '\''
        remove '\"',
        add spaces before and after '.'
        replace '<br \/>'with single space
        add spaces before and after ','
        add spaces before and after '('
        add spaces before and after ')'
        add spaces before and after '!'
        add spaces before and after '?'
        replace ';' with single space
        replace ':' with single space
        replace multiple spaces with single space

    Returns a list of tokens after splitting on whitespace.
    """

    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line.split()


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def nltk_word_tokenizer(sent):
    sent = re.sub(r'\d+,\d+', 'num', sent)
    words = nltk.word_tokenize(sent)
    words = ['num' if is_number(w) else w.lower() for w in words]
    return words


def tokenize_doc(document, max_doc_len):
    document = document.replace('\n', '')
    document = remove_emoji(document)
    document = remove_urls(document)
    document = remove_other_chars(document)
    document = remove_punctuation(document)
    document = remove_stop_words(document)
    document = remove_spaces(document)
    # sentence tokenization

    document = [word for word in nltk_word_tokenizer(document.replace('\n', ' '))]

    if max_doc_len is None:
        # to lowercase
        doc = " ".join(document)
    else:
        doc = " ".join(document[:max_doc_len])

    return doc


def tokenize_doc_transformers(all_docs, max_doc_len=None):
    all_docs_sent = []
    for doc in tqdm.tqdm(all_docs, desc='Transformers Document Tokenization'):
        text = doc.replace('\n', '')
        # text = remove_emoji(text)
        text = remove_urls(text)
        text = remove_other_chars(text)
        text = remove_spaces(text)
        text_tokenized = text.split(' ')
        text = " ".join(text_tokenized[:max_doc_len])

        all_docs_sent.append(text)
    return all_docs_sent


def tokenize_docs_raw(all_docs, max_doc_len=None):
    all_docs_ = []
    for doc in all_docs:
        # sentence tokenization
        doc = nltk_word_tokenizer(doc)  # [" ".join(nltk_word_tokenizer(sent)[:max_sent_len]) for sent in doc]
        if max_doc_len is not None:
            doc = doc[:max_doc_len]

        all_docs_.append(doc)
    return all_docs_


def is_english(doc, confidence=0.95):
    doc = nlp_ld(doc)
    detect_language = doc._.language  # 4
    return detect_language['language'] == 'en' and detect_language['score'] >= confidence


def log_heatmap(value, label, number_of_iterations, writer, is_lognorm=False):
    if is_lognorm:
        sns.heatmap(value, linewidth=0., norm=mcolors.LogNorm(value.min(), value.max()))
    else:
        sns.heatmap(value, linewidth=0.)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = torch.Tensor(image).unsqueeze(0)
    # image = ToTensor()(image).unsqueeze(0)
    writer.add_images(label, image, number_of_iterations)
    plt.close()


class PositionalEncoding(nn.Module):

    def __init__(self, d_embedding, max_len=100):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_embedding)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embedding, 2).float() * (-math.log(10000.0) / d_embedding))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x


# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b


def remove_emoji(string):
    emoji_pattern = re.compile(
        "["
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
        "]+",
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def remove_spaces(x):
    return re.sub('\\s+', ' ', x).strip()


def lemmatize_words(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])


def remove_stop_words(text):
    doc = nlp(text)
    return " ".join([token.text.lower() for token in doc if not token.is_stop])


def remove_urls(doc):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', doc)


def remove_other_chars(x: str):
    return x.replace('*',
                     '').replace('#', '').replace('&x200B', '').replace("&nbsp", '').replace("&amp;", '').replace('[', '').replace(']', '').replace(
                         '; ',
                         '').replace(' ;',
                                     '').replace("‘", '').replace("“", '').replace("“", '').replace("”", '').replace("x200b",
                                                                                                                     '').replace('"',
                                                                                                                                 '').replace('’', '')


def remove_punctuation(x: str):
    return x.replace('\'', '').translate(str.maketrans('', '', string.punctuation + '0123456789'))


funcs = [remove_emoji, remove_urls, remove_other_chars, lemmatize_words, remove_stop_words, remove_punctuation, remove_spaces]


def preprocess_text(doc: list):
    for fn in funcs:
        doc = fn(doc)
    return doc
