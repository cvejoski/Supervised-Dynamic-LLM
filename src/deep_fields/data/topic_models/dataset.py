import os
import pickle
from collections import namedtuple, defaultdict
from typing import Union

import numpy as np
from scipy.sparse import vstack
import torch

from torch.utils.data import Dataset
from transformers import BertTokenizer
from sklearn.preprocessing import StandardScaler

from deep_fields.models.generative_models.text.utils import Vocab, Time

PointL = namedtuple("Point", "text")
TextPoint = namedtuple("TextPoint", "x, y, seq_len")

PointTopicLanguageV = namedtuple("Point", "text, bow_h1, bow_h2, doc_id")
PointTopicLanguage = namedtuple("Point", "text, bow, doc_id")


class TopicDataset(Dataset):
    data: dict
    vocab: Vocab

    def __init__(
        self,
        path_to_data: str,
        ds_type: str,
        is_dynamic: bool,
        use_covar: bool,
        use_tmp_covariates: bool,
        normalize_data: Union[bool, StandardScaler],
        word_emb_type: str,
        tokenizer: BertTokenizer = None,
        reward_field: str = "reward",
    ):
        super(TopicDataset, self).__init__()
        assert os.path.exists(path_to_data)
        self.data = pickle.load(open(os.path.join(path_to_data, f"{ds_type}.pkl"), "rb"))
        self.ds_type = ds_type
        self.use_tmp_covariates = use_tmp_covariates
        self.normalize_data = normalize_data
        self.word_emb_type = word_emb_type
        self.reward_field = reward_field
        # if word_emb_type not in self.data.keys():
        #     raise ValueError(f"There is no `{word_emb_type}` word embedyint type in the dataset!")
        self.cov_stand = None
        if ds_type == "train" and self.normalize_data:
            self.cov_stand = StandardScaler()
            self.data["covariates"] = self.cov_stand.fit_transform(self.data["covariates"])
        else:
            if self.normalize_data:
                self.cov_stand = normalize_data
                self.data["covariates"] = self.cov_stand.transform(self.data["covariates"])

        has_covar = "covariates" in self.data.keys() and use_covar
        self.use_covar = use_covar
        self.tokenizer = tokenizer
        if is_dynamic:
            self.time = Time(**pickle.load(open(os.path.join(path_to_data, "time.pkl"), "rb")))
            self.corpus_per_time_period_avg = self.group_corpus_per_time_period(ds_type)
        if ds_type == "train":
            vocab = Vocab(**pickle.load(open(os.path.join(path_to_data, "vocabulary.pkl"), "rb")))
            self.vocab = vocab._replace(vectors=torch.from_numpy(vocab.vectors))
        if is_dynamic:
            if "reward_bin" in self.data.keys():
                self.reward_values = set(self.data["reward_bin"])
                self.reward_per_time_period_avg = self.__group_reward_per_time_period()
                if has_covar:
                    if ds_type != "test":
                        self.get_item = self.get_item_dynamic_reward_cov
                    else:
                        self.get_item = self.get_item_dynamic_validation_reward_cov
                else:
                    if ds_type != "test":
                        self.get_item = self.get_item_dynamic_reward
                    else:
                        self.get_item = self.get_item_dynamic_validation_reward
            else:
                if has_covar:
                    if ds_type != "test":
                        self.get_item = self.get_item_dynamic_cov
                    else:
                        self.get_item = self.get_item_dynamic_validation_cov
                else:
                    if ds_type != "test":
                        self.get_item = self.get_item_dynamic
                    else:
                        self.get_item = self.get_item_dynamic_validation

        else:
            if "reward_bin" in self.data.keys():
                self.reward_values = set(self.data["reward_bin"])
                if has_covar:
                    if ds_type != "test":
                        self.get_item = self.get_item_static_reward_cov
                    else:
                        self.get_item = self.get_item_static_validation_reward_cov
                else:
                    if ds_type != "test":
                        self.get_item = self.get_item_static_reward
                    else:
                        self.get_item = self.get_item_static_validation_reward
            else:
                if has_covar:
                    if ds_type != "test":
                        self.get_item = self.get_item_static_cov
                    else:
                        self.get_item = self.get_item_static_validation_cov
                else:
                    if ds_type != "test":
                        self.get_item = self.get_item_static
                    else:
                        self.get_item = self.get_item_static_validation

    def tokenize_text_transformer(self, i):
        if self.tokenizer is None:
            return 1.0
        text = self.data["text"][i]
        text_tok = self.tokenizer(text, return_tensors="pt")
        return text_tok

    def get_covar(self, i):
        covariates = self.data["covariates"][i][:-4]
        if self.use_tmp_covariates:
            covariates = self.data["covariates"][i]
        return covariates

    def append_reward(self, i, x):
        x["reward"] = self.data[self.reward_field][i]
        x["reward_bin"] = self.data["reward_bin"][i]
        return x

    def get_item_static(self, i):
        text_tok = self.tokenize_text_transformer(i)
        return {
            "text": text_tok,
            "seq": np.asarray(self.data["seq2seq"][0][i]),
            "seq_len": self.data["seq2seq"][1][i],
            "bow": self.data[self.word_emb_type][i].todense().view(np.ndarray).flatten().astype(np.float32),
        }

    def get_item_static_cov(self, i):
        x = self.get_item_static(i)
        cov = self.get_covar(i)
        x["covariates"] = cov
        return x

    def get_item_static_reward(self, i):
        x = self.get_item_static(i)
        x = self.append_reward(i, x)
        return x

    def get_item_static_reward_cov(self, i):
        x = self.get_item_static_cov(i)
        x = self.append_reward(i, x)
        return x

    def get_item_dynamic(self, i):
        x = self.get_item_static(i)
        x = self.append_dynamics(i, x)
        return x

    def append_dynamics(self, i, x):
        t = self.data["time"][i]
        x["time"] = t
        x["corpus"] = self.corpus_per_time_period_avg
        x["reward_proportion"] = self.reward_per_time_period_avg
        return x

    def get_item_dynamic_cov(self, i):
        x = self.get_item_static_cov(i)
        x = self.append_dynamics(i, x)
        return x

    def get_item_dynamic_reward(self, i):
        x = self.get_item_static_reward(i)
        x = self.append_dynamics(i, x)
        return x

    def get_item_dynamic_reward_cov(self, i):
        x = self.get_item_static_reward_cov(i)
        x = self.append_dynamics(i, x)
        return x

    def get_item_static_validation(self, i):
        x = self.get_item_static(i)
        x = self.append_validation(i, x)
        return x

    def append_validation(self, i, x):
        x[f"{self.word_emb_type}_h1"] = self.data[f"{self.word_emb_type}_h1"][i].todense().view(np.ndarray).flatten().astype(np.float32)
        x[f"{self.word_emb_type}_h2"] = self.data[f"{self.word_emb_type}_h2"][i].todense().view(np.ndarray).flatten().astype(np.float32)
        return x

    def get_item_static_validation_cov(self, i):
        x = self.get_item_static_cov(i)
        x = self.append_validation(i, x)
        return x

    def get_item_dynamic_validation(self, i):
        x = self.get_item_dynamic(i)
        x = self.append_validation(i, x)
        return x

    def get_item_dynamic_validation_cov(self, i):
        x = self.get_item_dynamic_cov(i)
        x = self.append_validation(i, x)
        return x

    def get_item_static_validation_reward(self, i):
        x = self.get_item_static_reward(i)
        x = self.append_validation(i, x)
        return x

    def get_item_static_validation_reward_cov(self, i):
        x = self.get_item_static_reward_cov(i)
        x = self.append_validation(i, x)
        return x

    def get_item_dynamic_validation_reward(self, i):
        x = self.get_item_dynamic_reward(i)
        x = self.append_validation(i, x)
        return x

    def get_item_dynamic_validation_reward_cov(self, i):
        x = self.get_item_dynamic_reward_cov(i)
        x = self.append_validation(i, x)
        return x

    def __getitem__(self, i):
        return self.get_item(i)

    def __len__(self):
        if self.ds_type == "test":
            return self.data[f"{self.word_emb_type}_h1"].shape[0]
        else:
            return self.data[self.word_emb_type].shape[0]

    def group_corpus_per_time_period(self, ds_type: str):
        key = self.word_emb_type
        if ds_type == "test":
            key = f"{self.word_emb_type}_h1"
        bow = self.data[key]
        corpus_per_period = defaultdict(list)
        for i, d in enumerate(self.data["time"]):
            corpus_per_period[d].append(bow[i])
        corpus_per_period_avg = [np.asarray(vstack(v).mean(0)) for _, v in sorted(corpus_per_period.items())]

        return np.vstack(corpus_per_period_avg)

    def get_one_hot(self, targets, nb_classes):
        targets = np.array(targets)
        res = np.eye(nb_classes)[targets.reshape(-1)]
        return res.reshape(list(targets.shape) + [nb_classes])

    def __group_reward_per_time_period(self):
        reward_per_year = defaultdict(list)
        rewards = list(map(int, self.data["reward_bin"]))
        rewards_one_hot = self.get_one_hot(rewards, self.number_of_rewards_categories())
        for i, d in enumerate(self.data["time"]):
            reward_per_year[d].append(rewards_one_hot[i])
        corpus_per_year_avg = [np.vstack(v).mean(0) for k, v in sorted(reward_per_year.items())]

        return np.asarray(corpus_per_year_avg, dtype=np.float)

    # def corpus_per_year(self, ds_type: str):
    #     key = self.word_emb_type
    #     if ds_type == 'test':
    #         key = f'{self.word_emb_type}_h1'
    #     bow = self.data[key].todense()
    #     corpus_per_year = defaultdict(list)
    #     doc_per_year = defaultdict(int)
    #     for i, d in enumerate(self.data['time']):
    #         corpus_per_year[d].append(bow[i].view(np.ndarray))
    #         doc_per_year[d] += 1
    #     corpus_per_year = [np.concatenate(v) for k, v in sorted(corpus_per_year.items())]
    #     return np.asarray(corpus_per_year, dtype=np.float)

    def discrete_reward(self):
        return True

    def number_of_rewards_categories(self):
        if "reward_bin" in self.data.keys():
            return len(self.reward_values)
        else:
            return None

    def rewards_categories(self):
        if "reward_bin" in self.data.keys():
            return self.reward_values
        else:
            return None

    def type_of_rewards(self):
        if "reward_bin" in self.data.keys():
            return "discrete"

    @property
    def covariates_size(self):
        if self.use_covar:
            if self.use_tmp_covariates:
                return len(self.data["covariates"][0])
            else:
                return len(self.data["covariates"][0][:-4])
        return 0
