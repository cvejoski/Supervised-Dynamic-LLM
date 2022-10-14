import os
from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from deep_fields import project_path
from deep_fields.data.topic_models.dataloaders import ADataLoader
from deep_fields.models.abstract_models import DeepBayesianModel
from deep_fields.models.basic_utils import create_instance
from deep_fields.models.deep_architectures.deep_nets import MLP
from deep_fields.models.deep_state_space.deep_state_space_recognition import \
    RecognitionModelFactory
from deep_fields.models.random_measures.random_measures_utils import \
    stick_breaking
from deep_fields.models.topic_models.topics_utils import reparameterize
from deep_fields.utils.loss_utils import (get_doc_freq, kullback_leibler_two_gaussians)
from torch import nn
from torch.distributions import Multinomial, Normal, Poisson
from tqdm import tqdm

cos = nn.CosineSimilarity(dim=1, eps=1e-6)


class DynamicLDA(DeepBayesianModel):
    topic_recognition_model: nn.Module
    eta_q: nn.Module
    theta_q: nn.Module

    mu_q_alpha: nn.Parameter
    logvar_q_alpha: nn.Parameter

    vocabulary_dim: int
    number_of_documents: int
    num_training_steps: int
    num_prediction_steps: int
    lambda_diversity: float = 0.1
    number_of_topics: int
    delta: float = 0.005

    topic_transition_dim: int

    def __init__(self, model_dir=None, data_loader=None, model_name=None, evaluate_mode=False, **kwargs):
        model_name = "dynamic_lda" if model_name is None else model_name
        DeepBayesianModel.__init__(self, model_name, model_dir=model_dir, data_loader=data_loader, evaluate_mode=evaluate_mode, **kwargs)

    @classmethod
    def get_parameters(cls):
        """
        here we provide an example of the minimum set of parameters requiered to instantiate the model
        """
        number_of_topics = 50

        r_hidden_transition_state_size = 32
        vocabulary_dim = 100
        parameters_sample = {
            "number_of_topics": number_of_topics,  # z
            "number_of_documents": 100,
            "number_of_words_per_document": 30,
            "vocabulary_dim": vocabulary_dim,
            "num_training_steps": 48,
            "lambda_diversity": 0.1,
            "num_prediction_steps": 2,
            "topic_proportion_transformation": "gaussian_softmax",
            "topic_lifetime_tranformation": "gaussian_softmax",
            "delta": 0.005,
            "theta_q_type": "q-INDEP",
            "theta_q_parameters": {
                "observable_dim": vocabulary_dim,
                "layers_dim": [250, 250],
                "output_dim": 250,
                "hidden_state_dim": number_of_topics,
                "dropout": .1,
                "out_dropout": 0.1
            },
            "eta_q_type": "q-RNN",
            "eta_q_parameters": {
                "observable_dim": vocabulary_dim,
                "layers_dim": 400,
                "num_rnn_layers": 4,
                "hidden_state_dim": r_hidden_transition_state_size,
                "hidden_state_transition_dim": 400,
                "dropout": .1,
                "out_dropout": 0.1
            },
            "model_path": os.path.join(project_path, 'results')
        }

        return parameters_sample

    def set_parameters(self, **kwargs):
        self.vocabulary_dim = kwargs.get("vocabulary_dim")
        self.number_of_documents = kwargs.get("number_of_documents", None)
        self.num_training_steps = kwargs.get("num_training_steps", None)
        self.num_prediction_steps = kwargs.get("num_prediction_steps", None)

        self.number_of_topics = kwargs.get("number_of_topics", 10)

        self.topic_transition_dim = self.number_of_topics * self.vocabulary_dim  # output text pad ad eos

        self.eta_q_type = kwargs.get("eta_q_type")
        self.eta_q_parameters = kwargs.get("eta_q_parameters")
        self.eta_dim = kwargs.get("eta_q_parameters").het("hidden_state_dim")

        # TEXT
        self.theta_q_type = kwargs.get("theta_q_type")
        self.theta_q_parameters = kwargs.get("theta_q_parameters")

        self.topic_proportion_transformation = kwargs.get("topic_proportion_transformation")

    def update_parameters(self, data_loader, **kwargs):
        kwargs.update({"vocabulary_dim": data_loader.vocabulary_dim})
        kwargs.update({"number_of_documents": data_loader.number_of_documents})
        kwargs.update({"num_training_steps": data_loader.num_training_steps})
        kwargs.update({"num_prediction_steps": data_loader.num_prediction_steps})

        kwargs.get("eta_q_parameters").update({"observable_dim": data_loader.vocabulary_dim, "hidden_state_dim": self.number_of_topics})
        kwargs.get("theta_q_parameters").update({
            "observable_dim": data_loader.vocabulary_dim,
            "hidden_state_dim": self.number_of_topics,
            "control_variable_dim": self.number_of_topics
        })

        return kwargs

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()
        inference_parameters.update({"regularizers": {"nll": None}})
        inference_parameters.update({"model_eval": "PPL-Blei"})
        inference_parameters.update({"gumbel": .0005})

        return inference_parameters

    def initialize_inference(self, data_loader, **inference_parameters):
        super().initialize_inference(data_loader=data_loader, **inference_parameters)
        regularizers = inference_parameters.get("regularizers")

        self.schedulers = {}
        for k, v in regularizers.items():
            if v is not None:
                self.schedulers[k] = create_instance(v)

        self.eta_q.device = self.device
        self.theta_q.device = self.device

    def move_to_device(self):
        self.eta_q.device = self.device

    def define_deep_models(self):
        recognition_factory = RecognitionModelFactory()

        self.eta_q = recognition_factory.create(self.eta_q_type, **self.eta_q_parameters)
        self.theta_q = recognition_factory.create(self.theta_q_type, **self.theta_q_parameters)

        self.mu_q_alpha = nn.Parameter(torch.randn(self.number_of_topics, self.num_training_steps, self.vocabulary_dim))
        self.logvar_q_alpha = nn.Parameter(torch.randn(self.number_of_topics, self.num_training_steps, self.vocabulary_dim))

    def forward(self, x):
        """
        parameters
        ----------
        batchdata ()
        returns
        -------
        z (batch_size*sequence_lenght,self.hidden_state_dim)
        recognition_parameters = (z_mean,z_var)
        (batch_size*sequence_lenght,self.hidden_state_dim)
        likelihood_parameters = (likelihood_mean,likelihood_variance)
        """

        time_idx = x['time'].squeeze().long().to(self.device, non_blocking=True)
        corpora = x['corpus'][:1].float().to(self.device, non_blocking=True)
        if 'bow' in x:
            bow = x['bow'].to(self.device, non_blocking=True)
            normalized_bow = bow.float() / bow.sum(1, True)
        else:
            bow = x['bow_h1'].to(self.device, non_blocking=True)
            normalized_bow = bow.float() / bow.sum(1, True)
            bow = x['bow_h2'].to(self.device, non_blocking=True)

        # Dynamic Stuff

        alpha, kl_alpha = self.q_alpha()  # [total_time, topic_transition_dim]
        eta, eta_params, kl_eta = self.eta_q(corpora)  # [total_time, topic_transition_dim]
        beta = self.get_beta(alpha)
        beta_per_document = beta[time_idx]  # [batch_size, topic_transition_dim]
        eta_per_document = eta[time_idx]  # [batch_size, number_of_topics]

        # Text Recognition
        theta_logits, _, kl_theta = self.theta_q(normalized_bow, eta_per_document)

        theta = self.proportion_transformation(theta_logits)
        nll = self.nll(theta, bow, beta_per_document)

        return nll, kl_theta, kl_alpha, kl_eta, eta_params, theta

    def loss(self, x, forward_results, data_set, epoch):
        """
        nll [batch_size, max_lenght]

        """
        if 'bow' in x:
            text_size = x['bow'].sum(1).double().to(self.device, non_blocking=True)
        else:
            text_size = x['bow_h2'].sum(1).double().to(self.device, non_blocking=True)

        nll = forward_results["nll"]
        kl_theta = forward_results["kl_theta"]
        kl_alpha = forward_results["kl_alpha"]
        kl_eta = forward_results["kl_eta"]

        coeff = 1.0
        if self.training:
            coeff = len(data_set)
        loss = (nll.sum() + kl_theta) * coeff + (kl_eta + kl_alpha)  # [number of batches]

        log_perplexity = nll / text_size.float()
        log_perplexity = torch.mean(log_perplexity)
        perplexity = torch.exp(log_perplexity)

        return {
            "loss": loss,
            "NLL-Loss": nll.sum() * coeff,
            "KL-Loss-Eta": kl_eta,
            "KL-Loss-Alpha": kl_alpha,
            "KL-Loss-Theta": kl_theta * coeff,
            "PPL-Blei": perplexity,
            "PPL": torch.exp(nll.sum() / text_size.sum().float()),
            "Log-Likelihood": log_perplexity
        }

    def q_alpha(self):
        rho_dim = self.mu_q_alpha.size(-1)
        alphas = torch.zeros(self.num_training_steps, self.number_of_topics, rho_dim).to(self.device)
        alphas[0] = reparameterize(self.mu_q_alpha[:, 0, :], self.logvar_q_alpha[:, 0, :])

        mu_0_p = torch.zeros(self.number_of_topics, rho_dim).to(self.device, non_blocking=True)
        logvar_0_p = torch.zeros(self.number_of_topics, rho_dim).to(self.device, non_blocking=True)
        logvar_t_p = torch.log(self.delta * torch.ones(self.number_of_topics, rho_dim).to(self.device, non_blocking=True))
        kl = kullback_leibler_two_gaussians(self.mu_q_alpha[:, 0, :], self.logvar_q_alpha[:, 0, :], mu_0_p, logvar_0_p, 'sum')

        for t in range(1, self.num_training_steps):
            alphas[t] = reparameterize(self.mu_q_alpha[:, t, :], self.logvar_q_alpha[:, t, :])
            mu_t_p = alphas[t - 1]
            kl += kullback_leibler_two_gaussians(self.mu_q_alpha[:, t, :], self.logvar_q_alpha[:, t, :], mu_t_p, logvar_t_p, 'sum')

        return alphas, kl

    def get_beta(self, alphas):
        topic_embeddings = alphas.view(self.num_training_steps, self.number_of_topics, self.vocabulary_dim)
        beta = torch.softmax(topic_embeddings, dim=2)
        return beta

    def get_beta_eval(self):
        return self.get_beta(self.mu_q_alpha)

    def nll(self, theta, bow, beta):
        if self.topic_embeddings != "static" and beta.dim() == 3:
            loglik = torch.bmm(theta.unsqueeze(1), beta).squeeze(1)
        else:
            loglik = torch.matmul(theta, beta)
        if self.training:
            loglik = torch.log(loglik + 1e-6)
        else:
            loglik = torch.log(loglik)
        nll = -loglik * bow

        return nll.sum(-1)

    def metrics(self, data, forward_results, epoch, mode="evaluation", data_loader=None):
        if mode == "validation_global":
            if epoch % self.metrics_logs == 0:
                top_word_per_topic = self.top_words(data_loader, num_of_top_words=20)
                TEXT = ""
                for topic, value in top_word_per_topic.items():
                    for time, words in value.items():
                        TEXT += f'{topic} --- {time}: {" ".join(words)}\n\n'
                    TEXT += "*" * 1000 + "\n"
                # TEXT = "\n".join(["TOPIC {0}: ".format(j) + " ".join(top_word_per_topic[j]) + "\n" for j in range(len(top_word_per_topic))])
                self.writer.add_text("/TEXT/", TEXT, epoch)
        return {}

    def proportion_transformation(self, proportions_logits):
        if self.topic_proportion_transformation == "gaussian_softmax":
            # theta = self.hidden_state_to_topics(proportions_logits)
            proportions = torch.softmax(proportions_logits, dim=1)
        elif self.topic_proportion_transformation == "gaussian_stick_break":
            sticks = torch.sigmoid(proportions_logits)
            proportions = stick_breaking(sticks, self.device)
        else:
            raise ValueError(f"{self.topic_proportion_transformation} transformation not implemented!")
        return proportions

    def sample(self):
        """
        returns
        ------
        (documents,documents_z,thetas,phis)
        """
        word_count_distribution = Poisson(self.number_of_words_per_document)
        prior_0 = Normal(torch.zeros(self.number_of_topics), torch.ones(self.number_of_topics))

        word_embeddings = nn.Embedding(self.vocabulary_size, self.word_embeddings_dim)
        topic_embeddings = nn.Embedding(self.number_of_topics, self.word_embeddings_dim)
        beta = torch.matmul(word_embeddings.weight, topic_embeddings.weight.T)
        beta = torch.softmax(beta, dim=0)

        if self.topic_proportion_transformation == "gaussian_softmax":
            proportions_logits = prior_0.sample(torch.Size([self.number_of_documents]))
            topic_proportions = torch.softmax(proportions_logits, dim=1)
        elif self.topic_proportion_transformation == "gaussian_stick_break":
            proportions_logits = prior_0.sample(torch.Size([self.number_of_documents]))
            sticks = torch.sigmoid(proportions_logits)
            proportions = stick_breaking(sticks)

        # LDA Sampling
        documents_z = []
        documents = []
        word_count = word_count_distribution.sample(torch.Size([self.number_of_documents]))
        words_distributions = []
        for k in range(self.number_of_topics):
            words_distributions.append(Multinomial(total_count=1, probs=beta.T[k]))
        # LDA Sampling
        for d_index in range(self.number_of_documents):
            mixing_distribution_document = Multinomial(total_count=1, probs=proportions[d_index])
            # sample word allocation
            number_of_words = int(word_count[d_index].item())
            z_document = mixing_distribution_document.sample(sample_shape=torch.Size([number_of_words]))
            selected_index = torch.argmax(z_document, dim=1)
            documents_z.append(selected_index.numpy())
            document = []
            for w_index in range(number_of_words):
                z_word = words_distributions[selected_index[w_index]].sample()
                document.append(torch.argmax(z_word).item())
            documents.append(document)

        return documents, documents_z

    def get_words_by_importance(self) -> torch.Tensor:
        with torch.no_grad():
            beta = self.get_beta(self.mu_q_alpha)
            important_words = torch.argsort(beta, dim=-1, descending=True)
            return important_words  # torch.transpose(important_words, 1, 0)

    # ======================================================
    # POST - PROCESSING
    # ======================================================
    def top_words(self, data_loader, num_of_top_words=20, num_of_time_steps=3):
        important_words = self.get_words_by_importance()
        # from index to words
        vocabulary = data_loader.vocab
        top_word_per_topic = dict()

        tt = important_words.size(1) // 2
        for k in range(self.number_of_topics):
            top_words_per_time = dict()
            for t in [0, tt, important_words.size(0) - 1]:
                topic_words = [vocabulary.itos[w] for w in important_words[t, k, :num_of_top_words].tolist()]
                top_words_per_time[f'Time {t}'] = topic_words
            top_word_per_topic[f'Topic {k}'] = top_words_per_time
        return top_word_per_topic

    def top_topics(self, data_loader):
        with torch.no_grad():
            # most important topics
            number_of_docs = 0.
            proportions = torch.zeros(self.number_of_topics, device=self.device)
            for databatch in data_loader.validation():
                likelihood, theta, theta_parameters, \
                    v_per_document, \
                    v_parameters, r_parameters, \
                    v_parameters_0, r_parameters_0, \
                    v_priors_parameters, r_priors_parameters = self(databatch)
                proportions += theta.sum(dim=0)
                batch_size, _ = theta.shape
                number_of_docs += batch_size

            proportions = proportions / number_of_docs
            proportions_index = torch.argsort(proportions, descending=True)
            proportions_values = proportions[proportions_index].cpu().detach().numpy()
            top_proportions = list(proportions_index.cpu().detach().numpy())
            return top_proportions, proportions_values

        # ===================================================================================
        # PREDICTION POST PROCESSING
        # ===================================================================================

    def words_time_series(self, dataloader):
        return None

    def topics_timeseries(self, dataloader):
        return None

    def get_prior_transitions_distributions(self, v_sample, r_sample):
        """
        returns torch.distributions Normal([montecarlo_samples,number_of_topics])
        """
        v_mean = v_sample  # [total_time-1, number_of_topics]
        v_std = (self.topic_transition_logvar**2.) * torch.ones_like(v_mean)  # [total_time-1, number_of_topics]
        v_priors_parameters = v_mean, v_std

        r_mean = r_sample
        r_std = (self.r_transition_logvar**2.) * torch.ones_like(r_sample)  # [total_time-1, number_of_topics]
        r_priors_parameters = r_mean, r_std

        transition_prior_v = Normal(*v_priors_parameters)
        transition_prior_r = Normal(*r_priors_parameters)

        return transition_prior_v, transition_prior_r

    def prediction_montecarlo_step(self, current_document_count, alpha_dist, eta_dist):
        alpha_sample = alpha_dist.sample()  # [mc, n_topics, topic_transition_dim]
        eta_sample = eta_dist.sample()  # [mc,n_topics, topic_transition_dim]
        n_mc, _, _ = alpha_sample.shape

        theta_dist = Normal(eta_sample, torch.ones_like(eta_sample))
        theta_sample = theta_dist.sample(sample_shape=torch.Size([current_document_count]))  # [mc,current_doc_count,number_of_topics]
        # [mc,count,number_of_topics]
        # alpha_per_doc = torch.repeat_interleave(alpha_sample, current_document_count, dim=0)
        # alpha_per_doc = alpha_per_doc.view(n_mc, current_document_count, self.number_of_topics, -1).contiguous()  # [mc,current_document_count,
        # number_of_topics]

        theta_sample = theta_sample.view(n_mc * current_document_count, -1).contiguous()
        theta = self.proportion_transformation(theta_sample)  # [mc,count,number_of_topics]
        theta = theta.view(n_mc, current_document_count, -1).contiguous()
        return theta, alpha_sample, eta_sample, alpha_sample

    def get_transitions_dist(self, alpha_sample, eta_sample):
        """
        returns torch.distributions Normal([montecarlo_samples,number_of_topics])
        """
        alpha_std = torch.ones_like(alpha_sample) * self.delta
        eta_std = torch.ones_like(eta_sample) * self.delta

        v_dist = Normal(alpha_sample, alpha_std)
        r_dist = Normal(eta_sample, eta_std)
        return v_dist, r_dist

    def prediction(self, data_loader: ADataLoader, montecarlo_samples=10):
        self.eval()
        x = next(iter(data_loader.train))
        with torch.no_grad():
            forward_results = self.forward(x)

            _, _, _, _, eta_params, _ = forward_results

            last_state_alpha = self.mu_q_alpha[:, -1, :], torch.exp(0.5 * self.logvar_q_alpha[:, -1, :]) * self.delta
            alpha_q_dist = Normal(*last_state_alpha)
            eta_std = torch.exp(0.5 * eta_params[1]) * self.delta
            last_state_eta = eta_params[0], eta_std
            eta_q_dist = Normal(*last_state_eta)

            alpha_sample = alpha_q_dist.sample(sample_shape=torch.Size([montecarlo_samples]))
            eta_sample = eta_q_dist.sample(sample_shape=torch.Size([montecarlo_samples]))
            pred_pp_all = []
            for tt in self.data_loader.prediction_times:
                current_document_count = int(data_loader.prediction_count_per_year)

                alpha_dist, eta_dist = self.get_transitions_dist(alpha_sample, eta_sample)
                theta, alpha_sample, eta_sample, alpha_doc = self.prediction_montecarlo_step(current_document_count, alpha_dist, eta_dist)
                # [mc,count,number_of_topics]
                pred_pp_step = torch.zeros((montecarlo_samples, current_document_count))
                bow = torch.from_numpy(data_loader.predict.dataset.corpus_per_year('prediction')[0], device=self.device)
                text_size = bow.sum(1).double().to(self.device, non_blocking=True)
                for mc_index in range(montecarlo_samples):
                    alpha_per_doc = alpha_sample[mc_index]
                    theta_mc = theta[mc_index]  # [count, number_of_topics]
                    beta = torch.softmax(alpha_per_doc, dim=-1)
                    pred_like = self.nll_test(theta_mc, bow, beta)
                    log_pp = (1. / text_size.float()) * pred_like
                    log_pp = torch.mean(log_pp)

                    pred_pp_step[mc_index] = log_pp
                pred_pp_all.append(pred_pp_step.mean())
            pred_pp = torch.mean(torch.stack(pred_pp_all)).item()
            return pred_pp

    def topic_diversity(self, topk: int = 25) -> float:
        important_words_per_topic = self.get_words_by_importance()
        time_steps = important_words_per_topic.size(1)
        td_all = torch.zeros((time_steps,))
        for tt in range(time_steps):
            list_w = important_words_per_topic[:, tt, :topk]
            n_unique = len(torch.unique(list_w))
            td = n_unique / (topk * self.number_of_topics)
            td_all[tt] = td
        return td_all.mean().item()

    def topic_coherence(self, data: np.ndarray) -> float:
        top_10 = self.get_words_by_importance()[:, :, :10]
        top_10 = torch.transpose(top_10, 1, 0)
        tc_all = []
        data = data > 0
        p_bar = tqdm(range(top_10.size(0)), desc="Calculating Topics Coherence:")
        for tt in p_bar:
            tc = self._topic_coherence(data, top_10[tt])
            tc_all.append(tc)
            p_bar.set_postfix_str(f'current topic coherence: {np.mean(tc_all)}')
        return np.mean(tc_all)

    def _topic_coherence(self, data: np.ndarray, top_10: torch.Tensor) -> np.ndarray:
        D = data.shape[0]
        TC = []
        # p_bar = tqdm(desc='calculating topics coherence', total=self.number_of_topics)
        for k in range(self.number_of_topics):
            top_10_k = top_10[k].tolist()
            TC_k = 0
            counter = 0
            word_count_ = self.data_loader.vocab.word_count
            for i, word in enumerate(top_10_k):
                # get D(w_i)
                D_wi = word_count_[word]
                j = i + 1
                tmp = 0
                while len(top_10_k) > j > i:
                    # get D(w_j) and D(w_i, w_j)
                    D_wj = word_count_[top_10_k[j]]
                    D_wi_wj = get_doc_freq(data, word, top_10_k[j])
                    # get f(w_i, w_j)
                    if D_wi_wj == 0:
                        f_wi_wj = -1
                    else:
                        f_wi_wj = -1 + (np.log(D_wi) + np.log(D_wj) - 2.0 * np.log(D)) / (np.log(D_wi_wj) - np.log(D))
                    # update tmp:
                    tmp += f_wi_wj
                    j += 1
                    counter += 1
                # update TC_k
                TC_k += tmp
            TC.append(TC_k / counter)
            # p_bar.set_postfix_str(f"TC: {np.mean(TC)}")
            # p_bar.update()
        TC = np.mean(TC)
        # print('Topic coherence is: {}'.format(TC))
        return TC

    @property
    def is_static_topic_embeddings(self):
        return self.topic_embeddings == 'static'


class DynamicTopicEmbeddings(DynamicLDA):
    """
    here we follow (overleaf link)

    https://www.overleaf.com/7689487332skzdrpbgmcdm

    """
    word_embeddings_dim: int
    rho: nn.Parameter
    name_ = "neural_dynamical_topic_embeddings"

    def __init__(self, model_dir=None, data_loader=None, model_name=None, evaluate_mode=False, **kwargs):
        model_name = "neural_dynamical_topic_embeddings" if model_name is None else model_name
        DynamicLDA.__init__(self, model_name=model_name, model_dir=model_dir, data_loader=data_loader, evaluate_mode=evaluate_mode, **kwargs)

    @classmethod
    def get_parameters(cls):
        """
        here we provide an example of the minimum set of parameters requiered to instantiate the model
        """
        number_of_topics = 50

        vocabulary_dim = 67
        parameters_sample = {
            "number_of_topics": number_of_topics,  # z
            "number_of_documents": 100,
            "number_of_words_per_document": 30,
            "vocabulary_dim": vocabulary_dim,
            "topic_embeddings": "dynamic",  # static
            "nonlinear_transition_prior": False,
            "num_training_steps": 48,
            "num_prediction_steps": 2,
            "train_word_embeddings": False,
            "topic_proportion_transformation": "gaussian_softmax",
            "topic_lifetime_tranformation": "gaussian_softmax",
            "delta": 0.005,
            "theta_q_type": "q-INDEP",
            "theta_q_parameters": {
                "observable_dim": vocabulary_dim,
                "layers_dim": [250, 250],
                "output_dim": 250,
                "hidden_state_dim": 32,
                "dropout": .1,
                "out_dropout": 0.1
            },
            "eta_q_type": "q-RNN",
            "eta_q_parameters": {
                "observable_dim": vocabulary_dim,
                "layers_dim": 400,
                "num_rnn_layers": 4,
                "hidden_state_dim": 32,
                "hidden_state_transition_dim": 400,
                "dropout": .1,
                "out_dropout": 0.0
            },
            "eta_prior_transition": {
                "layers_dim": [64, 64],
                "output_transformation": None,
                "dropout": 0.0
            },
            "model_path": os.path.join(project_path, 'results')
        }

        return parameters_sample

    def set_parameters(self, **kwargs):
        self.vocabulary_dim = kwargs.get("vocabulary_dim")
        self.number_of_documents = kwargs.get("number_of_documents", None)
        self.num_training_steps = kwargs.get("num_training_steps", None)
        self.num_prediction_steps = kwargs.get("num_prediction_steps", None)

        self.word_embeddings_dim = kwargs.get("word_embeddings_dim", 100)
        self.number_of_topics = kwargs.get("number_of_topics", 10)

        self.nonlinear_transition = kwargs.get("nonlinear_transition_prior")
        self.eta_prior_transition_params = kwargs.get("eta_prior_transition")

        self.train_word_embeddings = kwargs.get("train_word_embeddings")
        self.topic_transition_dim = self.number_of_topics * self.word_embeddings_dim

        self.eta_q_type = kwargs.get("eta_q_type")
        self.eta_q_parameters = kwargs.get("eta_q_parameters")
        self.eta_dim = kwargs.get("eta_q_parameters").get("hidden_state_dim")
        self.topic_embeddings = kwargs.get("topic_embeddings")
        # TEXT
        self.theta_q_type = kwargs.get("theta_q_type")
        self.theta_q_parameters = kwargs.get("theta_q_parameters")
        self.theta_emb_dim = self.theta_q_parameters.get("hidden_state_dim")
        self.topic_proportion_transformation = kwargs.get("topic_proportion_transformation")

    def update_parameters(self, data_loader, **kwargs):
        kwargs.update({"vocabulary_dim": data_loader.vocabulary_dim})
        kwargs.update({"number_of_documents": data_loader.number_of_documents})
        kwargs.update({"num_training_steps": data_loader.num_training_steps})
        kwargs.update({"num_prediction_steps": data_loader.num_prediction_steps})
        kwargs.get("eta_q_parameters").update({"observable_dim": data_loader.vocabulary_dim, "hidden_state_dim": self.number_of_topics})
        kwargs.get("theta_q_parameters").update({"observable_dim": data_loader.vocabulary_dim, "control_variable_dim": self.number_of_topics})
        kwargs.get("eta_prior_transition").update({"input_dim": self.number_of_topics, "output_dim": self.number_of_topics})

        return kwargs

    def define_deep_models(self):
        self.theta_stats = defaultdict(list)
        recognition_factory = RecognitionModelFactory()
        rho = nn.Embedding(self.vocabulary_dim, self.word_embeddings_dim)
        rho.weight.data = self.data_loader.vocab.vectors[:self.vocabulary_dim]
        self.rho = nn.Parameter(rho.weight.data.clone().float(), requires_grad=self.train_word_embeddings)
        self.eta_q = recognition_factory.create(self.eta_q_type, **self.eta_q_parameters)
        self.theta_q = recognition_factory.create(self.theta_q_type, **self.theta_q_parameters)

        if self.topic_embeddings == "dynamic":
            self.mu_q_alpha = nn.Parameter(torch.randn(self.number_of_topics, self.num_training_steps, self.word_embeddings_dim))
            self.logvar_q_alpha = nn.Parameter(torch.randn(self.number_of_topics, self.num_training_steps, self.word_embeddings_dim))
        elif self.topic_embeddings == "static":
            self.q_alpha = self.alpha_static_q
            self.mu_q_alpha = nn.Parameter(torch.randn(self.number_of_topics, self.word_embeddings_dim))
            self.div_reg = self.diversity_regularizer

        self.hidden_state_to_topics = nn.Linear(self.theta_emb_dim, self.number_of_topics)
        self.eta_transform = nn.Linear(self.number_of_topics, self.theta_emb_dim)

        self.p_eta_m = None
        if self.nonlinear_transition:
            self.p_eta_m = MLP(**self.eta_prior_transition_params)

    def forward(self, x):
        """
        parameters
        ----------
        batchdata ()
        returns
        -------
        z (batch_size*sequence_lenght,self.hidden_state_dim)
        recognition_parameters = (z_mean,z_var)
        (batch_size*sequence_lenght,self.hidden_state_dim)
        likelihood_parameters = (likelihood_mean,likelihood_variance)
        """

        time_idx = x['time'].squeeze().long().to(self.device, non_blocking=True)
        corpora = x['corpus'][:1].float().to(self.device, non_blocking=True)
        if 'bow' in x:
            bow = x['bow'].to(self.device, non_blocking=True)
            normalized_bow = bow.float() / bow.sum(1, True)
        else:
            bow = x['bow_h1'].to(self.device, non_blocking=True)
            normalized_bow = bow.float() / bow.sum(1, True)
            bow = x['bow_h2'].to(self.device, non_blocking=True)

        # Dynamic Stuff

        alpha, kl_alpha = self.q_alpha()  # [total_time, topic_transition_dim]
        eta, _, kl_eta = self.eta_q(corpora, self.p_eta_m)  # [total_time, topic_transition_dim]
        beta = self.get_beta(alpha)

        if self.topic_embeddings == 'dynamic':
            alpha = alpha[time_idx]  # [batch_size, topic_transition_dim]
            beta = beta[time_idx]

        eta_per_document = eta[time_idx]  # [batch_size, number_of_topics]

        # Text Recognition
        theta_logits, _, kl_theta = self.theta_q(normalized_bow, eta_per_document, self.eta_transform)

        theta = self.proportion_transformation(theta_logits)
        nll = self.nll(theta, bow, beta)

        return {"nll": nll, "kl_theta": kl_theta, "kl_alpha": kl_alpha, "kl_eta": kl_eta, "eta": eta[-1], "theta": theta}

    def proportion_transformation(self, proportions_logits):
        if self.topic_proportion_transformation == "gaussian_softmax":
            theta = self.hidden_state_to_topics(proportions_logits)
            proportions = torch.softmax(theta, dim=1)
        elif self.topic_proportion_transformation == "gaussian_stick_break":
            sticks = torch.sigmoid(proportions_logits)
            proportions = stick_breaking(sticks, self.device)
        else:
            raise ValueError(f"{self.topic_proportion_transformation} transformation not implemented!")
        return proportions

    def alpha_static_q(self):
        return self.mu_q_alpha, torch.tensor(0.0, device=self.device)

    def diversity_regularizer(self, x):
        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-12)
        cosine_simi = torch.tensordot(x, x, dims=[[1], [1]]).abs()
        angles = torch.acos(torch.clamp(cosine_simi, -1. + 1e-7, 1. - 1e-7))
        angles_mean = angles.mean()
        var = ((angles - angles_mean)**2).mean()
        return angles_mean - var

    def get_beta(self, alphas):
        beta = torch.matmul(alphas, self.rho.T)
        beta = torch.softmax(beta, dim=-1)
        return beta

    def get_beta_eval(self, n_samples=None):
        if n_samples is None:
            return self.get_beta(self.mu_q_alpha)
        else:
            samples = []
            for _ in range(n_samples):
                samples.append(self.get_beta(reparameterize(self.mu_q_alpha, self.logvar_q_alpha)))
            return torch.stack(samples, dim=0)

    def metrics(self, data, forward_results, epoch, mode="evaluation", data_loader=None):
        # if mode == "validation" and epoch % self.metrics_logs == 0:
        #     theta = forward_results["theta"]
        #     time_idx = data["time"].squeeze()

        #     for id, t in zip(time_idx, theta):
        #         self.theta_stats[id.item()].append(t)
        if mode == "validation_global" and epoch % self.metrics_logs == 0:
            # for key, value in self.theta_stats.items():
            #     self.theta_stats[key] = torch.stack(value).mean(0)

            # self.theta_stats = OrderedDict(sorted(self.theta_stats.items()))
            # theta_stats = torch.stack(list(self.theta_stats.values())).detach().cpu().numpy()

            # self.plot_topics_ts(theta_stats, '/Topic-Time-Series')
            # self.theta_stats.clear()
            # self.theta_stats = defaultdict(list)

            if self.is_dynamic_topc_embeddings:
                self._log_top_words_dynamic(data_loader)
            else:
                self._log_top_words(data_loader)
        return {}

    def plot_topics_ts(self, theta, label, y_lim=None):
        ntime, ntopics = theta.shape
        x = range(ntime)
        ncols = 5
        nrows = ntopics // ncols + ntopics % ncols
        fig, axis = plt.subplots(nrows, ncols, sharex=True, figsize=(15, nrows * 2))
        axis = axis.flatten()
        for i in range(ntopics):
            ax = axis[i]
            color = 'tab:red'
            ax.set_ylabel('theta', color=color)
            ax.tick_params(axis='y', labelcolor=color)
            ax.plot(x, theta[:, i], color=color)
            # sns.lineplot(x=x, y=theta[:, i], ax=ax, color=color)
        fig.tight_layout()
        self.writer.add_figure(label, fig, self.number_of_iterations)
        self.writer.flush()
        plt.close()
        del fig
        del axis

    # ======================================================
    # POST - PROCESSING
    # ======================================================
    def top_words(self, data_loader, num_of_top_words=20):
        important_words = self.get_words_by_importance()
        # from index to words
        vocabulary = data_loader.vocab
        top_word_per_topic = dict()

        for k in range(self.number_of_topics):
            topic_words = [vocabulary.itos[w] for w in important_words[k, :num_of_top_words].tolist()]
            top_word_per_topic[f'TOPIC {k}'] = topic_words
        return top_word_per_topic, important_words

    def top_words_dynamic(self, data_loader, num_of_top_words=20):
        important_words = self.get_words_by_importance()
        # from index to words
        vocabulary = data_loader.vocab
        top_word_per_topic = dict()

        tt = important_words.size(1) // 2
        for k in range(self.number_of_topics):
            top_words_per_time = dict()
            for t in range(0, important_words.size(1), 2):
                topic_words = [vocabulary.itos[w] for w in important_words[k, t, :num_of_top_words].tolist()]
                top_words_per_time[f'TIME {t}'] = topic_words
            top_word_per_topic[f'TOPIC {k}'] = top_words_per_time
        return top_word_per_topic, important_words

    def _log_top_words(self, data_loader):
        top_word_per_topic = self.top_words(data_loader, num_of_top_words=20)[0]
        TEXT = "\n".join(["<strong>{0}</strong> ".format(topic) + " ".join(words) + "\n" for topic, words in top_word_per_topic.items()])
        self.writer.add_text("/Topics-Top-Words", TEXT, self.number_of_iterations)

    def _log_top_words_dynamic(self, data_loader):
        top_word_per_topic = self.top_words_dynamic(data_loader, num_of_top_words=20)[0]
        TEXT = ""
        for topic, value in top_word_per_topic.items():
            for time, words in value.items():
                TEXT += f'<strong>{topic} --- {time}</strong>: {" ".join(words)}\n\n'
            TEXT += "*" * 1000 + "\n"
        self.writer.add_text("/Topics-Top-Words", TEXT, self.number_of_iterations)

    def __sample_eta(self, last_eta: torch.Tensor, montecarlo_samples: int = None) -> torch.Tensor:
        eta_m = last_eta if self.p_eta_m is None else self.p_eta_m(last_eta)
        eta_std = self.delta * torch.ones_like(last_eta)
        eta_p_dist = Normal(eta_m, eta_std)
        if montecarlo_samples is None:
            return eta_p_dist.sample()
        else:
            return eta_p_dist.sample(sample_shape=torch.Size([montecarlo_samples]))

    def __sample_alpha(self, alpha_m: torch.Tensor, montecarlo_samples: int = None) -> torch.Tensor:
        if self.is_dynamic_topc_embeddings:
            alpha_std = self.delta * torch.ones_like(alpha_m)
            alpha_p_dist = Normal(alpha_m, alpha_std)
            if montecarlo_samples is None:
                return alpha_p_dist.sample()
            else:
                return alpha_p_dist.sample(sample_shape=torch.Size([montecarlo_samples]))
        else:
            if montecarlo_samples is None:
                return alpha_m
            else:
                return self.mu_q_alpha.repeat(montecarlo_samples, 1, 1)

    def _sample_global_variables(self, train_ds, n_ts: int, n_ps: int, montecarlo_samples: int) -> dict:
        global_variables = defaultdict(dict)
        x = next(iter(train_ds))
        pbar = tqdm(range(n_ts + 1, n_ts + n_ps), total=n_ps, desc='Sampling Global Variables')
        eta = self.forward(x)["eta"]

        etas = self.__sample_eta(eta, montecarlo_samples)
        global_variables[n_ts]['eta'] = etas

        alpha, _ = self.q_alpha()

        alphas = self.__sample_alpha(alpha[-1], montecarlo_samples)
        global_variables[n_ts]['alpha'] = alphas
        pbar.update()
        for i in pbar:
            etas = self.__sample_eta(etas)
            global_variables[i]['eta'] = etas

            alphas = self.__sample_alpha(alphas)
            global_variables[i]['alpha'] = alphas
            pbar.update()
        return global_variables

    def __average_over_mc_steps(self, ix: torch.Tensor, mc_steps: int, etas: torch.Tensor, alphas: torch.Tensor, norm_bow: torch.Tensor,
                                bow: torch.Tensor) -> torch.Tensor:
        ppl = []
        text_size = bow.sum(1).double().to(self.device, non_blocking=True)
        for j in range(mc_steps):
            eta_per_d = etas[j].repeat((ix.sum(), 1))
            alpha_per_d = alphas[j]
            theta_logits = self.theta_q(norm_bow, eta_per_d, self.eta_transform)[0]
            theta = self.proportion_transformation(theta_logits)  # [batch_size, number_of_topics]
            beta = self.get_beta(alpha_per_d)
            nll_topic = self.nll(theta, bow, beta)
            log_perplexity = nll_topic / text_size.float()
            ppl.append(log_perplexity)
        return torch.mean(torch.stack(ppl), dim=0).cpu().detach().numpy()

    def prediction(self, data_loader: ADataLoader, montecarlo_samples=10):
        self.eval()

        n_ts = data_loader.num_training_steps
        n_ps = data_loader.num_prediction_steps
        with torch.no_grad():
            global_variables = self._sample_global_variables(data_loader.train, n_ts, n_ps, montecarlo_samples)
            log_ppls = defaultdict(list)
            for x in tqdm(data_loader.predict, desc='Minibatch'):
                for i in tqdm(range(n_ts, n_ts + n_ps), total=n_ps, desc='Prediction Step'):
                    time_idx = x['time'].squeeze().long().to(self.device, non_blocking=True)
                    ix = time_idx == i
                    if ix.sum() == 0:
                        continue
                    bow = x['bow'][ix].to(self.device, non_blocking=True)
                    norm_bow = bow.float() / bow.sum(1, True)

                    alpha_ = global_variables[i]['alpha']
                    etas = global_variables[i]['eta']
                    log_ppl = self.__average_over_mc_steps(ix, montecarlo_samples, etas, alpha_, norm_bow, bow)
                    log_ppls[i].extend(log_ppl)

        return np.asarray([np.mean(v) for v in log_ppls.values()])

    def topic_diversity(self, topk: int = 25) -> float:
        if self.topic_embeddings == "static":
            important_words_per_topic = self.get_words_by_importance()
            list_w = important_words_per_topic[:, :topk]
            n_unique = len(torch.unique(list_w))
            td = n_unique / (topk * self.number_of_topics)
        else:
            td = super().topic_diversity(topk)
        return td

    def get_time_series(self, dataset):
        theta_stats = defaultdict(list)
        for x in tqdm(dataset, desc="Building time series"):
            theta = self.forward(x)["theta"]
            time_idx = x['time'].squeeze()
            for id, t in zip(time_idx, theta):
                theta_stats[id.item()].append(t)

        theta_stats_m = OrderedDict()
        theta_stats_s = OrderedDict()
        theta_stats_size = OrderedDict()
        for key, value in theta_stats.items():
            theta_stats_m[key] = torch.stack(value).mean(0)
            theta_stats_s[key] = torch.stack(value).std(0)
            theta_stats_size[key] = len(value)
        theta_stats_m = OrderedDict(sorted(theta_stats_m.items()))
        theta_stats_std = OrderedDict(sorted(theta_stats_s.items()))
        theta_stats_size = OrderedDict(sorted(theta_stats_size.items()))

        return torch.stack(list(theta_stats_m.values())), torch.stack(list(theta_stats_std.values())), torch.tensor(list(theta_stats_size.values()))

    @property
    def is_dynamic_topc_embeddings(self):
        return self.topic_embeddings == 'dynamic'



