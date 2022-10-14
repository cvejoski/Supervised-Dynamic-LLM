import numpy as np
import torch
from deep_fields.data.topic_models.dataloaders import ADataLoader
from deep_fields.models.deep_architectures.deep_nets import MLP
from deep_fields.models.topic_models.attention import AttentionLayer
from deep_fields.models.topic_models.dynamic import DynamicTopicEmbeddings
from deep_fields.models.topic_models.llm_utils import load_backbone
from deep_fields.utils.weird_functions import filter_parameters
from torch import nn
from tqdm import tqdm
from deep_fields.models.topic_models.topics_utils import reparameterize
from deep_fields.utils.loss_utils import kullback_leibler_two_gaussians


class SupervisedDynamicTopicModel(DynamicTopicEmbeddings):
    name_: str = "supervised_dynamic_topic_model"
    TOPIC_PARAMS = ["rho", "mu_q_alpha", "eta_q", "theta_q", "hidden_state_to_topics", "eta_transform", "p_eta_m"]
    LANGUAGE_PARAMS = ["llm"]

    def __init__(self, model_dir=None, data_loader: ADataLoader = None, evaluate_mode=False, model_name=None, **kwargs):
        model_name = "supervised_dynamic_topic_model" if model_name is None else model_name
        DynamicTopicEmbeddings.__init__(self,
                                        model_name=self.name_,
                                        model_dir=model_dir,
                                        data_loader=data_loader,
                                        evaluate_mode=evaluate_mode,
                                        **kwargs)
        self.__loss_rmse = nn.MSELoss(reduction='sum')
        self.__loss_mae = nn.L1Loss(reduction='sum')

    @classmethod
    def get_parameters(cls):
        model_parameters = super().get_parameters()
        model_parameters.update({
            "topic_embeddings": "static",  # dynamic
            "nonlinear_transition_prior": True,
            "pre_trained_topic_model_path": None,
            "pre_trained_language_model_path": None,
            "freeze_topic_model_training": False,
            "freeze_language_model_training": False,
            "covariates_encoder": {
                "input_dim": 1,
                "layers_dim": [64, 64],
                "output_dim": 32,
                "dropout": .1,
            },
            "regression_head": {
                "layers_dim": [64, 64],
                "dropout": .1,
                "output_transformation": "relu"
            }
        })

        return model_parameters

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()

        inference_parameters.update({
            "train_llm": True,
            "train_tm": True,
            "alpha_y": 1,
            "model_eval": "RMSE",
            "regularizers": {
                "theta_kl": {
                    "module": "deep_fields.utils.schedulers",
                    "name": "ExponentialScheduler",
                    "args": {
                        "max_steps": 5000,
                        "decay_rate": 0.0025
                    }
                },
                "alpha_kl": {
                    "module": "deep_fields.utils.schedulers",
                    "name": "ExponentialScheduler",
                    "args": {
                        "max_steps": 5000,
                        "decay_rate": 0.0025
                    }
                },
                "eta_kl": {
                    "module": "deep_fields.utils.schedulers",
                    "name": "ExponentialScheduler",
                    "args": {
                        "max_steps": 5000,
                        "decay_rate": 0.0025
                    }
                },
                "alpha": {
                    "module": "deep_fields.utils.schedulers",
                    "name": "ExponentialScheduler",
                    "args": {
                        "max_steps": 5000,
                        "decay_rate": 0.0025
                    }
                }
            }
        })

        return inference_parameters

    def set_parameters(self, **kwargs):
        super().set_parameters(**kwargs)
        self.covariates_encoder_param = kwargs.get("covariates_encoder")
        self.covariates_dim = self.covariates_encoder_param.get("input_dim")
        self.covariates_hidden_size = self.covariates_encoder_param.get("output_dim")
        self.regressioin_head_param = kwargs.get("regression_head")
        self.two_optimizers = kwargs.get("two_optimizers", True)
        self.llm_param = kwargs.get("llm", {})
        self.delta = kwargs.get("delta")
        self.pre_trained_topic_model_path = kwargs.get("pre_trained_topic_model_path")
        self.pre_trained_language_model_path = kwargs.get("pre_trained_language_model_path")
        self.freeze_topic_model_training = kwargs.get("freeze_topic_model_training", False)
        self.freeze_language_model_training = kwargs.get("freeze_language_model_training", False)
        self.llm_type = self.llm_param.get("type", None)

    def __init_lm(self):
        if self.llm_type == "GRU":
            input_size_ = self.word_embeddings_dim
            hidden_size_ = self.llm_param.get("hidden_size")
            dropout_ = self.llm_param.get("dropout")
            num_layers_ = self.llm_param.get("num_layers")
            train_word_emb = self.llm_param.get('train_word_embeddings', False)
            self.llm = nn.GRU(input_size_, hidden_size_, num_layers_, dropout=dropout_, batch_first=True, bidirectional=True)
            vocab = self.data_loader.vocab
            self.voc_dim, self.emb_dim = vocab.vectors.size()
            self.llm_embedding = nn.Embedding(self.voc_dim, self.emb_dim)
            self._forward = self._forward_gru
            if vocab.vectors is not None:
                emb_matrix = vocab.vectors.to(self.device)
                self.llm_embedding.weight.data.copy_(emb_matrix)
                self.llm_embedding.weight.requires_grad = train_word_emb
            else:
                self.llm_embedding.weight.data.uniform_(-0.1, 0.1)
                self.llm_embedding.weight.requires_grad = True

            self.attention_head = AttentionLayer(2 * hidden_size_, self.word_embeddings_dim, self.delta)
            self.regressinon_input_size = 2 * hidden_size_
        elif self.llm_type in ["bert", "roberta", "albert"]:
            self.llm = load_backbone(name=self.llm_type)
            if self.pre_trained_language_model_path is not None and not self.evaluate_mode:
                print("Loading pre-trained language model from : " + self.pre_trained_language_model_path)
                params_llm = torch.load(self.pre_trained_language_model_path)['state_dict']
                keys = list(params_llm.keys())
                for k in keys:
                    if 'backbone' in k:
                        params_llm[k.replace('backbone.', '')] = params_llm.pop(k)
                    else:
                        del params_llm[k]
                self.llm.load_state_dict(params_llm, strict=False)
            self.attention_head = AttentionLayer(768, self.word_embeddings_dim, self.delta)
            self.regressinon_input_size = 768
            self._forward = self._forward_transformer
        else:
            self._forward = self._forward_no_lm
            self.llm = None
            self.regressinon_input_size = self.theta_emb_dim

    def update_parameters(self, data_loader, **kwargs):
        kwargs = super().update_parameters(data_loader, **kwargs)

        kwargs.get("covariates_encoder").update({"input_dim": data_loader.train.dataset.covariates_size})
        return kwargs

    def define_deep_models(self):
        super().define_deep_models()
        if self.pre_trained_topic_model_path is not None and not self.evaluate_mode:
            print("Loading pre-trained topic model from : " + self.pre_trained_topic_model_path)
            self.load_state_dict(torch.load(self.pre_trained_topic_model_path)['state_dict'], strict=False)

        self.__init_lm()

        self.regression_head = MLP(input_dim=self.regressinon_input_size,
                                   layers_dim=self.regressioin_head_param.get("layers_dim"),
                                   output_dim=1,
                                   ouput_transformation=self.regressioin_head_param.get("output_transformation"),
                                   dropout=self.regressioin_head_param.get("dropout"))

    def initialize_inference(self, data_loader: ADataLoader, **inference_parameters) -> None:
        self.alpha_y = inference_parameters.get("alpha_y")
        if self.two_optimizers:
            optimizer_lm_params = inference_parameters.get('optimizer_lm')
            params_topic_language_model = [p for n, p in self.named_parameters() if n.split('.')[0] in self.LANGUAGE_PARAMS + self.TOPIC_PARAMS]
            params_rest = [p for n, p in self.named_parameters() if n.split('.')[0] not in self.LANGUAGE_PARAMS + self.TOPIC_PARAMS]
            super().initialize_inference(data_loader=data_loader, parameters=params_rest, **inference_parameters)
            self.optimizer_lm = self._create_optimizer(optmimizer_name=optimizer_lm_params["name"],
                                                       lr=optimizer_lm_params['lr'],
                                                       weight_decay=optimizer_lm_params['weight_decay'],
                                                       parameters=params_topic_language_model)
            self.lr_scheduler_lm = self._init_lr_scheduler(self.optimizer_lm, **inference_parameters)
            self.optimizer = [self.optimizer, self.optimizer_lm]
            self.lr_scheduler = [self.lr_scheduler, self.lr_scheduler_lm]
        else:
            parameters = None
            if self.freeze_topic_model_training:
                for n, p in self.named_parameters():
                    if n in self.TOPIC_PARAMS:
                        p.requires_grad_(False)

            if self.freeze_language_model_training:
                for n, p in self.named_parameters():
                    if n in self.LANGUAGE_PARAMS:
                        p.requires_grad_(False)
            if self.freeze_topic_model_training or self.freeze_language_model_training:
                parameters = [p for n, p in self.named_parameters() if not filter_parameters(n, self.TOPIC_PARAMS + self.LANGUAGE_PARAMS)]
            super().initialize_inference(data_loader=data_loader, parameters=parameters, **inference_parameters)

    def forward(self, x):
        time_idx = x['time'].squeeze().long().to(self.device, non_blocking=True)
        corpora = x['corpus'][:1].float().to(self.device, non_blocking=True)

        if 'bow_h1' in x:
            bow = x['bow_h1'].to(self.device, non_blocking=True)
            norm_bow = bow.float() / bow.sum(1, True)
            bow = x['bow_h2'].to(self.device, non_blocking=True)
        else:
            bow = x['bow'].to(self.device, non_blocking=True)
            norm_bow = bow.float() / bow.sum(1, True)

        return self._forward(x, time_idx, corpora, bow, norm_bow)

    def _forward_transformer(self, x, time_idx, corpora, bow, norm_bow):
        out = self._forward_topic_model(time_idx, corpora, bow, norm_bow)

        s = self._get_regressor_transformer(x, out["theta"], out["alpha"])

        y_hat = self.regression_head(s)

        out["y_hat"] = y_hat
        return out

    def _get_regressor_transformer(self, x, theta, alpha, ksi=None):
        h, _ = self.llm(input_ids=x['text']["input_ids"].squeeze(1).to(self.device, non_blocking=True),
                        token_type_ids=x['text']["token_type_ids"].squeeze(1).to(self.device, non_blocking=True),
                        attention_mask=x['text']["attention_mask"].squeeze(1).to(self.device, non_blocking=True),
                        return_dict=False)  # [B, 512, 738]
        s, _ = self.attention_head(h, alpha, theta)
        return s

    def _forward_gru(self, x, time_idx, corpora, bow, norm_bow):
        out = self._forward_topic_model(time_idx, corpora, bow, norm_bow)

        s = self._get_regressor_gru(x['seq'], x['seq_len'], out["theta"], out["alpha"])

        y_hat = self.regression_head(s)
        out["y_hat"] = y_hat
        return out

    def _get_regressor_gru(self, text, text_len, theta, alpha, ksi=None):
        text = text.long().to(self.device, non_blocking=True)
        text = self.llm_embedding(text)  # [B, T, D]
        t_len = torch.max(text_len).item()
        text = torch.nn.utils.rnn.pack_padded_sequence(text, text_len, True, enforce_sorted=False)

        h, _ = self.llm(text)  # [B, T, 2*D]
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, True, total_length=t_len)
        s, _ = self.attention_head(h, alpha, theta)
        return s

    def _forward_no_lm(self, x, time_idx, corpora, bow, norm_bow):
        out = self._forward_topic_model(time_idx, corpora, bow, norm_bow)

        y_hat = self.regression_head(out["theta"])
        out["y_hat"] = y_hat
        return out

    def _forward_topic_model(self, time_idx, corpora, bow, norm_bow):
        alpha, alpha_kl = self.q_alpha()
        eta, _, eta_kl = self.eta_q(corpora, self.p_eta_m)
        eta_per_d = eta[time_idx]
        if eta_per_d.dim() == 1:
            eta_per_d = eta_per_d.unsqueeze(0)
        beta = self.get_beta(alpha)
        if self.topic_embeddings == 'dynamic':
            alpha = alpha[time_idx]
            beta = beta[time_idx]

        theta_logits, _, theta_kl = self.theta_q(norm_bow, eta_per_d, self.eta_transform)

        theta = self.proportion_transformation(theta_logits)  # [batch_size, number_of_topics]

        nll = self.nll(theta, bow, beta)

        return {"nll": nll, "alpha": alpha, "theta_kl": theta_kl, "alpha_kl": alpha_kl, "eta_kl": eta_kl, "eta": eta[-1], "theta": theta}

    def loss(self, x, forward_results, data_set, epoch):
        """
        nll [batch_size, max_lenght]

        """
        topic_diversity_reg = 0.0
        if 'bow_h2' in x:
            text_size = x['bow_h2'].sum(1).double().to(self.device, non_blocking=True)
        else:
            text_size = x['bow'].sum(1).double().to(self.device, non_blocking=True)

        nll_topic, theta_kl, alpha_kl, eta_kl, eta, y_hat = forward_results['nll'], forward_results['theta_kl'], forward_results[
            'alpha_kl'], forward_results['eta_kl'], forward_results['eta'], forward_results['y_hat']

        target = x['reward'].float().to(self.device, non_blocking=True)
        y_hat = y_hat.squeeze(1)
        # nll_regression = self.__loss_gaussian_nll(y_hat, target, torch.ones_like(target) * 0.0000001)
        loss_mse = self.__loss_rmse(y_hat, target)
        mae = self.__loss_mae(y_hat, target)
        coeff = 1.0
        if self.training:
            coeff = len(data_set)
        batch_size = nll_topic.size(0)

        if self.schedulers.get("alpha") is None:
            alpha = self.alpha_y
        else:
            alpha = self.schedulers["alpha"](self.number_of_iterations)

        if self.freeze_topic_model_training:
            loss = loss_mse
        else:
            loss = (nll_topic.sum() + theta_kl + alpha * loss_mse) + (alpha_kl + eta_kl) / coeff

        log_perplexity = nll_topic / text_size.float()
        log_perplexity = torch.mean(log_perplexity)
        perplexity = torch.exp(log_perplexity)

        return {
            "loss": loss,
            "MAE": mae / batch_size,
            "RMSE": torch.sqrt(loss_mse / batch_size),
            "Topic-Div-Reg": topic_diversity_reg,
            "REWARD-ALPHA": alpha,
            "NLL-Topic": nll_topic.sum(),  #* coeff,
            # "NLL-REGRESSION": nll_regression.sum() * coeff,
            "KL-Loss-Theta": theta_kl,  #* coeff,
            "KL-Loss-ALPHA": alpha_kl,
            "KL-Loss-ETA": eta_kl,
            "PPL-Blei": perplexity,
            "Log-Likelihood": log_perplexity,
            "ETA_MEAN": eta.mean(),
            "ETA_MIN": eta.min(),
            "ETA_MAX": eta.max()
        }

    def __average_over_mc_steps(self, ix: torch.Tensor, mc_steps: int, etas: torch.Tensor, alphas: torch.Tensor, norm_bow: torch.Tensor,
                                bow: torch.Tensor, x: dict) -> torch.Tensor:
        y_hat_t, ppl = [], []
        text_size = bow.sum(1).double().to(self.device, non_blocking=True)
        if self.llm_type in ["bert", "roberta", "albert"]:
            x_sliced = {'text': {}}
            x_sliced['text']["input_ids"] = x['text']["input_ids"][ix]
            x_sliced['text']["token_type_ids"] = x['text']["token_type_ids"][ix]
            x_sliced['text']["attention_mask"] = x['text']["attention_mask"][ix]
            h, _ = self.llm(input_ids=x_sliced['text']["input_ids"].squeeze(1).to(self.device, non_blocking=True),
                            token_type_ids=x_sliced['text']["token_type_ids"].squeeze(1).to(self.device, non_blocking=True),
                            attention_mask=x_sliced['text']["attention_mask"].squeeze(1).to(self.device, non_blocking=True),
                            return_dict=False)  # [B, 512, 738]
        for j in range(mc_steps):
            eta_per_d = etas[j].repeat((ix.sum(), 1))
            alpha_per_d = alphas[j]
            theta_logits = self.theta_q(norm_bow, eta_per_d, self.eta_transform)[0]
            theta = self.proportion_transformation(theta_logits)  # [batch_size, number_of_topics]
            beta = self.get_beta(alpha_per_d)
            nll_topic = self.nll(theta, bow, beta)
            log_perplexity = nll_topic / text_size.float()
            if self.llm_type is None:
                s = theta_logits
            elif self.llm_type == 'GRU':
                s = self._get_regressor_gru(x['seq'][ix], x['seq_len'][ix], theta, alpha_per_d)
            elif self.llm_type in ["bert", "roberta", "albert"]:
                s, _ = self.attention_head(h, alpha_per_d, theta)
            y_ = self.regression_head(s)
            ppl.append(log_perplexity)
            y_hat_t.append(y_)
        ppl = torch.mean(torch.stack(ppl), dim=0).cpu().detach().numpy().flatten().tolist()
        y_hat_t = torch.mean(torch.stack(y_hat_t), dim=0).cpu().detach().numpy().flatten().tolist()
        return y_hat_t, ppl

    def prediction(self, data_loader: ADataLoader, montecarlo_samples=10):
        self.eval()

        n_ts = data_loader.num_training_steps
        n_ps = data_loader.num_prediction_steps
        with torch.no_grad():
            global_variables = self._sample_global_variables(data_loader.train, n_ts, n_ps, montecarlo_samples)
            y, y_hats, time, log_ppls = [], [], [], []

            for x in tqdm(data_loader.predict, desc='Prediction'):
                for i in tqdm(range(n_ts, n_ts + n_ps), total=n_ps, desc='Prediction Step'):
                    time_idx = x['time'].squeeze().long().to(self.device, non_blocking=True)
                    ix = time_idx == i
                    if ix.sum() == 0:
                        continue
                    bow = x['bow'][ix].to(self.device, non_blocking=True)
                    norm_bow = bow.float() / bow.sum(1, True)

                    alpha_ = global_variables[i]['alpha']
                    etas = global_variables[i]['eta']

                    y_hat, log_ppl = self.__average_over_mc_steps(ix, montecarlo_samples, etas, alpha_, norm_bow, bow, x)
                    log_ppls.extend(log_ppl)
                    y_hats.extend(y_hat)
                    y.extend(x['reward'][ix].cpu().numpy())
                    time.extend(x['time'][ix].cpu().numpy())

        y = np.asarray(y)
        y_hats = np.asarray(y_hats)
        time = np.asarray(time)
        ppl = np.asarray(log_ppls)
        return y, y_hats, time, ppl
