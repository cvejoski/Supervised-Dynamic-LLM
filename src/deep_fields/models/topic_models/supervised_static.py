import os
from deep_fields.models.deep_state_space.deep_state_space_recognition import q_DIVA
import torch

from deep_fields.models.deep_architectures.deep_nets import MLP
from deep_fields.data.topic_models.dataloaders import ADataLoader
from deep_fields.data.topic_models.dataloaders import TopicTransformerLanguageDataloader
from deep_fields.models.topic_models.attention import AttentionLayer
from deep_fields.models import basic_utils
from deep_fields.models.topic_models.llm_utils import load_backbone
from deep_fields.models.topic_models.static import DiscreteLatentTopicNVI
from torch import nn
import pprint


class SupervisedTopicModel(DiscreteLatentTopicNVI):
    theta_q: nn.Module
    lambda_diversity: float = 0.1
    name_: str = "supervised_static_vanilla_topic"

    train_llm: bool = False

    def __init__(self, model_dir=None, data_loader: ADataLoader = None, **kwargs):
        DiscreteLatentTopicNVI.__init__(self, model_name=self.name_, model_dir=model_dir, data_loader=data_loader, **kwargs)
        self.__loss_mse = nn.MSELoss(reduction='sum')
        self.__loss_mae = nn.L1Loss(reduction='sum')
        # self.__loss_gaussian_nll = nn.GaussianNLLLoss(reduction='sum')

    @classmethod
    def get_parameters(cls):
        model_parameters = super().get_parameters()
        model_parameters.update({
            "llm": {},
            "covariates_encoder": {
                "input_dim": 1,
                "layers_dim": [64, 64],
                "output_dim": 32,
                "dropout": .1,
            },
            "regression_head": {
                "layers_dim": [64, 64],
                "dropout": .1,
                "output_transformation": "relu",
                "normalization": True
            }
        })
        return model_parameters

    def set_parameters(self, **kwargs):
        super().set_parameters(**kwargs)
        self.covariates_encoder_param = kwargs.get("covariates_encoder")
        self.covariates_dim = self.covariates_encoder_param.get("input_dim")
        self.covariates_hidden_size = self.covariates_encoder_param.get("output_dim")
        self.regressioin_head_param = kwargs.get("regression_head")
        self.llm_param = kwargs.get("llm", {})
        self.delta = kwargs.get("delta")

    def update_parameters(self, data_loader, **kwargs):
        kwargs = super().update_parameters(data_loader, **kwargs)
        kwargs.get("covariates_encoder").update({"input_dim": data_loader.train.dataset.covariates_size})
        return kwargs

    def define_deep_models(self):
        super().define_deep_models()
        self.__init_lm()
        self.regression_head = MLP(input_dim=self.regressinon_input_size,
                                   layers_dim=self.regressioin_head_param.get("layers_dim"),
                                   output_dim=1,
                                   normalization=self.regressioin_head_param.get("normalization"),
                                   ouput_transformation=self.regressioin_head_param.get("output_transformation"),
                                   dropout=self.regressioin_head_param.get("dropout"))

    def __init_lm(self):
        llm_type = self.llm_param.get("type", None)

        if llm_type == "GRU":
            input_size_ = self.word_embeddings_dim
            hidden_size_ = self.llm_param.get("hidden_size")
            dropout_ = self.llm_param.get("dropout")
            num_layers_ = self.llm_param.get("num_layers")
            train_word_emb = self.llm_param.get('train_word_embeddings', False)
            self.llm = nn.GRU(input_size_, hidden_size_, num_layers_, dropout=dropout_, batch_first=True, bidirectional=True)
            vocab = self.data_loader.vocab
            self.voc_dim, self.emb_dim = vocab.vectors.size()
            self.embedding = nn.Embedding(self.voc_dim, self.emb_dim)
            self._forward = self._forward_gru
            if vocab.vectors is not None:
                emb_matrix = vocab.vectors.to(self.device)
                self.embedding.weight.data.copy_(emb_matrix)
                self.embedding.weight.requires_grad = train_word_emb
            else:
                self.embedding.weight.data.uniform_(-0.1, 0.1)
                self.embedding.weight.requires_grad = True

            self.attention_head = AttentionLayer(2 * hidden_size_, self.word_embeddings_dim, self.delta)
            self.regressinon_input_size = 2 * hidden_size_
        elif llm_type in ["bert", "roberta", "albert"]:

            self.llm = load_backbone(name=llm_type)
            self.attention_head = AttentionLayer(768, self.word_embeddings_dim, self.delta)
            self.regressinon_input_size = 768
            self._forward = self._forward_transformer
            if llm_type == "roberta":
                self._forward = self._forward_transformer_roberta
        else:
            self._forward = self._forward_no_lm
            self.llm = None
            self.regressinon_input_size = self.theta_size

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()

        inference_parameters.update({
            "alpha_y": 1.,
            "model_eval": "RMSE",
            "regularizers": {
                "kltheta": {
                    "lambda_0": 1.,
                    "percentage": .2
                },
                "alpha": {
                    "lambda_0": 1.,
                    "percentage": .2
                }
            }
        })

        return inference_parameters

    def initialize_inference(self, data_loader: ADataLoader, **inference_parameters) -> None:
        if self.llm_param.get('type') in ['bert', 'albert']:
            optimizer_lm_params = inference_parameters.get('optimizer_lm')
            params_topic_model = [p for n, p in self.named_parameters() if 'llm' not in n]
            params_language_model = [p for n, p in self.named_parameters() if 'llm' in n]
            super().initialize_inference(data_loader=data_loader, parameters=params_topic_model, **inference_parameters)
            self.optimizer_lm = self._create_optimizer(optmimizer_name=optimizer_lm_params["name"],
                                                       lr=optimizer_lm_params['lr'],
                                                       weight_decay=optimizer_lm_params['weight_decay'],
                                                       parameters=params_language_model)
            self.optimizer = [self.optimizer, self.optimizer_lm]
        else:

            super().initialize_inference(data_loader=data_loader, **inference_parameters)

        self.alpha_y = inference_parameters.get("alpha_y")

        basic_utils.set_cuda(self, **inference_parameters)
        basic_utils.set_cuda(self.theta_q, **inference_parameters)

    def forward(self, x):
        if 'bow_h1' in x:
            bow = x['bow_h1'].to(self.device, non_blocking=True)
            normalized_bow = bow.float() / bow.sum(1, True)
            bow = x['bow_h2'].to(self.device, non_blocking=True)
        else:
            bow = x['bow'].to(self.device, non_blocking=True)
            normalized_bow = bow.float() / bow.sum(1, True)

        out = self._forward(x, bow, normalized_bow)

        return out

    def _forward_transformer_roberta(self, x, bow, normalized_bow):
        out = self._forward_topic_model(bow, normalized_bow)
        text = x['text']
        h, _ = self.llm(input_ids=text["input_ids"].squeeze(1).to(self.device, non_blocking=True),
                        attention_mask=text["attention_mask"].squeeze(1).to(self.device, non_blocking=True),
                        return_dict=False)  # hidden, pooled
        s, _ = self.attention_head(h, self.topic_embeddings, out["theta"])

        y_hat = self.regression_head(s)
        out["y_hat"] = y_hat
        return out

    def _forward_transformer(self, x, bow, normalized_bow):
        out = self._forward_topic_model(bow, normalized_bow)
        text = x['text']
        h, _ = self.llm(input_ids=text["input_ids"].squeeze(1).to(self.device, non_blocking=True),
                        token_type_ids=text["token_type_ids"].squeeze(1).to(self.device, non_blocking=True),
                        attention_mask=text["attention_mask"].squeeze(1).to(self.device, non_blocking=True),
                        return_dict=False)  # [B, 512, 738]
        s, _ = self.attention_head(h, self.topic_embeddings, out["theta"])

        y_hat = self.regression_head(s)

        out["y_hat"] = y_hat
        return out

    def _forward_gru(self, x, bow, normalized_bow):
        out = self._forward_topic_model(bow, normalized_bow)

        text, text_len = x['seq'].long().to(self.device, non_blocking=True), x['seq_len']
        text = self.embedding(text)  # [B, T, D]
        t_len = torch.max(text_len).item()
        text = torch.nn.utils.rnn.pack_padded_sequence(text, text_len, True, enforce_sorted=False)

        h, _ = self.llm(text)  # [B, T, 2*D]
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, True, total_length=t_len)
        s, _ = self.attention_head(h, self.topic_embeddings, out["theta"])

        y_hat = self.regression_head(s)
        out["y_hat"] = y_hat
        return out

    def _forward_no_lm(self, x, bow, normalized_bow):
        out = self._forward_topic_model(bow, normalized_bow)

        y_hat = self.regression_head(out["theta_logits"])
        out["y_hat"] = y_hat
        return out

    def _forward_topic_model(self, bow, normalized_bow):
        theta_logits, _, kl_theta = self.theta_q(normalized_bow.unsqueeze(1))
        theta = self.proportion_transformation(theta_logits)  # [batch_size, number_of_topics]
        beta = self.get_beta()
        nll = self.nll(theta, bow, beta)
        return {"theta": theta, "theta_logits": theta_logits, "kl_theta": kl_theta, "nll": nll}

    def loss(self, x, forward_results, data_loader, epoch):
        """
        nll [batch_size, max_lenght]

        """
        topic_diversity_reg = 0.0
        if 'bow_h2' in x:
            text_size = x['bow_h2'].sum(1).double().to(self.device, non_blocking=True)
        else:
            text_size = x['bow'].sum(1).double().to(self.device, non_blocking=True)

        y_hat = forward_results["y_hat"].squeeze(1)
        nll_topic = forward_results["nll"]
        kl_theta = forward_results["kl_theta"]
        target = x['reward'].float().to(self.device, non_blocking=True)
        batch_size = nll_topic.size(0)

        # nll_regression = self.__loss_gaussian_nll(y_hat, target, torch.ones_like(target) * 0.0001)
        loss_mse = self.__loss_mse(y_hat, target)
        mae = self.__loss_mae(y_hat, target)
        # beta_kltheta = self.schedulers["kltheta"](self.number_of_iterations)

        if self.schedulers.get("alpha") is None:
            alpha = self.alpha_y
        else:
            alpha = self.schedulers["alpha"](self.number_of_iterations)

        loss = (nll_topic.sum() + alpha * loss_mse + kl_theta) / batch_size

        # topic_diversity_reg = self.topic_diversity_regularizer()
        # loss = loss - self.lambda_diversity * topic_diversity_reg  # / batch_size

        log_perplexity = nll_topic / text_size.float()
        log_perplexity = torch.mean(log_perplexity)
        perplexity = torch.exp(log_perplexity)

        return {
            "loss": loss,
            "MAE": mae / batch_size,
            "RMSE": torch.sqrt(loss_mse / batch_size),
            "Topic-Div-Reg": topic_diversity_reg,
            "NLL-Topic": nll_topic.sum().item() / batch_size,
            # "NLL-Regression": nll_regression.item() / batch_size,
            "KL-Loss-Theta": kl_theta / batch_size,
            "KL-Y-Beta": 0,
            "Alpha": alpha,
            "PPL-Blei": perplexity,
            "Log-Likelihood": log_perplexity
        }

    @property
    def is_static_topic_embeddings(self):
        return True



