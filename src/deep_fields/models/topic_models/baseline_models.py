import os

import torch
from torch import nn

from deep_fields import project_path
from deep_fields.data.topic_models.dataloaders import (
    ADataLoader,
    TopicDataloader,
    TopicTransformerLanguageDataloader,
)
from deep_fields.models.abstract_models import DeepBayesianModel
from deep_fields.models.deep_architectures.deep_nets import MLP


def exp_nllloss(y_hat, y, var):
    """
    Calculates the negative log-likelihood loss for exponential distribution.
    Parameters:
    - y_hat (torch.Tensor): Predicted values.
    - y (torch.Tensor): True values.

    Returns:
    - torch.Tensor: Negative log-likelihood loss.
    """

    dist = torch.distributions.exponential.Exponential(y_hat)
    log_likelihood = dist.log_prob(y).sum()
    return -log_likelihood


def load_backbone(name, output_attentions=False, small=False):
    """
    Load a backbone model for topic modeling.
    Parameters:
        name (str): The name of the backbone model. Supported options are "bert", "roberta", and "albert".
        output_attentions (bool, optional): Whether to output attentions. Defaults to False.
        small (bool, optional): Whether to use a smaller version of the model. Defaults to False.
    Returns:
        backbone: The loaded backbone model.
    Raises:
        ValueError: If the provided name does not match any supported backbone network.
    """

    if name == "bert":
        from transformers import BertModel  # , BertTokenizer

        backbone = BertModel.from_pretrained(
            "bert-base-uncased", output_attentions=output_attentions
        )

    elif name == "roberta":
        from transformers import RobertaModel  # , RobertaTokenizer

        backbone = RobertaModel.from_pretrained(
            "roberta-base", output_attentions=output_attentions
        )

    elif name == "albert":
        from transformers import AlbertModel

        backbone = AlbertModel.from_pretrained(
            "albert-base-v2", output_attentions=output_attentions
        )

    else:
        raise ValueError("No matching backbone network")

    return backbone


class NonSequentialModel(DeepBayesianModel):
    
    def __init__(
        self, model_dir=None, data_loader: ADataLoader = None, model_name=None, **kwargs
    ):
        DeepBayesianModel.__init__(
            self, model_name, model_dir=model_dir, data_loader=data_loader, **kwargs
        )

    @classmethod
    def get_parameters(cls):
        """
        here we provide an example of the minimum set of parameters requiered to instantiate the model
        """
        parameters_sample = {
            "vocabulary_dim": 100,
            "dropout": 0.1,
            "layers_dim": [10, 10],
            "bow_layers_dim": [10, 10],
            "cov_layers_dim": [10, 10],
            "bow_emb_dim": 50,
            "cov_emb_dim": 50,
            "output_dim": 1,
            "covariates_dim": 0,
            "model_path": os.path.join(project_path, "results"),
        }

        return parameters_sample

    def set_parameters(self, **kwargs):
        self.covariates_dim = kwargs.get("covariates_dim")
        self.vocabulary_dim = kwargs.get("vocabulary_dim")
        self.output_dim = kwargs.get("output_dim")
        self.dropout = kwargs.get("dropout")
        self.layers_dim = kwargs.get("layers_dim")
        self.bow_layers_dim = kwargs.get("bow_layers_dim")
        self.cov_layers_dim = kwargs.get("cov_layers_dim")
        self.cov_emb_dim = kwargs.get("cov_emb_dim")
        self.bow_emb_dim = kwargs.get("bow_emb_dim")
        self.word_emb_type = kwargs.get("word_emb_type")

    def update_parameters(self, data_loader: ADataLoader, **kwargs):
        loss_type = self.get_inference_parameters()["loss_type"]
        kwargs.update(
            {
                "vocabulary_dim": data_loader.vocabulary_dim,
                "word_emb_type": data_loader.word_emb_type,
                "covariates_dim": data_loader.train.dataset.covariates_size,
                "output_dim": data_loader.number_of_reward_categories
                if loss_type == "classification"
                else 1,
            }
        )

        return kwargs

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()
        inference_parameters.update({"regularizers": {"nll": None}})
        inference_parameters.update({"model_eval": "RMSE"})

        return inference_parameters

    def define_deep_models(self):
        if self.covariates_dim == 0:
            self.bow2emb = MLP(
                input_dim=self.vocabulary_dim,
                output_dim=self.bow_emb_dim,
                layers_dim=self.bow_layers_dim,
                dropout=self.dropout,
                ouput_transformation="relu",
            )
            self.regressor = MLP(
                input_dim=self.bow_emb_dim,
                output_dim=self.output_dim,
                layers_dim=self.layers_dim,
                dropout=self.dropout,
                ouput_transformation="exp",
            )
        else:
            self.bow2emb = MLP(
                input_dim=self.vocabulary_dim,
                output_dim=self.bow_emb_dim,
                layers_dim=self.bow_layers_dim,
                dropout=self.dropout,
                ouput_transformation="relu",
            )
            self.cov2emb = MLP(
                input_dim=self.covariates_dim,
                output_dim=self.cov_emb_dim,
                layers_dim=self.cov_layers_dim,
                dropout=self.dropout,
                ouput_transformation="relu",
            )
            self.regressor = MLP(
                input_dim=self.bow_emb_dim + self.cov_emb_dim,
                output_dim=self.output_dim,
                layers_dim=self.layers_dim,
                dropout=self.dropout,
                ouput_transformation="exp",
            )

    def forward(self, x):
        """
        Forward pass of the baseline model.
        Args:
            x (dict): Input dictionary containing the following keys:
                - "bow" (torch.Tensor): Bag-of-words representation of the input data.
                - "covariates" (torch.Tensor): Covariates data.
        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        pass

        bow = x["bow"].to(self.device, non_blocking=True)

        emb = bow
        if self.word_emb_type == "bow":
            emb = bow.float() / bow.sum(1, True)

        emb = self.bow2emb(emb)
        if self.covariates_dim != 0:
            cov_emb = self.cov2emb(x["covariates"].to(self.device, non_blocking=True))
            emb = torch.cat((emb, cov_emb), dim=1)
        y = self.regressor(emb)
        return y


class NonSequentialClassifier(NonSequentialModel):
    name_: str = "nonsequential_classifier"

    def __init__(self, model_dir=None, data_loader: ADataLoader = None, **kwargs):
        NonSequentialModel.__init__(
            self,
            model_name=self.name_,
            model_dir=model_dir,
            data_loader=data_loader,
            **kwargs,
        )

    def initialize_inference(
        self, data_loader: ADataLoader, parameters=None, **inference_parameters
    ) -> None:
        super().initialize_inference(
            data_loader=data_loader, parameters=parameters, **inference_parameters
        )
        self.__loss = nn.CrossEntropyLoss(reduction="mean")

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()
        inference_parameters.update({"regularizers": {"nll": None}})
        inference_parameters.update({"model_eval": "accuracy"})
        inference_parameters.update({"loss_type": "classification"})
        return inference_parameters

    def loss(self, x, y, data_loader, epoch):
        """
        nll [batch_size, max_lenght]

        """
        target = x["reward_bin"].long().to(self.device, non_blocking=True)
        l = self.__loss(y, target)
        accuracy = (torch.argmax(y, dim=1) == target).float().mean()
        return {"loss": l, "accuracy": accuracy}

    def metrics(
        self, data, forward_results, epoch, mode="evaluation", data_loader=None
    ):
        if mode == "validation":
            accuracy = (
                (
                    torch.argmax(forward_results, dim=1)
                    == data["reward_bin"].to(self.device, non_blocking=True)
                )
                .float()
                .mean()
            )
            return {"accuracy": accuracy}
        return {}


class NonSequentialRegression(NonSequentialModel):
    name_: str = "nonsequential_regressor"

    def __init__(self, model_dir=None, data_loader: ADataLoader = None, **kwargs):
        NonSequentialModel.__init__(
            self,
            model_name=self.name_,
            model_dir=model_dir,
            data_loader=data_loader,
            **kwargs,
        )

    def initialize_inference(
        self, data_loader: ADataLoader, parameters=None, **inference_parameters
    ) -> None:
        super().initialize_inference(
            data_loader=data_loader, parameters=parameters, **inference_parameters
        )

        self.__loss_mse = nn.MSELoss(reduction="sum")
        self.__loss_mae = nn.L1Loss(reduction="sum")

        if self.regression_dist == "normal":
            self.nll_regression = nn.GaussianNLLLoss(reduction="sum")
        elif self.regression_dist == "exp":
            self.nll_regression = exp_nllloss

    @classmethod
    def get_parameters(cls):
        parameters = super().get_parameters()
        parameters["regression_dist"] = "exp"  # exp or normal
        return parameters

    def set_parameters(self, **kwargs):
        super().set_parameters(**kwargs)
        self.regression_dist = kwargs.get("regression_dist", "exp")

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()
        inference_parameters.update({"regularizers": {"nll": None}})
        inference_parameters.update({"model_eval": "RMSE"})
        inference_parameters.update({"loss_type": "regression"})
        return inference_parameters

    def loss(self, x, y, data_loader, epoch):
        """
        nll [batch_size, max_lenght]

        """
        y = y.squeeze(1)
        target = x["reward"].float().to(self.device, non_blocking=True)
        batch_size = y.size(0)
        # nll_regression = self.nll_regression(y, target, torch.ones_like(target) * 0.0001)

        mse = self.__loss_mse(y, target)
        mae = self.__loss_mae(y, target)
        return {
            "loss": mse,
            "MAE": mae / batch_size,
            "RMSE": torch.sqrt(mse / batch_size),
        }

    def metrics(
        self, data, forward_results, epoch, mode="evaluation", data_loader=None
    ):
        if mode == "validation":
            return {}
        return {}


class SequentialModel(DeepBayesianModel):
    def __init__(
        self, model_dir=None, data_loader: ADataLoader = None, model_name=None, **kwargs
    ):
        DeepBayesianModel.__init__(
            self, model_name, model_dir=model_dir, data_loader=data_loader, **kwargs
        )

    @classmethod
    def get_parameters(cls):
        """
        here we provide an example of the minimum set of parameters requiered to instantiate the model
        """
        parameters_sample = {
            "backbone_name": "bert",
            "output_dim": 1,
            "dropout": 0.1,
            "cov_layers_dim": [10, 10],
            "layers_dim": [],
            "cov_emb_dim": 50,
            "covariates_dim": 0,
            "model_path": os.path.join(project_path, "results"),
        }

        return parameters_sample

    def set_parameters(self, **kwargs):
        self.backbone_name = kwargs.get("backbone_name")
        self.covariates_dim = kwargs.get("covariates_dim")
        self.train_backbone = kwargs.get("train_backbone", False)
        self.output_dim = kwargs.get("output_dim")
        self.dropout = kwargs.get("dropout")
        self.cov_layers_dim = kwargs.get("cov_layers_dim")
        self.layers_dim = kwargs.get("layers_dim")
        self.cov_emb_dim = kwargs.get("cov_emb_dim")

    def update_parameters(self, data_loader: ADataLoader, **kwargs):
        loss_type = self.get_inference_parameters()["loss_type"]
        kwargs.update(
            {
                "covariates_dim": data_loader.train.dataset.covariates_size,
                "output_dim": data_loader.number_of_reward_categories
                if loss_type == "classification"
                else 1,
            }
        )
        return kwargs

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()
        inference_parameters.update({"regularizers": {"nll": None}})
        inference_parameters.update({"model_eval": "RMSE"})

        return inference_parameters

    def define_deep_models(self):
        self.backbone: nn.Module = load_backbone(self.backbone_name)
        if not self.train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.dropout_layer = nn.Dropout(self.dropout)
        if self.covariates_dim == 0:
            self.out_layer = MLP(
                input_dim=768,
                output_dim=self.output_dim,
                layers_dim=self.layers_dim,
                dropout=self.dropout,
                ouput_transformation="exp",
            )

        else:
            self.cov2emb = MLP(
                input_dim=self.covariates_dim,
                output_dim=self.cov_emb_dim,
                layers_dim=self.cov_layers_dim,
                dropout=self.dropout,
                ouput_transformation=True,
            )
            self.out_layer = MLP(
                input_dim=768 + self.cov_emb_dim,
                output_dim=self.output_dim,
                layers_dim=self.layers_dim,
                dropout=self.dropout,
                ouput_transformation="exp",
            )
        self._forward_backbone = self.__forward_bert_albert
        if self.backbone_name == "roberta":
            self._forward_backbone = self.__forward_roberta

    def forward(self, x):
        text = x["text"]

        out_p = self._forward_backbone(x)
        emb = self.dropout_layer(out_p)
        if self.covariates_dim != 0:
            cov_emb = self.cov2emb(x["covariates"].to(self.device, non_blocking=True))
            emb = torch.cat((emb, cov_emb), dim=1)

        out = self.out_layer(emb)
        return out

    def __forward_bert_albert(self, x):
        text = x["text"]

        _, out_p = self.backbone(
            input_ids=text["input_ids"].squeeze(1).to(self.device, non_blocking=True),
            token_type_ids=text["token_type_ids"]
            .squeeze(1)
            .to(self.device, non_blocking=True),
            attention_mask=text["attention_mask"]
            .squeeze(1)
            .to(self.device, non_blocking=True),
            return_dict=False,
        )  # hidden, pooled
        return out_p

    def __forward_roberta(self, x):
        text = x["text"]

        _, out_p = self.backbone(
            input_ids=text["input_ids"].squeeze(1).to(self.device, non_blocking=True),
            attention_mask=text["attention_mask"]
            .squeeze(1)
            .to(self.device, non_blocking=True),
            return_dict=False,
        )  # hidden, pooled
        return out_p

    def initialize_inference(
        self, data_loader: ADataLoader, parameters=None, **inference_parameters
    ) -> None:
        super().initialize_inference(
            data_loader=data_loader, parameters=parameters, **inference_parameters
        )


class SequentialClassifier(SequentialModel):
    name_: str = "sequential_classification"

    def __init__(self, model_dir=None, data_loader: ADataLoader = None, **kwargs):
        super().__init__(
            model_name=self.name_,
            model_dir=model_dir,
            data_loader=data_loader,
            **kwargs,
        )

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()
        inference_parameters.update({"regularizers": {"nll": None}})
        inference_parameters.update({"model_eval": "accuracy"})
        inference_parameters.update({"loss_type": "classification"})

        return inference_parameters

    def initialize_inference(
        self, data_loader: ADataLoader, parameters=None, **inference_parameters
    ) -> None:
        super().initialize_inference(
            data_loader, parameters=parameters, **inference_parameters
        )
        self.__loss = nn.CrossEntropyLoss()

    def loss(self, x, y, data_loader, epoch):
        """
        nll [batch_size, max_lenght]

        """
        target = x["reward_bin"].long().to(self.device, non_blocking=True)
        v = self.__loss(y.squeeze(1), target)
        return {"loss": v}

    def metrics(
        self, data, forward_results, epoch, mode="evaluation", data_loader=None
    ):
        if mode == "validation":
            accuracy = (
                (
                    torch.argmax(forward_results, dim=1)
                    == data["reward_bin"].to(self.device, non_blocking=True)
                )
                .float()
                .mean()
            )
            return {"accuracy": accuracy}
        return {}


class SequentialRegression(SequentialModel):
    name_: str = "sequential_regression"

    def __init__(self, model_dir=None, data_loader: ADataLoader = None, **kwargs):
        super().__init__(
            model_name=self.name_,
            model_dir=model_dir,
            data_loader=data_loader,
            **kwargs,
        )

    def initialize_inference(
        self, data_loader: ADataLoader, parameters=None, **inference_parameters
    ) -> None:
        super().initialize_inference(
            data_loader=data_loader, parameters=parameters, **inference_parameters
        )

        self.__loss_mse = nn.MSELoss(reduction="sum")
        self.__loss_mae = nn.L1Loss(reduction="sum")
        if self.regression_dist == "normal":
            self.nll_regression = nn.GaussianNLLLoss(reduction="sum")
        elif self.regression_dist == "exp":
            self.nll_regression = exp_nllloss

    def set_parameters(self, **kwargs):
        super().set_parameters(**kwargs)
        self.regression_dist = kwargs.get("regression_dist", "exp")

    @classmethod
    def get_parameters(cls):
        parameters = super().get_parameters()
        parameters["regression_dist"] = "exp"  # exp or normal
        return parameters

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()

        inference_parameters.update({"model_eval": "RMSE"})
        inference_parameters.update({"loss_type": "regression"})
        return inference_parameters

    def loss(self, x, y, data_loader, epoch):
        """
        nll [batch_size, max_lenght]

        """
        y = y.squeeze(1)
        target = x["reward"].float().to(self.device, non_blocking=True)
        batch_size = y.size(0)

        mse = self.__loss_mse(y, target)
        mae = self.__loss_mae(y, target)
        return {
            "loss": mse,
            "MAE": mae / batch_size,
            "RMSE": torch.sqrt(mse / batch_size),
        }

    def metrics(
        self, data, forward_results, epoch, mode="evaluation", data_loader=None
    ):
        if mode == "validation":
            return {}
        return {}


if __name__ == "__main__":
    from deep_fields import data_path

    data_dir = os.path.join(
        data_path, "preprocessed", "reddit-19-20", "submissions", "language"
    )
    dataloader_params = {"path_to_data": data_dir, "batch_size": 128}

    data_loader = TopicDataloader("cpu", **dataloader_params)
    databatch = next(data_loader.train.__iter__())

    model_parameters = NonSequentialModel.get_parameters()
    model = NonSequentialModel(data_loader=data_loader, **model_parameters)
    forward_data = model(databatch)
    print(forward_data)
    loss = model.loss(databatch, forward_data, data_loader, 0)
    print(loss)
    del forward_data
    del loss

    # TEST BERT MODEL

    data_dir = os.path.join(
        data_path, "preprocessed", "reddit-19-20", "submissions", "language-transformer"
    )
    dataloader_params = {"path_to_data": data_dir, "batch_size": 2, "name": "bert"}
    data_loader = TopicTransformerLanguageDataloader("cpu", **dataloader_params)
    databatch = next(data_loader.train.__iter__())

    model_parameters = SequentialModel.get_parameters()
    model_parameters["backbone_name"] = "bert"
    model = SequentialModel(data_loader=data_loader, **model_parameters)
    forward_data = model(databatch)
    print(forward_data)
    loss = model.loss(databatch, forward_data, data_loader, 0)
    print(loss)
    del forward_data
    del loss

    # TEST ROBERTA MODEL

    data_dir = os.path.join(
        data_path, "preprocessed", "reddit-19-20", "submissions", "language-transformer"
    )
    dataloader_params = {"path_to_data": data_dir, "batch_size": 2, "name": "roberta"}
    data_loader = TopicTransformerLanguageDataloader("cpu", **dataloader_params)
    databatch = next(data_loader.train.__iter__())

    model_parameters = SequentialModel.get_parameters()
    model_parameters["backbone_name"] = "roberta"
    model = SequentialModel(data_loader=data_loader, **model_parameters)
    forward_data = model(databatch)
    print(forward_data)
    loss = model.loss(databatch, forward_data, data_loader, 0)
    print(loss)
    del forward_data
    del loss

    # TEST ALBERT MODEL

    data_dir = os.path.join(
        data_path, "preprocessed", "reddit-19-20", "submissions", "language-transformer"
    )
    dataloader_params = {"path_to_data": data_dir, "batch_size": 2, "name": "albert"}
    data_loader = TopicTransformerLanguageDataloader("cpu", **dataloader_params)
    databatch = next(data_loader.train.__iter__())

    model_parameters = SequentialModel.get_parameters()
    model_parameters["backbone_name"] = "albert"
    model = SequentialModel(data_loader=data_loader, **model_parameters)
    forward_data = model(databatch)
    print(forward_data)
    loss = model.loss(databatch, forward_data, data_loader, 0)
    print(loss)
    del forward_data
    del loss
