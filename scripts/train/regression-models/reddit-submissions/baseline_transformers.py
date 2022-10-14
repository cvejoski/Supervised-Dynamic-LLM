import os
from deep_fields.models.topic_models.baseline_models import SequentialRegression
import numpy as np
import torch
from deep_fields import data_path
from deep_fields.data.topic_models.dataloaders import TopicDataloader
import click


@click.command()
@click.option(
    "--transformer-type",
    type=click.Choice(["bert", "albert", "roberta"], case_sensitive=False),
    default="bert",
    help="Choose wich type of transformer you want to use.",
)
@click.option(
    "--dataset", type=click.Choice(["wallstreetbets", "politics", "the_donald", "askscience", "tifu"], case_sensitive=False), default=None, help="Datasets"
)
@click.option("--small-ds", is_flag=True, help="Use small version of the dataset")
@click.option("--frequency", type=click.Choice(["daily", "weekly", "monthly"], case_sensitive=False), default="weekly")
def train(transformer_type: str, dataset: str, small_ds: bool, frequency: str):
    rewards = [f"num-comments-{frequency}", f"score-{frequency}"]
    if small_ds:
        rewards = [f"num-comments-{frequency}-medium"]
    for reward_type in rewards:
        data_dir = os.path.join(data_path, "preprocessed", "reddit-50000", dataset, "submissions", "language", reward_type)
        for lr in [0.00001, 0.000001]:
            for bs in [8, 16]:
                for regression_head in [[], [256, 256]]:
                    dataloader_params = {
                        "path_to_data": data_dir,
                        "batch_size": bs,
                        "use_covariates": False,
                        "use_tmp_covariates": False,
                        "normalize": False,
                        "transformer_name": transformer_type,
                        "n_workers": 4,
                        "reward_field": "reward_normalized",
                    }

                    model_parameters = SequentialRegression.get_parameters()
                    model_parameters["backbone_name"] = transformer_type
                    model_parameters["train_backbone"] = True
                    model_parameters["model_path"] = f"./results-tmp/{dataset}/submissions/{reward_type}/regression"
                    model_inference_parameters = SequentialRegression.get_inference_parameters()

                    model_parameters["layers_dim"] = regression_head
                    model_parameters["dropout"] = 0.3

                    model_inference_parameters.update(
                        {
                            "learning_rate": lr,
                            "cuda": 0,
                            # "clip_norm": 2.0,
                            "number_of_epochs": 20,
                            "metrics_log": 5,
                            "tau": 0.75,
                            "debug": False,
                            "reduced_num_batches": 2,
                            "min_lr_rate": 1e-12,
                            "anneal_lr_after_epoch": 1,
                            "gumbel": None,
                        }
                    )
                    model_inference_parameters.get("lr_scheduler").update({"counter": 2})
                    experiment_name = f'{transformer_type}-lr-{lr}-bs-{bs}-layers-dim-{"-".join(map(str, regression_head))}'
                    model_parameters.update({"experiment_name": experiment_name})
                    if os.path.exists(os.path.join(model_parameters["model_path"], SequentialRegression.name_, experiment_name)):
                        print(f"Skipping {experiment_name}")
                        continue
                    data_loader = TopicDataloader("cpu", **dataloader_params)
                    try:
                        np.random.seed(2021)
                        torch.backends.cudnn.deterministic = True
                        torch.manual_seed(2021)
                        model = SequentialRegression(data_loader=data_loader, **model_parameters)
                        click.echo("\nModel architecture: {}".format(model))
                        model.inference(data_loader, **model_inference_parameters)
                    except Exception as e:
                        click.echo(e)


if __name__ == "__main__":
    train()
