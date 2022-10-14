import os

import click
import numpy as np
import torch
from deep_fields import data_path
from deep_fields.data.topic_models.dataloaders import TopicDataloader
from deep_fields.models.topic_models.baseline_models import NonSequentialRegression


@click.command()
@click.option(
    "--dataset", type=click.Choice(["wallstreetbets", "politics", "the_donald", "askscience", "tifu"], case_sensitive=False), default=None, help="Datasets"
)
@click.option("--small-ds", is_flag=True, help="Use small version of the dataset")
@click.option("--frequency", type=click.Choice(["daily", "weekly", "monthly"], case_sensitive=False), default="weekly")
def train(dataset: str, small_ds: bool, frequency: str):
    rewards = [f"num-comments-{frequency}", f"score-{frequency}"]
    if small_ds:
        rewards = [f"num-comments-{frequency}-medium"]
    for reward_type in rewards:
        data_dir = os.path.join(data_path, "preprocessed", "reddit-50000", dataset, "submissions", "language", reward_type)
        for lr in [0.0005, 0.001]:
            for bs in [128, 32]:
                for enc_dim in [[256, 256]]:
                    for emb_size in [128]:
                        for layers_dim in [[], [256, 256]]:

                            dataloader_params = {
                                "path_to_data": data_dir,
                                "batch_size": bs,
                                "use_covariates": False,
                                "use_tmp_covariates": False,
                                "normalize": False,
                                "word_emb_type": "bow",
                                "n_workers": 4,
                                "reward_field": "reward_normalized",
                            }

                            model_parameters = NonSequentialRegression.get_parameters()
                            model_parameters = {
                                "layers_dim": layers_dim,
                                "bow_layers_dim": enc_dim,
                                "cov_layers_dim": enc_dim,
                                "bow_emb_dim": emb_size,
                                "cov_emb_dim": emb_size,
                                "regression_dist": "normal",
                                "dropout": 0.5,
                                "model_path": f"results-tmp/{dataset}/submissions/{reward_type}/regression",
                            }

                            model_inference_parameters = NonSequentialRegression.get_inference_parameters()

                            model_inference_parameters.update(
                                {
                                    "learning_rate": lr,
                                    "cuda": 0,
                                    "clip_norm": 2.0,
                                    "number_of_epochs": 200,
                                    "metrics_log": 5,
                                    "tau": 0.75,
                                    "debug": False,
                                    "reduced_num_batches": 10,
                                    "min_lr_rate": 1e-12,
                                    "anneal_lr_after_epoch": 50,
                                    "loss_type": "regression",
                                    "gumbel": None,
                                }
                            )
                            model_inference_parameters.get("lr_scheduler").update({"counter": 5})

                            experiment_name = (
                                f'layers-dim-{"-".join(map(str, layers_dim))}-enc-dim-{"-".join(map(str, enc_dim))}-lr-{lr}-bs-{bs}-emb-size-{emb_size}-exp'
                            )

                            model_parameters.update({"experiment_name": experiment_name})

                            if os.path.exists(os.path.join(model_parameters["model_path"], NonSequentialRegression.name_, experiment_name)):
                                print(f"Skipping {experiment_name}")
                                continue

                            data_loader = TopicDataloader("cpu", **dataloader_params)
                            try:
                                np.random.seed(2021)
                                torch.backends.cudnn.deterministic = True
                                torch.manual_seed(2021)
                                model = NonSequentialRegression(data_loader=data_loader, **model_parameters)
                                print("\nMLP architecture: {}".format(model))
                                model.inference(data_loader, **model_inference_parameters)
                            except Exception as e:
                                print(e)


if __name__ == "__main__":
    train()
