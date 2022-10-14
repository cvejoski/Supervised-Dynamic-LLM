import os
from deep_fields.models.topic_models.supervised_static import SupervisedTopicModel
import numpy as np
import torch
from deep_fields import data_path
from deep_fields.data.topic_models.dataloaders import TopicDataloader
import pprint
import click


@click.command()
@click.option(
    "--dataset", type=click.Choice(["wallstreetbets", "politics", "the_donald", "askscience", "tifu"], case_sensitive=False), default=None, help="Datasets"
)
@click.option("--small-ds", is_flag=True, help="Use small version of the dataset")
@click.option("--frequency", type=click.Choice(["daily", "weekly", "monthly"], case_sensitive=False), default="weekly")
def train(dataset: str, small_ds: bool, frequency: str):
    rewards = [f"num-comments-{frequency}", f"score-{frequency}"]
    if small_ds:
        rewards = [f"num-comments-{frequency}-medium", f"score-{frequency}-medium"]
    for reward_type in rewards:
        data_dir = os.path.join(data_path, "preprocessed", "reddit", dataset, "submissions", "language", reward_type)
        for lr in [0.0005, 0.001]:
            for regressor_dim in [[], [256, 256]]:
                for enc_dim in [[64, 64], [256, 256]]:
                    for emb_size in [128]:
                        for bs in [32, 128]:
                            for n_topics in [25, 50, 100]:
                                for alpha_y in [5000]:

                                    dataloader_params = {
                                        "path_to_data": data_dir,
                                        "batch_size": bs,
                                        "use_covariates": False,
                                        "use_tmp_covariates": False,
                                        "normalize": False,
                                        "n_workers": 4,
                                        "reward_field": "reward_normalized",
                                    }  # "reward_normalized"

                                    model_parameters = SupervisedTopicModel.get_parameters()

                                    model_parameters.get("theta_q_parameters").update(
                                        {"layers_dim": enc_dim, "output_dim": enc_dim[-1], "dropout": 0.3, "hidden_state_dim": emb_size, "out_dropout": 0.1}
                                    )
                                    model_parameters.get("covariates_encoder").update(
                                        {"layers_dim": enc_dim, "output_dim": enc_dim[-1], "dropout": 0.3, "hidden_state_dim": emb_size}
                                    )
                                    model_parameters.get("regression_head").update(
                                        {"layers_dim": regressor_dim, "dropout": 0.3, "output_transformation": "relu"}
                                    )
                                    model_parameters.update(
                                        {
                                            "word_embeddings_dim": 300,
                                            "lambda_diversity": 0.1,
                                            "no_topics": False,
                                            "no_embeddings": False,
                                            "number_of_topics": n_topics,
                                            "model_path": "./results-tmp/reddit/{dataset}/submissions/{reward_type}/regression",
                                        }
                                    )

                                    inference_parameters = SupervisedTopicModel.get_inference_parameters()
                                    inference_parameters.update(
                                        {
                                            "learning_rate": lr,
                                            "cuda": 0,
                                            "alpha_y": alpha_y,
                                            "clip_norm": 2.0,
                                            "number_of_epochs": 500,
                                            "metrics_log": 5,
                                            "tau": 0.75,
                                            "debug": False,
                                            "reduced_num_batches": 10,
                                            "min_lr_rate": 1e-12,
                                            "anneal_lr_after_epoch": 50,
                                            "is_loss_minimized": True,
                                            "gumbel": None,
                                        }
                                    )
                                    inference_parameters.get("lr_scheduler").update({"counter": 5})
                                    del inference_parameters["regularizers"]["alpha"]

                                    experiment_name = f'ST-lr-{lr}-bs-{bs}-nt-{n_topics}-regressor-dim-{"-".join(map(str, regressor_dim))}-enc-dim-{"-".join(map(str, enc_dim))}-emb-size-{emb_size}-alphay-{alpha_y}'
                                    model_parameters.update({"experiment_name": experiment_name})

                                    if os.path.exists(os.path.join(model_parameters["model_path"], SupervisedTopicModel.name_, experiment_name)):
                                        print(f"Skipping {experiment_name}")
                                        continue

                                    data_loader = TopicDataloader("cpu", **dataloader_params)

                                    try:
                                        np.random.seed(2021)
                                        torch.backends.cudnn.deterministic = True
                                        torch.manual_seed(2021)
                                        model = SupervisedTopicModel(data_loader=data_loader, **model_parameters)
                                        pprint.pprint("\nModel architecture: {}".format(model))
                                        pprint.pprint("\nInference Parameters: {}".format(inference_parameters))
                                        model.inference(data_loader, **inference_parameters)
                                    except Exception as e:
                                        print(e)


if __name__ == "__main__":
    train()
