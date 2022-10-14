import os

import click
import numpy as np
import torch
from deep_fields import data_path
from deep_fields.data.topic_models.dataloaders import TopicDataloader
from deep_fields.models.topic_models.dynamic import DynamicTopicEmbeddings


@click.command()
@click.option(
    "--dataset", type=click.Choice(["wallstreetbets", "politics", "the_donald", "askscience", "tifu"], case_sensitive=False), default=None, help="Datasets"
)
@click.option("--small-ds", is_flag=True, help="Use small version of the dataset")
@click.option("--frequency", type=click.Choice(["daily", "weekly", "monthly"], case_sensitive=False), default="weekly")
def train(dataset: str, small_ds: bool, frequency: str):
    rewards = [f"num-comments-{frequency}"]
    if small_ds:
        rewards = [f"num-comments-{frequency}-medium"]
    for reward_type in rewards:
        data_dir = os.path.join(data_path, "preprocessed", "reddit-50000", dataset, "submissions", "language", reward_type)
        for n_topics in [25, 50, 100]:
            for enc_dim in [[400]]:
                for lr in [0.0008, 0.001, 0.0001]:
                    for bs in [100, 200]:
                        for nonlinear_transition in [True]:
                            for topic_embeddings in ["static"]:
                                if nonlinear_transition:
                                    eta_infer_size = [[64, 64]]
                                else:
                                    eta_infer_size = [[]]
                                for eta_transition_size in eta_infer_size:
                                    dataloader_params = {"path_to_data": data_dir, "batch_size": bs, "is_dynamic": True, "n_workers": 4}

                                    model_parameters = DynamicTopicEmbeddings.get_parameters()
                                    ndt_inference_parameters = DynamicTopicEmbeddings.get_inference_parameters()

                                    ndt_inference_parameters.update(
                                        {
                                            "learning_rate": lr,
                                            "cuda": 0,
                                            "clip_norm": 2.0,
                                            "number_of_epochs": 1000,
                                            "metrics_log": 10,
                                            "tau": 0.75,
                                            "min_lr_rate": 1e-12,
                                            "anneal_lr_after_epoch": 100,
                                            "gumbel": None,
                                        }
                                    )
                                    ndt_inference_parameters.get("lr_scheduler").update({"counter": 5})
                                    experiment_name = (
                                        f'DTEM{"-NL" if nonlinear_transition else ""}{"-D-EMB" if topic_embeddings == "dynamic" else ""}'
                                        f'-{lr}-bs-{bs}-nt-{n_topics}-enc-dim-{"-".join(map(str, enc_dim))}'
                                        f'-eta-trans-{"-".join(map(str, eta_transition_size))}'
                                    )

                                    model_parameters.update(
                                        {
                                            "number_of_topics": n_topics,
                                            "word_embeddings_dim": 300,
                                            "nonlinear_transition_prior": nonlinear_transition,
                                            "topic_embeddings": topic_embeddings,
                                            "model_path": "./results-tmp/reddit/topic/dynamic_supervised/{dataset}/submissions/{reward_type}/topic",
                                            "experiment_name": experiment_name,
                                        }
                                    )

                                    model_parameters.get("eta_prior_transition").update({"layers_dim": eta_transition_size, "dropout": 0.5, "out_dropout": 0.1})

                                    model_parameters.get("theta_q_parameters").update(
                                        {"layers_dim": enc_dim, "output_dim": enc_dim[-1], "hidden_state_dim": 128, "dropout": 0.0, "out_dropout": 0.1}
                                    )

                                    model_parameters.get("eta_q_parameters").update(
                                        {"hidden_state_transition_dim": 400, "layers_dim": 400, "num_rnn_layers": 4, "dropout": 0.0, "out_dropout": 0.1}
                                    )
                                    if os.path.exists(os.path.join(model_parameters["model_path"], DynamicTopicEmbeddings.name_, experiment_name)):
                                        print(f"Skipping {experiment_name}")
                                        continue

                                    data_loader = TopicDataloader("cpu", **dataloader_params)
                                    hyperparams = {
                                        "n_topics": n_topics,
                                        "enc_dim": "-".join(map(str, enc_dim)),
                                        "eta_transition_size": "-".join(map(str, eta_transition_size)),
                                        "lr": lr,
                                        "bs": bs,
                                        "nonlinear_transition": nonlinear_transition,
                                        "topic_embeddings": topic_embeddings,
                                    }
                                    try:
                                        np.random.seed(2021)
                                        torch.backends.cudnn.deterministic = True
                                        torch.manual_seed(2021)
                                        model = DynamicTopicEmbeddings(data_loader=data_loader, **model_parameters)
                                        print(f"\nDETM architecture: {model}")
                                        model.inference(data_loader, hyperparams, **ndt_inference_parameters)
                                    except Exception as e:
                                        print(e)



if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    train()
