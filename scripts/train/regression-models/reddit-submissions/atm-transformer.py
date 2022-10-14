import os
import pprint

import click
import numpy as np
import torch
from deep_fields import data_path
from deep_fields.data.topic_models.dataloaders import TopicDataloader
from deep_fields.models.topic_models.supervised_static import \
    SupervisedTopicModel


@click.command()
@click.option("--dataset",
              type=click.Choice(["wallstreetbets", "politics", "the_donald", "askscience"], case_sensitive=False),
              default=None,
              help='Datasets')
@click.option("--small-ds", is_flag=True, help='Use small version of the dataset')
@click.option("--frequency", type=click.Choice(['daily', 'weekly', 'monthly'], case_sensitive=False), default='weekly')
@click.option("--transformer-type",
              type=click.Choice(['bert', 'albert', 'roberta'], case_sensitive=False),
              default='bert',
              help='Choose wich type of transformer you want to use.')
def train(dataset: str, small_ds: bool, frequency: str, transformer_type: str):
    rewards = [f"num-comments-{frequency}", f"score-{frequency}"]
    if small_ds:
        rewards = [f"num-comments-{frequency}-medium"]
    for reward_type in rewards:
        data_dir = os.path.join(data_path, "preprocessed", "reddit-50000", dataset, "submissions", "language", reward_type)
    for lr in [(0.001, 0.000001), (0.0005, 0.000001)]:
        for regressor_dim in [[256, 256]]:
            for enc_dim in [[256, 256]]:
                for emb_size in [128]:
                    for bs in [16, 32]:
                        for n_topics, delta in [(25, 0.2), (50, 0.1), (100, 0.05)]:
                            for alpha_y in [1, 1_00, 1_000, 10_000]:
                                dataloader_params = {
                                    "path_to_data": data_dir,
                                    "batch_size": bs,
                                    "use_covariates": False,
                                    "use_tmp_covariates": False,
                                    "normalize": False,
                                    "n_workers": 4,
                                    "reward_field": "reward_normalized",
                                    "transformer_name": transformer_type,
                                }  # "reward_normalized"

                                model_parameters = SupervisedTopicModel.get_parameters()

                                model_parameters.get("theta_q_parameters").update({
                                    "layers_dim": enc_dim,
                                    "output_dim": enc_dim[-1],
                                    "dropout": 0.1,
                                    "hidden_state_dim": emb_size,
                                    "out_dropout": 0.3
                                })

                                model_parameters.get("regression_head").update({
                                    "layers_dim": regressor_dim,
                                    "dropout": 0.3,
                                    "output_transformation": "exp"
                                })
                                model_parameters["llm"] = {"type": transformer_type, "train_lm": True}

                                model_parameters.update({
                                    "delta":
                                        delta,
                                    "word_embeddings_dim":
                                        300,
                                    "lambda_diversity":
                                        0.1,
                                    "no_topics":
                                        False,
                                    "no_embeddings":
                                        False,
                                    "number_of_topics":
                                        n_topics,
                                    "model_path": f"results-tmp/{dataset}/submissions/{reward_type}/regression"
                                })

                                inference_parameters = SupervisedTopicModel.get_inference_parameters()
                                inference_parameters.update({
                                    "optimizer_name": "AdamW",
                                    "learning_rate": lr[0],
                                    "cuda": 0,
                                    "alpha_y": alpha_y,
                                    # "clip_norm": 2.0,
                                    'number_of_epochs': 50,
                                    'metrics_log': 5,
                                    "tau": 0.75,
                                    "debug": False,
                                    "reduced_num_batches": 10,
                                    "min_lr_rate": 1e-12,
                                    "anneal_lr_after_epoch": 5,
                                    "is_loss_minimized": True,
                                    "gumbel": None,
                                    "optimizer_lm": {
                                        "name": "AdamW",
                                        "lr": lr[1],
                                        "weight_decay": 0.0000001
                                    }
                                })
                                inference_parameters.get("lr_scheduler").update({"counter": 5})
                                del inference_parameters["regularizers"]["alpha"]

                                experiment_name = f'TAM-{transformer_type.upper()}-lr-{"-".join(map(str, lr))}-bs-{bs}-nt-{n_topics}-regressor-dim-{"-".join(map(str, regressor_dim))}-enc-dim-{"-".join(map(str, enc_dim))}-emb-size-{emb_size}-alphay-{alpha_y}'

                                model_parameters.update({'experiment_name': experiment_name})

                                if os.path.exists(os.path.join(model_parameters["model_path"], SupervisedTopicModel.name_, experiment_name)):
                                    print(f"Skipping {experiment_name}")
                                    continue

                                data_loader = TopicDataloader('cpu', **dataloader_params)

                                try:
                                    np.random.seed(2021)
                                    torch.backends.cudnn.deterministic = True
                                    torch.manual_seed(2021)
                                    model = SupervisedTopicModel(data_loader=data_loader, **model_parameters)
                                    pprint.pprint(f'\nModel architecture: {model}')
                                    pprint.pprint(f'\nInference Parameters: {inference_parameters}')
                                    model.inference(data_loader, **inference_parameters)
                                except Exception as e:
                                    print(e)



if __name__ == "__main__":
    train()
