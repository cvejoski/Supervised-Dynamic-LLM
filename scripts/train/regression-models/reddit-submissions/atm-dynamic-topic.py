import os
import pprint

import click
import numpy as np
import torch
from deep_fields import data_path
from deep_fields.data.topic_models.dataloaders import TopicDataloader
from deep_fields.models.topic_models.supervised_dynamic import \
    SupervisedDynamicTopicModel


@click.command()
@click.option("--dataset",
              type=click.Choice(["wallstreetbets", "politics", "the_donald", "askscience"], case_sensitive=False),
              default=None,
              help='Datasets')
@click.option("--small-ds", is_flag=True, help='Use small version of the dataset')
@click.option("--frequency", type=click.Choice(['daily', 'weekly', 'monthly'], case_sensitive=False), default='weekly')
def train(dataset: str, small_ds: bool, frequency: str):
    rewards = [f"num-comments-{frequency}", f"score-{frequency}"]
    if small_ds:
        rewards = [f"num-comments-{frequency}-medium"]
    for reward_type in rewards:
        data_dir = os.path.join(data_path, "preprocessed", "reddit-50000", dataset, "submissions", "language", reward_type)
        for lr in [(0.001, 0.0005), (0.001, 0.00005), (0.0001, 0.0005), (0.0001, 0.00005)]:
            for regressor_dim in [[256, 256]]:
                for enc_dim in [[256, 256]]:
                    for bs in [32, 128]:
                        for n_topics, delta in [(25, 0.2), (50, 0.1), (100, 0.05)]:
                            for alpha_y in [1, 100, 500, 1000]:
                                for nonlinear_transition in [True]:
                                    for topic_embeddings in ['static']:
                                        if nonlinear_transition:
                                            eta_infer_size = [[64, 64], [128, 128]]
                                        else:
                                            eta_infer_size = [[]]
                                        for eta_transition_size in eta_infer_size:
                                            dataloader_params = {
                                                "path_to_data": data_dir,
                                                "batch_size": bs,
                                                "use_covariates": False,
                                                "use_tmp_covariates": False,
                                                "normalize": False,
                                                "transformer_name": None,
                                                "is_dynamic": True,
                                                "n_workers": 4,
                                                "reward_field": "reward_normalized"
                                            }


                                            model_parameters = SupervisedDynamicTopicModel.get_parameters()

                                            inference_parameters = SupervisedDynamicTopicModel.get_inference_parameters()

                                            two_optimizers = False
                                            model_parameters.update({
                                                "two_optimizers": two_optimizers,
                                                "number_of_topics": n_topics,
                                                "word_embeddings_dim": 300,
                                                "nonlinear_transition_prior": nonlinear_transition,
                                                "topic_embeddings": topic_embeddings,
                                                "model_path": "./results-tmp/reddit/regression/dynamic_supervised",
                                            })

                                            model_parameters.get("eta_prior_transition").update({
                                                "layers_dim": eta_transition_size,
                                                "dropout": 0.5,
                                                "out_dropout": 0.1
                                            })

                                            model_parameters.get("theta_q_parameters").update({
                                                "layers_dim": enc_dim,
                                                "output_dim": enc_dim[-1],
                                                "dropout": 0.3,
                                                "hidden_state_dim": 128,
                                                "out_dropout": 0.1
                                            })

                                            model_parameters.get("eta_q_parameters").update({
                                                "hidden_state_transition_dim": 400,
                                                "layers_dim": 400,
                                                "num_rnn_layers": 4,
                                                "dropout": 0.3,
                                                "out_dropout": 0.1
                                            })

                                            model_parameters.get("regression_head").update({
                                                "layers_dim": regressor_dim,
                                                "dropout": 0.3,
                                                "output_transformation": "exp"
                                            })

                                            model_parameters["llm"] = {
                                                "type": "GRU",
                                                "hidden_size": 128,
                                                "num_layers": 1,
                                                "dropout": 0.0,
                                                "train_word_emb": True
                                            }

                                            inference_parameters.update({
                                                "learning_rate": lr[0],
                                                "cuda": 0,
                                                "delta": delta,
                                                "clip_norm": 2.0,
                                                'number_of_epochs': 200,
                                                'metrics_log': 1,
                                                "tau": 0.75,
                                                "debug": False,
                                                "alpha_y": alpha_y,
                                                "reduced_num_batches": 10,
                                                "min_lr_rate": 1e-12,
                                                "anneal_lr_after_epoch": 25,
                                                "is_loss_minimized": True,
                                                "gumbel": None,
                                                "optimizer_lm": {
                                                    "name": "Adam",
                                                    "lr": lr[1],
                                                    "weight_decay": 0.0000001
                                                }
                                            })
                                            inference_parameters.get("lr_scheduler").update({"counter": 5})
                                            del inference_parameters["regularizers"]["alpha"]
                                            # inference_parameters.get("regularizers").get("alpha").get("args").update({"max_value": alpha_y})

                                            experiment_name = f'D-TAM{"-NL" if nonlinear_transition  else ""}{"-D-EMB" if topic_embeddings == "dynamic" else ""}' \
                                                f'-lr-{lr if two_optimizers else lr[0]}-bs-{bs}-nt-{n_topics}-regression-dim-{"-".join(map(str, regressor_dim))}' \
                                                f'-enc-dim-{"-".join(map(str, enc_dim))}-eta-trans-{"-".join(map(str, eta_transition_size))}-alphay-{alpha_y}'
                                            # experiment_hash = hashlib.sha224(experiment_name.encode()).hexdigest()
                                            model_parameters.update({'experiment_name': experiment_name})

                                            if os.path.exists(
                                                    os.path.join(model_parameters["model_path"], SupervisedDynamicTopicModel.name_, experiment_name)):
                                                print(f"Skipping {experiment_name}")
                                                continue

                                            data_loader = TopicDataloader('cpu', **dataloader_params)
                                            hyperparams = {
                                                'n_topics': n_topics,
                                                'enc_dim': "-".join(map(str, enc_dim)),
                                                'eta_transition_size': "-".join(map(str, eta_transition_size)),
                                                'regression_dim': "-".join(map(str, regressor_dim)),
                                                'lr': lr,
                                                'bs': bs,
                                                'two_optimizers': two_optimizers,
                                                'alpha_y': alpha_y,
                                                'nonlinear_transition': nonlinear_transition,
                                                'topic_embeddings': topic_embeddings
                                            }
                                            try:
                                                np.random.seed(2021)
                                                torch.backends.cudnn.deterministic = True
                                                torch.manual_seed(2021)
                                                model = SupervisedDynamicTopicModel(data_loader=data_loader, **model_parameters)
                                                pprint.pprint(f'Model architecture: {model}')
                                                pprint.pprint(f'Inference Parameters: {inference_parameters}')
                                                model.inference(data_loader, hyperparams, **inference_parameters)
                                            except Exception as e:
                                                print(e)



if __name__ == "__main__":
    train()
