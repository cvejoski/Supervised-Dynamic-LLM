import json
import os

import torch
from deep_fields.data.topic_models.dataloaders import TopicDataloader
from deep_fields.models.topic_models import ModelFactory


def load_model(model_path, data_path, gpu):
    data_params = load_parameters(os.path.join(model_path, 'dataloaders_parameters.json'))
    inference_params = load_parameters(os.path.join(model_path, 'inference_parameters.json'))
    model_params = load_parameters(os.path.join(model_path, 'parameters.json'))
    data_params['path_to_data'] = data_path
    data_params['is_dynamic'] = True

    data_loader = load_data(data_path, data_params, gpu)
    model = load_trained_model(model_path, data_loader, gpu)
    experiment_name = build_experiment_name(model.model_name, data_params, inference_params, model_params)
    return data_loader, model, experiment_name


def build_experiment_name(model_name: str, data_parameters: dict, inference_parameters: dict, model_parameters: dict) -> str:
    if model_name == 'nonsequential_regressor':
        enc_dim = model_parameters.get("bow_layers_dim", [])
        layers_dim = model_parameters.get('layers_dim', [])
        emb_size = model_parameters.get("bow_emb_dim")
        model_name = f"MLP Regression class-dim {'-'.join(map(str, layers_dim))}"
    elif model_name == 'nonsequential_classifier':
        enc_dim = model_parameters.get("bow_layers_dim", [])
        layers_dim = model_parameters.get('layers_dim', [])
        emb_size = model_parameters.get("bow_emb_dim")
        model_name = f"MLP Classification class-dim {'-'.join(map(str, layers_dim))}"
    elif model_name == 'sequential_classification' or model_name == 'sequential_regression':
        model_name = model_parameters.get('backbone_name').upper()
        enc_dim = model_parameters.get("cov_layers_dim", [])
        layers_dim = model_parameters.get('layers_dim', [])
        emb_size = model_parameters.get("covemb_dim")
        model_name = f"{model_parameters.get('backbone_name').upper()} class-dim {'-'.join(map(str, layers_dim))}"
    elif model_name in ['regression_miao', 'classification_miao', 'supervised_static_vanilla_topic', 'supervised_dynamic_topic_model']:
        llm = 'no LM' if data_parameters.get('transformer_name', None) is None else data_parameters['transformer_name']
        layers_dim_classifier = model_parameters.get("theta_q_parameters").get('layers_dim_classifier', [])
        enc_dim = model_parameters.get("theta_q_parameters").get('layers_dim_zx', [])
        emb_size = model_parameters.get("theta_q_parameters").get("hidden_state_dim_zx", [])
        model_name = f'MIAO {llm.upper()} n-top-{model_parameters.get("number_of_topics")}-alphay-{inference_parameters.get("alpha_y", 1000)}-enc {"-".join(map(str, enc_dim))} class-dim {"-".join(map(str, layers_dim_classifier))}'
    elif model_name in ['classification_dte', 'regression_dte', 'supervised_dynamic_topic_model']:
        llm = 'no LM' if data_parameters['transformer_name'] is None else data_parameters['transformer_name']
        layers_dim_classifier = model_parameters.get("theta_q_parameters").get('layers_dim_classifier', [])
        enc_dim = model_parameters.get("theta_q_parameters").get('layers_dim_zx', [])
        emb_size = model_parameters.get("theta_q_parameters").get("hidden_state_dim_zy", [])
        model_name = f'DTE {llm.upper()} n-top-{model_parameters.get("number_of_topics")}-alphay-{inference_parameters.get("alpha_y", 1000)}-enc {"-".join(map(str, enc_dim))} class-dim {"-".join(map(str, layers_dim_classifier))}'
    elif model_name == 'discrete_latent_topic':
        model_name = f'Neural LDA{" Embeddings" if model_parameters.get("no_embeddings") else ""} n-top-{model_parameters.get("number_of_topics")}'
        return f"{model_name} bs {data_parameters['batch_size']} lr {inference_parameters['learning_rate']}"
    elif model_name == 'neural_dynamical_topic_embeddings':
        model_name = f'D-LDA n-top-{model_parameters.get("number_of_topics")}'
        return f"{model_name} bs {data_parameters['batch_size']} lr {inference_parameters['learning_rate']}"
    elif model_name == 'neural_dynamical_topic_embeddings_v2':
        model_name = f'D-LDA (V2) n-top-{model_parameters.get("number_of_topics")}'
        return f"{model_name} bs {data_parameters['batch_size']} lr {inference_parameters['learning_rate']}"
    else:
        raise ValueError(f'Unknown model name {model_name}')

    weight_decay = inference_parameters.get("weight_decay", -1)
    if data_parameters['use_covariates']:
        name = f"{model_name} cov "
        if data_parameters['use_tmp_covariates']:
            name += f"{'tmp' if data_parameters['use_tmp_covariates'] else ''} "
    else:
        name = f"{model_name}"
    return f"{name} bs {data_parameters['batch_size']} lr {inference_parameters['learning_rate']} enc-dim {'-'.join(map(str, enc_dim))} emb-size {emb_size} wd {weight_decay}"


def load_parameters(path):
    with open(path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    return params


def load_data(data_path, params, gpu):
    dataloader_params = {"path_to_data": data_path, **params}
    dataloader_params["batch_size"] = 500
    dataloader_params["n_workers"] = 4
    if gpu > -1:
        data_loader = TopicDataloader(f'cuda:{gpu}', **dataloader_params)
    else:
        data_loader = TopicDataloader('cpu', **dataloader_params)
    return data_loader


def load_trained_model(model_path: str, data_loader, gpu: str):
    if gpu > -1:
        state = torch.load(os.path.join(model_path, 'best_model.p'), map_location=f'cuda:{gpu}')
    else:
        state = torch.load(os.path.join(model_path, 'best_model.p'), map_location='cpu')

    model_name = state['model_name']
    del state

    model = ModelFactory.get_instance(model_type=model_name, **{'model_dir': model_path, 'data_loader': data_loader, 'evaluate_mode': True})

    model.inference_parameters['cuda'] = gpu if gpu > -1 else 'cpu'
    model.initialize_inference(data_loader, **model.inference_parameters)
    model.eval()
    model.to(model.device)

    return model


def rename_model(model_name):
    if model_name.startswith('layer'):
        return 'MLP'
    elif model_name.startswith('bert'):
        return 'BERT'
    elif model_name.startswith('albert'):
        return 'ALBERT'
    elif model_name.startswith('roberta'):
        return 'RoBERTa'
    elif model_name.startswith('D-ST'):
        return 'D-ST'
    elif model_name.startswith('TAM-GRU'):
        return 'TAM-GRU'
    elif model_name.startswith('TAM-BERT'):
        return 'TAM-BERT'
    elif model_name.startswith('TAM-ALBERT'):
        return 'TAM-ALBERT'
    elif model_name.startswith('TAM-ROBERTA'):
        return 'TAM-RoBERTa'
    elif model_name.startswith('D-TAM-BERT'):
        return 'D-TAM-BERT'
    elif model_name.startswith('D-TAM'):
        if '-D-EMB' in model_name:
            return 'D-TAM-GRU-DE'
        else:
            return 'D-TAM-GRU'
    else:
        raise Exception(f"Invalid Model: {model_name}!")
