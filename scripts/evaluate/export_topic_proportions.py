from collections import defaultdict
import pickle
import logging
import os
import csv

import sys
import click
from deep_fields.models.utils.eval import load_model

import torch
from tqdm import tqdm

import glob
import csv

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@click.command()
@click.option('--models-root-dir', type=click.Path(exists=True), required=True, help='Path to the trained model')
@click.option('--data-path', type=click.Path(exists=True), required=True, help='Path to the data directory')
@click.option('--gpu', type=int, required=True, default=0, help='Select the GPU number')
@click.option('--filter-models', type=click.File('r'), default=None, required=False, help='Path csv file containing filtered models')
def main(models_root_dir: click.Path, data_path: click.Path, gpu: int, filter_models: click.File):
    important_models = defaultdict(list)
    if filter_models is not None:
        reader = csv.reader(filter_models)
        next(reader)
        for row in reader:
            important_models[row[0]].append(row[-2])
    dataset = data_path.split(os.path.sep)[-4] + '-' + data_path.split(os.path.sep)[-3]

    export_topic_proportions(models_root_dir, data_path, gpu, important_models[dataset])


def export_topic_proportions(models_root_dir, data_path, gpu, filter_models):

    for model_path in tqdm(sorted(glob.glob(os.path.join(models_root_dir, '**', 'best_model.p'), recursive=True), key=os.path.getmtime)):
        model_path_splitted = model_path.split(os.sep)
        # print(filter_models)
        if (filter_models and not model_path_splitted[-3] in filter_models):
            continue
        model_path = os.path.split(model_path)[0]
        if gpu > -1:
            state = torch.load(os.path.join(model_path, 'best_model.p'), map_location=f'cuda:{gpu}')
        else:
            state = torch.load(os.path.join(model_path, 'best_model.p'), map_location=f'cpu')

        model_name = state['model_name']
        print(f"Exporting for model: {model_path_splitted[-3]}")
        if model_name not in ['supervised_dynamic_topic_model', 'neural_dynamical_topic_embeddings', 'neural_dynamical_topic_embeddings_v2']:
            continue
        data_loader, model, _ = load_model(model_path, data_path, gpu)
        model.eval()
        with torch.no_grad():
            logging.info("saving beta ...")
            beta = model.get_beta_eval().detach().cpu().numpy()
            pickle.dump(beta, open(os.path.join(model_path, 'beta.pkl'), 'bw'))

            logging.info("saving theta and b ...")
            theta = model.get_time_series(data_loader.train)
            pickle.dump((theta[0].detach().cpu().numpy(), theta[1].detach().cpu().numpy(), theta[2].detach().cpu().numpy()),
                        open(os.path.join(model_path, 'theta.pkl'), 'bw'))
            del theta


if __name__ == '__main__':
    main()
