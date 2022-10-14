import csv
import glob
import json
import logging
import os
import sys
from collections import defaultdict

import click
import numpy as np
import pandas as pd
import torch
from deep_fields.models.utils.eval import load_model
from deep_fields.utils.loss_utils import (topic_coherence_dynamic, topic_coherence_static)
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error, median_absolute_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from tqdm import tqdm

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@click.command()
@click.option('--models-root-dir', type=click.Path(exists=True), required=True, help='Path to the trained model')
@click.option('--data-path', type=click.Path(exists=True), required=True, help='Path to the data directory')
@click.option('--results-output', type=click.Path(exists=False), required=True, help='Path to the data directory')
@click.option('--target-type', type=click.Choice(['regression', 'classification', 'topic-model'], case_sensitive=False), default='regression')
@click.option('--gpu', type=int, required=True, default=0, help='Select the GPU number')
@click.option('--split', type=click.Choice(['train', 'validate', 'test', 'predict'], case_sensitive=False), default='test')
@click.option('--filter-models', type=click.File('r'), default=None, required=False, help='Path csv file containing filtered models')
def main(models_root_dir: click.Path, data_path: click.Path, results_output: click.Path, target_type: str, gpu: int, split: str,
         filter_models: click.File):
    # torch.multiprocessing.set_start_method('spawn')
    out_dir = os.path.split(results_output)[0]
    os.makedirs(out_dir, exist_ok=True)
    header = ['id', 'name']
    if target_type == 'classification':
        header = header + ['aroc', 'acc', 'prec', 'recall', 'f1'] + ['PPL-Blei']
        evaluate_model = evaluate_classification_model
    elif target_type == 'regression':
        header = header + ['r2', 'rmse', 'mae', 'meae'] + ['PPL-Blei']
        evaluate_model = evaluate_regression_model
    elif target_type == 'topic-model':
        header += ['PPL-Blei', 'Epoch', 'PPL-Train']
        evaluate_model = evaluate_topic_model

    if split != 'predict':
        header += ['TC', 'TD', 'TQ']

    important_models = defaultdict(list)
    if filter_models is not None:
        reader = csv.reader(filter_models)
        next(reader)
        for row in reader:
            important_models[row[0]].append(row[-2])
        dataset = data_path.split(os.path.sep)[-4] + '-' + data_path.split(os.path.sep)[-3]
        filter_models = important_models[dataset]
    # print(data_path.split(os.path.sep))
    # print(dataset, important_models[dataset])

    is_out_exist = os.path.isfile(results_output)
    if split in ['train', 'validate', 'test']:
        export_test(models_root_dir, data_path, results_output, gpu, split, out_dir, header, evaluate_model, is_out_exist)
    elif split == 'predict':
        export_predict(models_root_dir, data_path, results_output, gpu, split, header, evaluate_model, is_out_exist, filter_models)


def export_predict(models_root_dir, data_path, results_output, gpu, split, header, evaluate_model, is_out_exist, filter_models):
    with open(results_output, 'a', newline='', encoding='utf-8') as f:
        header += ['time_steps']
        csv_writer = csv.DictWriter(f, fieldnames=header)
        if is_out_exist:
            finished_evals = pd.read_csv(results_output)
            evaluated_models_id = finished_evals['id'].values.tolist()
        else:
            csv_writer.writeheader()
            f.flush()
            finished_evals = pd.DataFrame(columns=header)
            evaluated_models_id = []

        for model_path in tqdm(sorted(glob.glob(os.path.join(models_root_dir, '**', 'best_model.p'), recursive=True), key=os.path.getmtime)):
            model_path_splitted = model_path.split(os.sep)
            if model_path_splitted[-3] in evaluated_models_id or (filter_models is not None and not model_path_splitted[-3] in filter_models) or 'albert' in model_path_splitted[-3].lower():
                continue
            print(f"Evaluating model {model_path_splitted[-3]}")
            model_path = os.path.split(model_path)[0]
            last_log = read_train_log(model_path)

            model_results = evaluate_model(model_path, data_path, gpu, split)
            model_results["Epoch"] = last_log["epoch"]
            model_results["PPL-Train"] = last_log["best_eval_criteria"]
            model_results["time_steps"] = 0
            keys = list(model_results.keys())
            for k in keys:
                if k not in header:
                    del model_results[k]

            csv_writer.writerow(model_results)

            f.flush()


def read_train_log(model_path: str) -> int:
    with open(os.path.join(model_path, "inference_results.json"), "rb") as f:
        try:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
    return json.loads(last_line)


def export_test(models_root_dir, data_path, results_output, gpu, split, out_dir, header, evaluate_model, is_out_exist):
    with open(results_output, 'a', newline='', encoding='utf-8') as f:
        csv_writer = csv.DictWriter(f, fieldnames=header)
        if is_out_exist:
            finished_evals = pd.read_csv(results_output)
            evaluated_models_id = finished_evals['id'].values.tolist()
        else:
            csv_writer.writeheader()
            f.flush()
            finished_evals = pd.DataFrame(columns=header)
            evaluated_models_id = []

        for model_path in tqdm(sorted(glob.glob(os.path.join(models_root_dir, '**', 'best_model.p'), recursive=True), key=os.path.getmtime)):
            model_path_splitted = model_path.split(os.sep)
            if model_path_splitted[-3] in evaluated_models_id or 'albert' in model_path_splitted[-3].lower():
                continue
            print(f"Evaluating model {model_path_splitted[-3]}")
            model_path = os.path.split(model_path)[0]
            last_log = read_train_log(model_path)

            model_results = evaluate_model(model_path, data_path, gpu, split)
            model_results["Epoch"] = last_log["epoch"]
            model_results["PPL-Train"] = last_log["best_eval_criteria"]
            if model_results['top_words'] is not None:
                file_name = os.path.split(os.path.splitext(results_output)[0])[-1]
                l = os.path.join(out_dir, file_name)
                os.makedirs(l, exist_ok=True)
                with open(os.path.join(l, model_results['id'] + ".csv"), 'a', newline='', encoding='utf-8') as f_topic:
                    for topics in model_results['top_words']:
                        f_topic.write(topics)
                        f_topic.write("\n")

            keys = list(model_results.keys())
            for k in keys:
                if k not in header:
                    del model_results[k]

            csv_writer.writerow(model_results)
            f.flush()


def evaluate_topic_model(model_path: str, data_path: str, gpu: int, split: str):
    data_loader, model, experiment_name = load_model(model_path, data_path, gpu)
    dataset = getattr(data_loader, split)
    ppl, td, tc, tq, top_words = evaluate_topic_model_dataset(model, dataset, data_loader, split)
    return {'id': model_path.split(os.sep)[-2], 'name': experiment_name, 'PPL-Blei': ppl, 'TC': tc, 'TD': td, 'TQ': tq, 'top_words': top_words}


def evaluate_regression_model(model_path: str, data_path: str, gpu: int, split: str):
    data_loader, model, experiment_name = load_model(model_path, data_path, gpu)
    dataset = getattr(data_loader, split)
    r2, rmse, mae, meae, ppl, td, tc, tq, top_words, time_steps = evaluate_regression_model_dataset(model, dataset, data_loader, split)
    return {
        'id': model_path.split(os.sep)[-2],
        'name': experiment_name,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'meae': meae,
        'PPL-Blei': ppl,
        'TC': tc,
        'TD': td,
        'TQ': tq,
        'top_words': top_words,
        'time_steps': time_steps
    }


def evaluate_classification_model(model_path: str, data_path: str, gpu: int, split: str):
    data_loader, model, experiment_name = load_model(model_path, data_path, gpu)
    dataset = getattr(data_loader, split)
    aroc, acc, prec, recall, f1, ppl, td, tc, tq, top_words = evaluate_classification_model_dataset(model, dataset, data_loader, split)
    return {
        'id': model_path.split(os.sep)[-2],
        'name': experiment_name,
        'aroc': aroc,
        'acc': acc,
        'prec': prec,
        'recall': recall,
        'f1': f1,
        'PPL-Blei': ppl,
        'TC': tc,
        'TD': td,
        'TQ': tq,
        'top_words': top_words
    }


def evaluate_topic_model_dataset(model, data, data_loader, dataset_type):
    logging.info("%s Evaluating %s dataset %s", '=' * 10, dataset_type, '=' * 10)
    with torch.no_grad():
        if dataset_type == 'predict':
            log_ppl = model.prediction(data_loader, 50)
            td, tc, tq = 0, 0, 0
            top_words = None
        else:
            log_ppl = []
            for x in tqdm(data):
                forward_results = model.forward(x)
                loss_stats = model.loss(x, forward_results, data_loader, 0)
                log_ppl.append(loss_stats['Log-Likelihood'].item())
            td = model.topic_diversity()

            if model.topic_embeddings == 'static':
                top_word_per_topic, important_words_id = model.top_words(data_loader, num_of_top_words=40)
                important_words_id = important_words_id.detach().cpu().numpy()
                tc = topic_coherence_static(important_words_id[:, :10], data_loader.train.dataset.data['bow'], data_loader.vocab.word_count)
                top_words = ["TOPIC {0}: ".format(j) + " ".join(top_word_per_topic[f'TOPIC {j}']) + "\n" for j in range(len(top_word_per_topic))]
            else:
                top_word_per_topic, important_words_id = model.top_words_dynamic(data_loader, num_of_top_words=40)
                top_words = []
                for topic, value in top_word_per_topic.items():
                    for time, words in value.items():
                        top_words.append(f'{topic} --- {time}: {" ".join(words)}')
                tc = topic_coherence_dynamic(important_words_id[:, :, :10], data_loader.train.dataset.data['bow'], data_loader.vocab.word_count, 10)
            tq = td * tc
    ppl = np.exp(np.mean(log_ppl))
    return ppl, td, tc, tq, top_words


def evaluate_classification_model_dataset(model, data, data_loader, dataset_type):
    with torch.no_grad():
        logging.info("%s Evaluating %s dataset %s", '=' * 10, dataset_type, '=' * 10)
        y_hat, y_hat_c, y, ppl = [], [], [], []
        if model.name_ == 'classification_dte' and dataset_type == 'predict':
            y, y_hat_c, y_hat = model.prediction(data_loader, 50)
        else:
            for x in tqdm(data):
                if model.name_ in ['classification_miao', 'classification_dte']:
                    forward_results = model(x)
                    y_predict = forward_results[-1]
                    loss_stats = model.loss(x, forward_results, data_loader, 0)
                    ppl.append(loss_stats['Log-Likelihood'].item())
                else:
                    y_predict = model(x)
                y_predict = torch.softmax(y_predict, dim=-1).cpu().detach().numpy()
                y_predict_class = np.argmax(y_predict, axis=-1)
                y_target = x['reward_bin'].cpu().numpy()
                y.append(y_target)
                y_hat.append(y_predict)
                y_hat_c.append(y_predict_class)
            y_hat = np.concatenate(y_hat)
            y_hat_c = np.concatenate(y_hat_c)
            y = np.concatenate(y)
        accuracy = accuracy_score(y, y_hat_c)
        precision = precision_score(y, y_hat_c, average='weighted', zero_division=0)
        recall = recall_score(y, y_hat_c, average='weighted', zero_division=0)
        f1 = f1_score(y, y_hat_c, average='weighted', zero_division=0)
        aroc = roc_auc_score(y, y_hat, average='weighted', multi_class='ovr')
        if model.name_ in ['classification_miao', 'classification_dte'] and dataset_type == 'test':
            td = model.topic_diversity()
            tc = model.topic_coherence(data_loader.train.dataset.data['bow'])
            tq = td * tc
            if 'dte' in model.name_:
                top_word_per_topic = model.top_words(data_loader, num_of_top_words=40, num_of_time_steps=40)
                top_words = []
                for topic, value in top_word_per_topic.items():
                    for time, words in value.items():
                        top_words.append(f'{topic} --- {time}: {" ".join(words)}\n\n')
                    top_words.append("*" * 50 + "\n")
            else:
                top_word_per_topic = model.top_words(data_loader, num_of_top_words=40)
                top_words = ["TOPIC {0}: ".format(j) + " ".join(top_word_per_topic[j]) + "\n" for j in range(len(top_word_per_topic))]
        else:
            ppl, td, tc, tq = 0, 0, 0, 0
            top_words = None
        return aroc, accuracy, precision, recall, f1, np.exp(np.mean(ppl)), td, tc, tq, top_words


def evaluate_regression_model_dataset(model, data, data_loader, dataset_type):
    logging.info("%s Evaluating %s dataset %s", '=' * 10, dataset_type, '=' * 10)
    model.eval()
    with torch.no_grad():
        _rmse, _mae, _meae, _r2, _ppl = [], [], [], [], []
        repeat = 5
        if dataset_type == 'predict':
            repeat = 1
        for _ in range(repeat):
            y, y_hat, ppl, time = [], [], [], []
            time_steps = 0
            if model.name_ in ['supervised_dynamic_topic_model'] and dataset_type == 'predict':
                y, y_hat, time, ppl = model.prediction(data_loader, 50)
            else:
                for x in tqdm(data):
                    if model.name_ in ['regression_miao', 'regression_dte', 'supervised_static_vanilla_topic', 'supervised_dynamic_topic_model']:
                        forward_results = model(x)
                        y_predict = forward_results["y_hat"].cpu().numpy().flatten()
                        loss_stats = model.loss(x, forward_results, data_loader, 0)
                        ppl.append(loss_stats['Log-Likelihood'].item())
                    else:
                        y_predict = model(x).cpu().numpy().flatten()
                        ppl.append(0)
                    y_target = x['reward'].cpu().numpy()
                    # print(y_predict.sum(), y_target.sum(), y_target.shape)
                    y.append(y_target)
                    y_hat.append(y_predict)
                    time.append(x['time'].cpu().numpy().flatten())
                y = np.concatenate(y)
                y_hat = np.concatenate(y_hat)
                time = np.concatenate(time)

            if dataset_type == 'predict':
                rmse, mae, meae, r2 = [], [], [], []
                time_steps = np.sort(np.unique(time)).tolist()
                for t in time_steps:
                    ix = t == time
                    rmse.append(mean_squared_error(y[ix], y_hat[ix], squared=False))
                    mae.append(mean_absolute_error(y[ix], y_hat[ix]))
                    meae.append(median_absolute_error(y[ix], y_hat[ix]))
                    r2.append(r2_score(y[ix], y_hat[ix]))
            else:
                rmse = mean_squared_error(y, y_hat, squared=False)
                mae = mean_absolute_error(y, y_hat)
                meae = median_absolute_error(y, y_hat)
                r2 = r2_score(y, y_hat)
            _rmse.append(rmse)
            _mae.append(mae)
            _meae.append(meae)
            _r2.append(r2)
            _ppl.append(np.exp(np.mean(ppl)))
        if dataset_type == 'predict':
            rmse = _rmse[0]
            mae = _mae[0]
            meae = _meae[0]
            r2 = _r2[0]
            ppl = _ppl[0]
        else:
            rmse = np.mean(_rmse)
            mae = np.mean(_mae)
            meae = np.mean(_meae)
            r2 = np.mean(_r2)
            ppl = np.mean(_ppl)
        if model.name_ in ['regression_miao', 'regression_dte', 'supervised_static_vanilla_topic', 'supervised_dynamic_topic_model'
                          ] and dataset_type == 'test':
            td = model.topic_diversity()
            if model.is_static_topic_embeddings:
                top_word_per_topic, important_words_id = model.top_words(data_loader, num_of_top_words=40)
                important_words_id = important_words_id.detach().cpu().numpy()
                tc = topic_coherence_static(important_words_id[:, :10], data_loader.train.dataset.data['bow'], data_loader.vocab.word_count)
                top_words = ["TOPIC {0}: ".format(j) + " ".join(top_word_per_topic[f'TOPIC {j}']) + "\n" for j in range(len(top_word_per_topic))]
            else:
                top_word_per_topic, important_words_id = model.top_words_dynamic(data_loader, num_of_top_words=40)
                top_words = []
                for topic, value in top_word_per_topic.items():
                    for time, words in value.items():
                        top_words.append(f'{topic} --- {time}: {" ".join(words)}')
                tc = topic_coherence_dynamic(important_words_id[:, :, :10], data_loader.train.dataset.data['bow'], data_loader.vocab.word_count, 10)
            tq = td * tc

        else:
            td, tc, tq = 0, 0, 0
            top_words = None
    return r2, rmse, mae, meae, ppl, td, tc, tq, top_words, time_steps


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
