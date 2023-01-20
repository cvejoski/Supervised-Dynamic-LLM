# supervised-dynamic-topic-model

> This is the code acompaning the paper "The Future is Different: Predicting Reddits Popularity with
Variational Dynamic Language Models".

Large pre-trained language models (LPLM)
have shown spectacular success when finetuned 
on downstream supervised tasks. Yet, it
is known that their performance can drastically
drop when there is a distribution shift between
the data used during training and that used at inference time. 
In this paper we focus on data distributions that 
naturally change over time and
introduce four new REDDIT datasets, namely
the WALLSTREETBETS , ASKSCIENCE, THE
DONALD, and POLITICS sub-reddits. First, we
empirically demonstrate that LPLM can display average performance drops of about 79%
in the best case (103% in the worst case) when
predicting the popularity of future posts from
sub-reddits whose topic distribution changes
with time. We then introduce a simple methodology that leverages neural variational dynamic
topic models and attention mechanisms to infer temporal language model representations
for regression tasks. Our models display performance drops of only about 33% in the best
cases (82% in the worst ones) when predicting the popularity of future posts, while using
only about 7% of the total number of parameters of LPLM and providing interpretable rep-
resentations that offer insight into real-world
events, like the GameStop short squeeze of
2021.


## Installation

In order to set up the necessary environment:

1. review and uncomment what you need in `environment.yml` and create an environment `drf` with the help of [conda]:
   ```bash
   conda env create -f environment.yml
   ```
2. activate the new environment with:
   ```bash
   conda activate drf
   ```

> **_NOTE:_**  The conda environment will have supervised_dynamic_topim_model installed in editable mode.
> Some changes, e.g. in `setup.cfg`, might require you to run `pip install -e .` again.


Then take a look into the `scripts` folders.

## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yml` for the exact reproduction of your
   environment with:
   ```bash
   conda env export -n drf -f environment.lock.yml
   ```
   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yml` using:
   ```bash
   conda env update -f environment.lock.yml --prune
   ```


## Model Training

All the training scripts are in the `scripts\train\regression-models` for the supervised dynamic topic model and in `scripts\train\regression-models\topic-models` for the dynamic topic model.

We provide training scripts for all the new models introduced in the paper nad as well as for all the baseline models. For example if we want to run the training of the D-TAM-GRU model one should execute the following command

```bash
python scripts\train\regression-models\reddit-submissions\atm-dynamic-topic.py
```
>NOTE: all the grid search values for the hyperparamters used for training the models used in the paper can be found here

## Model Evaluation

Evaluation of a trained model is done by using the evaluations script `scripts\evaluate\eval_model.py`. The script has the following arguments

- `--models-root-dir` - path to the root dir where the trained model(s) is/are stored,
- `--data-path` - path to the root dir where the test data is stored,
- `--results-output` - path to the output dir where the results will be stored,
- `--gpu` - select the GPU number,
