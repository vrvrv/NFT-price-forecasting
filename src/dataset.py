import pandas as pd
import pytorch_forecasting.data as pfd
from omegaconf import ListConfig, DictConfig
from pytorch_forecasting import TimeSeriesDataSet


def preprocess_config(config):
    preprocessed_config = dict()
    for k, v in config.items():
        if isinstance(v, ListConfig):
            preprocessed_config[k] = list(v)
        elif isinstance(v, DictConfig):
            preprocessed_config[k] = preprocess_config(v)
        else:
            preprocessed_config[k] = v

    return preprocessed_config


def load_data(configs):
    if configs.data_name == 'stallion':
        data = pd.read_pickle(configs.data_dir)
        training_cutoff = data['time_idx'].max() - configs.preprocess.max_prediction_length
        data = data[lambda x: x.time_idx <= training_cutoff]

        target_normalizer = getattr(pfd, configs.others.dataset.target_normalizer)(
            **preprocess_config(configs.others.dataset.target_normalizer_cfg)
        )

        training = TimeSeriesDataSet(
            data=data, **preprocess_config(configs.dataset), target_normalizer=target_normalizer
        )
    else:
        raise NotImplementedError

    return data, training
