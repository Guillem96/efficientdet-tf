import json
import base64
import hashlib
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import numpy as np
import tensorflow as tf


def _serialize_value(value):
    if tf.is_tensor(value):
        return value.numpy().item()
    
    if isinstance(value, (np.ndarray, np.float64, 
                          np.float32, np.int32, np.int64)):
        return value.item()
    
    return value


def _deserialize_optimizer(optim: dict) -> tf.optimizers.Optimizer:
    from importlib import import_module

    def import_cls(path):
        mod, cls_ = path.rsplit('.', 1)
        mod = import_module(mod)
        return getattr(mod, cls_)

    optim_cls = import_cls(optim['class_module'])

    if isinstance(optim['config']['learning_rate'], dict):
        scheduler = optim['config']['learning_rate']
        scheduler_cls = import_cls(scheduler['class_module'])
        scheduler = scheduler_cls.from_config(scheduler['config'])
        optim['config']['learning_rate'] = scheduler
    
    optim_instance = optim_cls(**optim['config'])
    # for n, v in optim['weights'].items():
    #     optim_instance._set_hyper(n, v)
    # optim_instance._create_hypers()
    optim_instance.iterations = tf.Variable(optim['iterations'], 
                                            trainable=False)
    # optim_instance.set_weights([v for n, v in optim['weights'].items()])
    return optim_instance


def _serialize_optimizer(optim: tf.optimizers.Optimizer) -> dict:
    def get_import_path(o):
        module = o.__class__.__module__
        name = o.__class__.__name__
        return f'{module}.{name}'

    config = optim.get_config()
    import_path = get_import_path(optim)
    
    if isinstance(config['learning_rate'], dict):
        # Serialize scheduler
        scheduler = optim.learning_rate
        scheduler_config = scheduler.get_config()
        scheduler_config = {k: _serialize_value(v) 
                            for k, v in scheduler_config.items()}
        scheduler_serialized = dict(
            class_module=get_import_path(scheduler),
            config=scheduler_config)
        config['learning_rate'] = scheduler_serialized
    
    # symbolic_weights = getattr(optim, 'weights')
    # weight_names = [str(w.name) for w in symbolic_weights]
    # weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
    # optim_weights = {n: v.tolist() 
    #                  for n, v in zip(weight_names, weight_values)}

    config = {k: _serialize_value(v) for k, v in config.items()}
    return dict(
        class_module=import_path,
        config=config,
        iterations=_serialize_value(optim.iterations),
        # weights=optim_weights
    )


def _md5(fname):
    hash_md5 = hashlib.md5()
    with open(str(fname), "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return base64.b64encode(hash_md5.digest()).decode()


def save(model: 'EfficientDet',
         parameters: dict,
         save_dir: Union[str, Path],
         optimizer: tf.optimizers.Optimizer = None,
         to_gcs: bool = False):
    """
    Keras model checkpointing with extra functionalities

    Parameters
    ----------
    model: EfficientDet
        Model to be serialized
    parameters: dict
        Dictionary containing the CLI arguments used to train the model
    save_dir: Union[str, Path]
        Directory to store the model
    optimizer: tf.optimizers.Optimizer, default None
        If left to none the optimizer won't be serialized
    to_gcs: bool, default False
        Wether or not to store the model on google cloud too
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    model_fname = save_dir / 'model.tf'
    hp_fname = save_dir / 'hp.json'
    
    json.dump(parameters, hp_fname.open('w'))
    model.save_weights(str(model_fname))

    if optimizer is not None:
        optimizer_fname = save_dir / 'optimizer.json'
        optimizer_ser = _serialize_optimizer(optimizer)
        json.dump(optimizer_ser, optimizer_fname.open('w'))
    
    if to_gcs:
        from google.cloud import storage

        client = storage.Client(project='ml-generic-purpose')
        bucket = client.bucket('ml-generic-purpose-tf-models')
        prefix = save_dir.stem

        blob = bucket.blob(f'{prefix}/hp.json')
        blob.upload_from_filename(str(hp_fname))

        if optimizer is not None:
            blob = bucket.blob(f'{prefix}/optimizer.json')
            blob.upload_from_filename(str(optimizer_fname))

        model.save_weights(
            f'gs://ml-generic-purpose-tf-models/{prefix}/model.tf')


def download_folder(gs_path: str) -> str:
    from google import auth
    from google.cloud import storage

    save_dir_url = urlparse(str(gs_path))
    assert save_dir_url.scheme == 'gs'

    save_dir = Path.home() / '.effdet-checkpoints' / save_dir_url.path[1:]
    save_dir.mkdir(exist_ok=True, parents=True)
    
    client = storage.Client(
        project='ml-generic-purpose',
        credentials=auth.credentials.AnonymousCredentials())
    bucket = client.bucket('ml-generic-purpose-tf-models')

    prefix = save_dir_url.path[1:] + '/'
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        blob.reload()
        name = blob.name.replace(prefix, '')
        fname = save_dir / name
        fname.parent.mkdir(parents=True, exist_ok=True)
        if not fname.exists() or (fname.exists() and 
                                    _md5(fname) != blob.md5_hash):
            blob.download_to_filename(fname)

    return save_dir


def load(save_dir_or_url: Union[str, Path], 
         load_optimizer: bool = False,
         **kwargs) -> 'EfficientDet':
    """
    Load efficientdet model from google cloud storage or from local
    file.

    In case you want to download the model from gsc use a path formated
    as follows: gs://{bucket}/{model_dir}
    """
    save_dir_url = urlparse(str(save_dir_or_url))

    if save_dir_url.scheme == 'gs':
        save_dir = download_folder(save_dir_or_url)

        hp_fname = save_dir / 'hp.json'
        model_path = save_dir / 'model.tf'
        optimizer_fname = save_dir / 'optimizer.json'

    else:
        hp_fname = Path(save_dir_or_url) / 'hp.json'
        model_path = Path(save_dir_or_url) / 'model.tf'
        optimizer_fname = Path(save_dir_or_url) / 'optimizer.json'

    assert hp_fname.exists()

    with hp_fname.open() as f:
        hp = json.load(f)

    from efficientdet.models import EfficientDet

    model = EfficientDet(
        hp['n_classes'],
        D=hp['efficientdet'],
        bidirectional=hp['bidirectional'],
        freeze_backbone=True,
        weights=None,
        **kwargs)
    
    print('Loading model weights from {}...'.format(str(model_path)))
    model.load_weights(str(model_path))
    for l in model.layers:
        l.trainable = False
    model.trainable = False

    if load_optimizer:
        assert optimizer_fname.exists()

        with optimizer_fname.open() as f:
            optim_config = json.load(f)
        
        optimizer = _deserialize_optimizer(optim_config)
        return (model, optimizer), hp
        
    return model, hp
    