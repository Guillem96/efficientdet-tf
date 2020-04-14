import json
import base64
import hashlib
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import tensorflow as tf


def _md5(fname):
    hash_md5 = hashlib.md5()
    with open(str(fname), "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return base64.b64encode(hash_md5.digest()).decode()


def save(model: 'EfficientDet',
         parameters: dict,
         save_dir: Union[str, Path],
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
    to_gcs: bool, default False
        Wether or not to store the model on google cloud too
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    model_fname = save_dir / 'model.tf'
    hp_fname = save_dir / 'hp.json'
    
    json.dump(parameters, hp_fname.open('w'))
    model.save_weights(str(model_fname))

    if to_gcs:
        from google.cloud import storage

        client = storage.Client(project='ml-generic-purpose')
        bucket = client.bucket('ml-generic-purpose-tf-models')
        prefix = save_dir.stem

        blob = bucket.blob(f'{prefix}/hp.json')
        blob.upload_from_filename(str(hp_fname))

        model.save_weights(
            f'gs://ml-generic-purpose-tf-models/{prefix}/model.tf')


def load(save_dir_or_url: Union[str, Path], **kwargs) -> 'EfficientDet':
    """
    Load efficientdet model from google cloud storage or from local
    file.

    In case you want to download the model from gsc use a path formated
    as follows: gs://{bucket}/{model_dir}
    """
    save_dir_url = urlparse(str(save_dir_or_url))

    if save_dir_url.scheme == 'gs':
        from google import auth
        from google.cloud import storage

        save_dir = Path.home() / '.effdet-checkpoints'
        save_dir.mkdir(exist_ok=True, parents=True)
        
        client = storage.Client(
            project='ml-generic-purpose',
            credentials=auth.credentials.AnonymousCredentials())
        bucket = client.bucket('ml-generic-purpose-tf-models')

        prefix = save_dir_url.path[1:] + '/'
        blobs = bucket.list_blobs(prefix=prefix)

        hp_fname = save_dir / 'hp.json'
        model_path = save_dir / 'model.tf'

        hp_blob = bucket.blob(prefix + 'hp.json')
        hp_blob.reload()

        if hp_fname.exists() and _md5(hp_fname) != hp_blob.md5_hash:
            for blob in blobs:
                fname = save_dir / blob.name.replace(prefix, '')
                blob.download_to_filename(fname)
        elif not hp_fname.exists():
            for blob in blobs:
                fname = save_dir / blob.name.replace(prefix, '')
                blob.download_to_filename(fname)
    else:
        hp_fname = Path(save_dir_or_url) / 'hp.json'
        model_path = Path(save_dir_or_url) / 'model.tf'

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
    
    return model, hp
    
