import json
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

from google.cloud import storage

from .models import EfficientDet


def save(model: EfficientDet,
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
        client = storage.Client()
        bucket = client.bucket('ml-generic-purpose-tf-models')
        prefix = save_dir.stem
        for p in save_dir.iterdir():
            blob = bucket.blob(f'{prefix}/{p.stem}{p.suffix}')
            blob.upload_from_filename(str(p))


def load(save_dir: Union[str, Path]) -> EfficientDet:
    """
    Load efficientdet model from google cloud storage or from local
    file.

    In case you want to download the model from gsc use a path formated
    as follows: gs://{bucket}/{model_dir}
    """
    save_dir_url = urlparse(str(save_dir))

    if save_dir_url.scheme == 'gs':
        save_dir = Path('.checkpoints', save_dir_url.path)
        save_dir.mkdir(exist_ok=True, parents=True)

        client = storage.Client()
        bucket = client.bucket()
        blobs = bucket.list_blobs(prefix=save_dir_url.path)
        for blob in blobs:
            blob.download_to_filename(save_dir / blob.name)
    else:
        save_dir = Path(save_dir)

    chkp_check = save_dir / 'model.tf.index'
    model_fname = save_dir / 'model.tf'
    hp_fname = save_dir / 'hp.json'

    assert chkp_check.exists() and hp_fname.exists()

    hp = json.load(hp_fname.open())

    model = EfficientDet(
        hp['n_classes'],
        D=hp['efficientdet'],
        bidirectional=hp['bidirectional'],
        freeze_backbone=True,
        weights=None)
    model.load_weights(str(model_fname))
    return model
    
