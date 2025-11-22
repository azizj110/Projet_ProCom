import pickle
from braindecode.datasets.base import BaseConcatDataset, BaseDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import mne
import seaborn as sns
from moabb.datasets import BNCI2014_001
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import MotorImagery
from moabb.utils import setup_seed
from sklearn.pipeline import make_pipeline
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.dataset import ValidSplit

from braindecode import EEGClassifier
from braindecode.models import EEGNet

mne.set_log_level(False)

# Set up GPU if it is there
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
print("GPU is", "AVAILABLE" if cuda else "NOT AVAILABLE")

LEARNING_RATE = 0.0625 * 0.01  # parameter taken from Braindecode
WEIGHT_DECAY = 0  # parameter taken from Braindecode
EPOCH = 10
PATIENCE = 3
BATCH_SIZE = 32
NUM_WORKERS = 0

with open("windows_by_subject.pkl", "rb") as f:
    windows_by_subject = pickle.load(f)
with open("subject_ids.pkl", "rb") as f:
    subject_ids = pickle.load(f)

case =0
subject=subject_ids[case]
windows_dataset=windows_by_subject[subject]
print(dir(windows_dataset))
print(windows_dataset._description)
#print(windows_dataset.window_kwargs[0])
#print(windows_dataset.y)print(windows_dataset.window_kwargs[0][0])
print(len(windows_dataset.y))
#print(windows_dataset.metadata)
metadata_df=windows_dataset.metadata
print('a:',(metadata_df['target'].value_counts()!=12).sum())
print(metadata_df['target'].nunique())
raw=windows_dataset.raw
annotations = raw.annotations
aan_df = annotations.to_data_frame()
print(aan_df.head())
print(len(annotations))