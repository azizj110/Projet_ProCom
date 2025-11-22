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

test_subject = subject_ids[-1]
trainval_subjects = subject_ids[:-1]
split_idx = max(1, int(len(trainval_subjects) * 0.8))
train_subjects = trainval_subjects[:split_idx]
val_subjects = trainval_subjects[split_idx:] or trainval_subjects[-1:]

train_dataset = BaseConcatDataset([windows_by_subject[s] for s in train_subjects])
val_dataset = BaseConcatDataset([windows_by_subject[s] for s in val_subjects])
test_dataset = BaseDataset(windows_by_subject[test_subject])
print(dir(windows_by_subject[subject_ids[0]]))
print(type(windows_by_subject[subject_ids[0]]))
print(windows_by_subject[subject_ids[0]].description)
raw=windows_by_subject[subject_ids[0]].raw
annotations = raw.annotations
df_ann = annotations.to_data_frame()
print(df_ann.head())
print(df_ann["description"].nunique())
print(df_ann["description"].value_counts())
print(raw.get_data().shape)

def eegnet_collate(batch):
    """Collate function that drops crop indices and returns torch tensors."""
    xs, ys, _ = zip(*batch)
    x_tensor = torch.tensor(np.stack(xs), dtype=torch.float32)
    y_tensor = torch.tensor(ys, dtype=torch.long)
    return x_tensor, y_tensor

train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=eegnet_collate,
        pin_memory=torch.cuda.is_available(),
    )
val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=eegnet_collate,
        pin_memory=torch.cuda.is_available(),
    )
test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=eegnet_collate,
        pin_memory=torch.cuda.is_available(),
    )

X, y = next(iter(train_loader))
print(f"Train batch X shape: {X.shape}")
print(f"Train batch y shape: {y.shape}")
print(f"Train subjects: {train_subjects}")
print(f"Val subjects: {val_subjects}")
print(f"Test subject: {test_subject}")

clf = EEGClassifier(
    module=EEGNet,
    optimizer=torch.optim.Adam,
    optimizer__lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    max_epochs=EPOCH,
    train_split=train_loader,
    device=device,
    callbacks=[
        EarlyStopping(monitor="valid_loss", patience=PATIENCE),
        EpochScoring(
            scoring="accuracy", on_train=True, name="train_acc", lower_is_better=False
        ),
        EpochScoring(
            scoring="accuracy", on_train=False, name="valid_acc", lower_is_better=False
        ),
    ],
    verbose=1,  # Not printing the results for each epoch
)

# Create the pipelines
pipes = {}
pipes["EEGNet"] = make_pipeline(clf)


