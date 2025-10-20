"""Sets paths based on configuration files."""

import configparser
import os
import types

_FILENAME = None
_PARAM = {}

CONFIG = types.SimpleNamespace(
    FILENAME=_FILENAME,
    DATASETS=_PARAM.get("datasets", "datasets"),
    OUTPUT=_PARAM.get("output", "output"),
    CACHE=_PARAM.get("cache", ".cache"),
)

# Define the paths via environment variables so the repository can run outside
# the original Stanford infrastructure.
PATH_TO_RAW_DATA = os.environ.get("SLEEPFM_RAW_PATH", "data/raw")
PATH_TO_PROCESSED_DATA = os.environ.get("SLEEPFM_PROCESSED_PATH", "data/processed")

# Define Sleep related global variables

LABELS_DICT = {
    "Wake": 0, 
    "Stage 1": 1, 
    "Stage 2": 2, 
    "Stage 3": 3, 
    "REM": 4
}

MODALITY_TYPES = ["respiratory", "sleep_stages", "ekg"]
CLASS_LABELS = ["Wake", "Stage 1", "Stage 2", "Stage 3", "REM"]
NUM_CLASSES = 5

EVENT_TO_ID = {
    "Wake": 1, 
     "Stage 1": 2, 
     "Stage 2": 3, 
     "Stage 3": 4, 
     "Stage 4": 4, 
     "REM": 5,
}

LABEL_MAP = {
    "Sleep stage W": "Wake",
    "Sleep stage N1": "Stage 1",
    "Sleep stage N2": "Stage 2",
    "Sleep stage N3": "Stage 3",
    "Sleep stage R": "REM",
    "Sleep Stage W": "Wake",
    "Sleep Stage N1": "Stage 1",
    "Sleep Stage N2": "Stage 2",
    "Sleep Stage N3": "Stage 3",
    "Sleep Stage R": "REM",
    "W": "Wake",
    "N1": "Stage 1",
    "N2": "Stage 2",
    "N3": "Stage 3",
    "N4": "Stage 3",
    "REM": "REM",
    "R": "REM",
    "wake": "Wake",
    "nonrem1": "Stage 1",
    "nonrem2": "Stage 2",
    "nonrem3": "Stage 3",
    "rem": "REM",
    "Stage W": "Wake",
    "Stage N1": "Stage 1",
    "Stage N2": "Stage 2",
    "Stage N3": "Stage 3",
    "Stage N4": "Stage 3",
    "Stage R": "REM",
    "Stage 1": "Stage 1",
    "Stage 2": "Stage 2",
    "Stage 3": "Stage 3",
    "Stage 4": "Stage 3",
    "S0": "Wake",
    "S1": "Stage 1",
    "S2": "Stage 2",
    "S3": "Stage 3",
    "S4": "Stage 3",
    "MT": "Wake",
    "Movement": "Wake",
    "Movement Time": "Wake",
    "Awake": "Wake",
    "Wake": "Wake",
    "REM sleep": "REM",
    "REM Sleep": "REM",
    "NonREM1": "Stage 1",
    "NonREM2": "Stage 2",
    "NonREM3": "Stage 3",
}


# Define the channels in your dataset
ALL_CHANNELS = ['F3-M2',
 'F4-M1',
 'C3-M2',
 'C4-M1',
 'O1-M2',
 'O2-M1',
 'E1-M2',
 'Chin1-Chin2',
 'ABD',
 'CHEST',
 'AIRFLOW',
 'SaO2',
 'ECG']


CHANNEL_DATA = {
    "Respiratory": ["CHEST", "SaO2", "ABD"],
    "Sleep_Stages": ["C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2"],
    "EKG": ["ECG"], 
    }


CHANNEL_DATA_IDS = {
    "Respiratory": [ALL_CHANNELS.index(item) for item in CHANNEL_DATA["Respiratory"]], 
    "Sleep_Stages": [ALL_CHANNELS.index(item) for item in CHANNEL_DATA["Sleep_Stages"]], 
    "EKG": [ALL_CHANNELS.index(item) for item in CHANNEL_DATA["EKG"]], 
 }