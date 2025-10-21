import argparse
import json
import multiprocessing
import os
import pickle
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (
    ADDITIONAL_LABEL_MAPS,
    LABEL_MAP,
    PATH_TO_PROCESSED_DATA,
    canonical_stage,
)

def parallel_prepare_data(args):

    (
        mrns,
        dataset_dir,
        mrn_pretrain,
        mrn_train,
        mrn_valid,
        mrn_test,
        label_key,
        custom_label_map,
    ) = args

    data_dict = {
        "pretrain": [],
        "train": [],
        "valid": [],
        "test": [],
    }

    custom_events = {
        "pretrain": [],
        "train": [],
        "valid": [],
        "test": [],
    }

    empty_label_dict_counts = 0
    path_to_Y = os.path.join(dataset_dir, "Y")
    for mrn in tqdm(mrns):
        one_patient_dict = {
            mrn: {}
        }

        path_to_X = os.path.join(dataset_dir, "X")
        path_to_patient = os.path.join(path_to_X, mrn)
        path_to_label = os.path.join(path_to_Y, f"{mrn}.pickle")

        if mrn in mrn_pretrain:
            split_name = "pretrain"
        elif mrn in mrn_train:
            split_name = "train"
        elif mrn in mrn_valid:
            split_name = "valid"
        elif mrn in mrn_test:
            split_name = "test"
        else:
            raise Warning(f"{mrn} Not in any split")

        if os.path.exists(path_to_label):
            with open(path_to_label, 'rb') as file:
                labels_dict = pickle.load(file)
        else:
            logger.info(f"{mrn} label does not exist")
            continue
        
        if len(labels_dict) == 0:
            logger.info(f"{mrn} label_dict is empty")
            empty_label_dict_counts += 1
            continue
        
        if not os.path.exists(path_to_patient):
            logger.info(f"{mrn} data does not exist")
            continue

        for event_data_name in os.listdir(path_to_patient):
            event_data_path = os.path.join(path_to_patient, event_data_name)

            label_entry = labels_dict.get(event_data_name)
            stage_label_raw = None
            custom_label_raw = None

            if isinstance(label_entry, dict):
                stage_label_raw = label_entry.get("sleep_stage") or label_entry.get("stage")
                custom_label_raw = label_entry.get(label_key)
            else:
                stage_label_raw = label_entry
                if label_key == "sleep_stage":
                    custom_label_raw = label_entry

            if stage_label_raw is None:
                continue

            stage_label = canonical_stage(stage_label_raw)
            stage_label = LABEL_MAP.get(stage_label, stage_label)

            if stage_label not in one_patient_dict[mrn]:
                one_patient_dict[mrn][stage_label] = []

            one_patient_dict[mrn][stage_label].append(event_data_path)

            if custom_label_raw is not None and custom_label_raw != "":
                mapped_label = custom_label_map.get(custom_label_raw, custom_label_raw)
                custom_events[split_name].append((event_data_path, mapped_label))

        data_dict[split_name].append(one_patient_dict)

    logger.info(f"Total Empty label Dicts: {empty_label_dict_counts}")

    return data_dict, custom_events

def main():
    parser = argparse.ArgumentParser(description="Process data and create a dataset")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to the data directory")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for train-test split")
    parser.add_argument("--test_size", type=int, default=100, help="Size of test set")
    parser.add_argument("--debug", action="store_true", help="Debugging")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for parallel processing")
    parser.add_argument("--min_sample", type=int, default=-1, help="Sample dataset")
    parser.add_argument(
        "--label_key",
        type=str,
        default="sleep_stage",
        help="Label key stored in the event dictionaries for downstream tasks",
    )
    parser.add_argument(
        "--label_map_json",
        type=str,
        default=None,
        help="Optional JSON mapping to canonicalize custom labels",
    )

    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    if dataset_dir == None:
        dataset_dir = PATH_TO_PROCESSED_DATA

    random_state = args.random_state
    num_threads = args.num_threads
    test_size = args.test_size

    path_to_X = os.path.join(dataset_dir, "X")
    mrns = os.listdir(path_to_X)

    if args.debug:
        logger.info("Running in Debug Mode")
        mrns = mrns[:100]
    logger.info(f"Number of Mrns being processed: {len(mrns)}")

    mrn_pretrain, mrn_train = train_test_split(mrns, test_size=0.25, random_state=random_state)
    mrn_train, mrn_test = train_test_split(mrn_train, test_size=test_size, random_state=random_state)
    mrn_train, mrn_valid = train_test_split(mrn_train, test_size=0.10, random_state=random_state)

    mrn_pretrain = set(mrn_pretrain)
    mrn_train = set(mrn_train)
    mrn_valid = set(mrn_valid)
    mrn_test = set(mrn_test)

    logger.info(f"Total Pretrain/Train/Valid/Test Splits: {len(mrn_pretrain), len(mrn_train), len(mrn_valid), len(mrn_test)}")

    if args.label_key == "sleep_stage":
        custom_label_map = LABEL_MAP
    else:
        custom_label_map = dict(ADDITIONAL_LABEL_MAPS.get(args.label_key, {}))
    if args.label_map_json:
        custom_label_map.update(json.loads(args.label_map_json))

    mrns_per_thread = np.array_split(mrns, num_threads)

    tasks = [
        (
            mrns_one_thread,
            dataset_dir,
            mrn_pretrain,
            mrn_train,
            mrn_valid,
            mrn_test,
            args.label_key,
            custom_label_map,
        )
        for mrns_one_thread in mrns_per_thread
    ]
    with multiprocessing.Pool(num_threads) as pool:
        preprocessed_results = list(pool.imap_unordered(parallel_prepare_data, tasks))

    dataset = {}
    custom_events = {"pretrain": [], "train": [], "valid": [], "test": []}
    for data_dict, events_dict in preprocessed_results:
        for key, value in data_dict.items():
            if key not in dataset:
                dataset[key] = value
            else:
                dataset[key].extend(value)
        for key, value in events_dict.items():
            custom_events[key].extend(value)

    for key in dataset:
        dataset[key] = sorted(dataset[key], key=lambda x: list(x.keys())[0])

    logger.info(f"Saving in path: {dataset_dir}")
    with open(os.path.join(dataset_dir, f"dataset.pickle"), 'wb') as file:
        pickle.dump(dataset, file)

    dataset_event = {}
    for split, split_data in tqdm(dataset.items(), total=len(dataset)):
        sampled_data = []
        for item in split_data:
            mrn = list(item.keys())[0]
            patient_data = item[mrn]
            for event, event_data in patient_data.items():
                if args.min_sample == -1:
                    sampled_events = event_data
                else:
                    random.seed(args.random_state)
                    sampled_events = random.sample(event_data, args.min_sample) if len(event_data) > args.min_sample else event_data
                
                sampled_events = [(path, event) for path in sampled_events]
                sampled_data.extend(sampled_events)
        
        random.seed(args.random_state)
        random.shuffle(sampled_data)
        dataset_event[split] = sampled_data

    with open(os.path.join(dataset_dir, f"dataset_events_{args.min_sample}.pickle"), 'wb') as file:
        pickle.dump(dataset_event, file)

    has_custom_labels = any(len(values) > 0 for values in custom_events.values())
    if has_custom_labels and args.label_key != "sleep_stage":
        dataset_event_custom: Dict[str, list] = {}
        for split, events in custom_events.items():
            sampled = events
            if args.min_sample != -1 and len(events) > args.min_sample:
                random.seed(args.random_state)
                sampled = random.sample(events, args.min_sample)
            random.seed(args.random_state)
            random.shuffle(sampled)
            dataset_event_custom[split] = sampled

        label_key_safe = args.label_key.replace(" ", "_").lower()
        custom_filename = f"dataset_events_{label_key_safe}_{args.min_sample}.pickle"
        with open(os.path.join(dataset_dir, custom_filename), "wb") as file:
            pickle.dump(dataset_event_custom, file)

        logger.info(
            "Saved custom label dataset to %s", os.path.join(dataset_dir, custom_filename)
        )

if __name__ == "__main__":
    main()



