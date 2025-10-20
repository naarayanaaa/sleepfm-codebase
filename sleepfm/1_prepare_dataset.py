import argparse
import json
import multiprocessing
import os
import pickle
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import LABEL_MAP, LABELS_DICT, PATH_TO_PROCESSED_DATA
from utils import assert_disjoint


LABELS_REVERSE = {value: key for key, value in LABELS_DICT.items()}


def resolve_label(label_value):
    canonical_label = None
    if isinstance(label_value, dict):
        label_value = next(iter(label_value))
    elif isinstance(label_value, (list, tuple, set)):
        label_value = next(iter(label_value), None)

    if label_value is None:
        return None

    if isinstance(label_value, str):
        canonical_label = LABEL_MAP.get(label_value, label_value)
    elif isinstance(label_value, (int, np.integer)):
        canonical_label = LABELS_REVERSE.get(int(label_value))

    if canonical_label is not None and canonical_label in LABELS_DICT:
        return canonical_label

    return None


def parallel_prepare_data(
    args: Tuple[List[str], str, Iterable[str], Iterable[str], Iterable[str], Iterable[str]]
):
    mrns, dataset_dir, mrn_pretrain, mrn_train, mrn_valid, mrn_test = args

    data_dict: Dict[str, List[Dict[str, Dict[str, List[str]]]]] = {
        "pretrain": [],
        "train": [],
        "valid": [],
        "test": [],
    }

    class_counts: Dict[str, Counter] = {split: Counter() for split in data_dict}
    stats = {
        "empty_label_dict_counts": 0,
        "missing_label_files": 0,
        "missing_data_files": 0,
        "missing_event_labels": 0,
        "unmapped_labels": Counter(),
    }

    path_to_Y = os.path.join(dataset_dir, "Y")
    path_to_X = os.path.join(dataset_dir, "X")

    for mrn in tqdm(mrns):
        patient_events: Dict[str, List[str]] = defaultdict(list)
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

        if not os.path.exists(path_to_label):
            logger.warning(f"{mrn} label does not exist")
            stats["missing_label_files"] += 1
            continue

        with open(path_to_label, "rb") as file:
            labels_dict = pickle.load(file)

        if len(labels_dict) == 0:
            logger.info(f"{mrn} label_dict is empty")
            stats["empty_label_dict_counts"] += 1
            continue

        if not os.path.exists(path_to_patient):
            logger.warning(f"{mrn} data does not exist")
            stats["missing_data_files"] += 1
            continue

        for event_data_name in sorted(os.listdir(path_to_patient)):
            event_data_path = os.path.join(path_to_patient, event_data_name)

            if event_data_name not in labels_dict:
                logger.warning(f"{mrn} missing label for event {event_data_name}")
                stats["missing_event_labels"] += 1
                continue

            raw_label = labels_dict[event_data_name]
            canonical_label = resolve_label(raw_label)

            if canonical_label is None:
                stats["unmapped_labels"].update([str(raw_label)])
                logger.warning(
                    f"{mrn} event {event_data_name} has unmapped label {raw_label}"
                )
                continue

            patient_events[canonical_label].append(event_data_path)
            class_counts[split_name].update([canonical_label])

        if patient_events:
            data_dict[split_name].append({mrn: dict(patient_events)})

    return {
        "data": data_dict,
        "class_counts": class_counts,
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Process data and create a dataset")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to the data directory")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for train-test split")
    parser.add_argument("--test_size", type=int, default=100, help="Size of test set")
    parser.add_argument("--debug", action="store_true", help="Debugging")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for parallel processing")
    parser.add_argument("--min_sample", type=int, default=-1, help="Sample dataset")

    args = parser.parse_args()

    dataset_dir = args.dataset_dir or PATH_TO_PROCESSED_DATA

    random_state = args.random_state
    num_threads = args.num_threads
    test_size = args.test_size

    random.seed(random_state)
    np.random.seed(random_state)

    path_to_X = os.path.join(dataset_dir, "X")
    mrns = sorted(os.listdir(path_to_X))

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

    assert_disjoint(
        mrn_pretrain,
        mrn_train,
        mrn_valid,
        mrn_test,
        names=["pretrain", "train", "valid", "test"],
    )

    split_subjects = {
        "pretrain": sorted(mrn_pretrain),
        "train": sorted(mrn_train),
        "valid": sorted(mrn_valid),
        "test": sorted(mrn_test),
    }

    logger.info(
        "Total Pretrain/Train/Valid/Test Splits: {}",
        (len(mrn_pretrain), len(mrn_train), len(mrn_valid), len(mrn_test)),
    )

    with open(os.path.join(dataset_dir, "split_subjects.json"), "w", encoding="utf-8") as f:
        json.dump(split_subjects, f, indent=2)

    mrns_array = np.array(mrns, dtype=object)
    mrns_per_thread = [
        chunk.tolist()
        for chunk in np.array_split(mrns_array, num_threads)
        if len(chunk) > 0
    ]

    tasks = [
        (mrns_one_thread, dataset_dir, mrn_pretrain, mrn_train, mrn_valid, mrn_test)
        for mrns_one_thread in mrns_per_thread
    ]

    with multiprocessing.Pool(num_threads) as pool:
        preprocessed_results = list(pool.imap_unordered(parallel_prepare_data, tasks))

    dataset: Dict[str, List[Dict[str, Dict[str, List[str]]]]] = {key: [] for key in split_subjects}
    aggregated_counts: Dict[str, Counter] = {key: Counter() for key in split_subjects}
    aggregated_stats = {
        "empty_label_dict_counts": 0,
        "missing_label_files": 0,
        "missing_data_files": 0,
        "missing_event_labels": 0,
        "unmapped_labels": Counter(),
    }

    for result in preprocessed_results:
        data_dict = result["data"]
        counts = result["class_counts"]
        stats = result["stats"]

        for key, value in data_dict.items():
            dataset[key].extend(value)

        for split, counter in counts.items():
            aggregated_counts[split].update(counter)

        aggregated_stats["empty_label_dict_counts"] += stats["empty_label_dict_counts"]
        aggregated_stats["missing_label_files"] += stats["missing_label_files"]
        aggregated_stats["missing_data_files"] += stats["missing_data_files"]
        aggregated_stats["missing_event_labels"] += stats["missing_event_labels"]
        aggregated_stats["unmapped_labels"].update(stats["unmapped_labels"])

    for key in dataset:
        dataset[key] = sorted(dataset[key], key=lambda x: list(x.keys())[0])

    logger.info(f"Total Empty label Dicts: {aggregated_stats['empty_label_dict_counts']}")
    if aggregated_stats["missing_label_files"]:
        logger.warning(f"Total Missing label files: {aggregated_stats['missing_label_files']}")
    if aggregated_stats["missing_data_files"]:
        logger.warning(f"Total Missing data directories: {aggregated_stats['missing_data_files']}")
    if aggregated_stats["missing_event_labels"]:
        logger.warning(f"Total Events missing labels: {aggregated_stats['missing_event_labels']}")
    if aggregated_stats["unmapped_labels"]:
        logger.warning(f"Unmapped labels encountered: {dict(aggregated_stats['unmapped_labels'])}")

    for split, counter in aggregated_counts.items():
        logger.info(
            f"{split.capitalize()} class counts: {dict(sorted(counter.items()))} (total={sum(counter.values())})"
        )

    logger.info(f"Saving in path: {dataset_dir}")
    with open(os.path.join(dataset_dir, "dataset.pickle"), "wb") as file:
        pickle.dump(dataset, file)

    dataset_event: Dict[str, List[Tuple[str, str]]] = {}
    split_keys = list(split_subjects.keys())
    for idx, split in enumerate(tqdm(split_keys, total=len(split_keys))):
        split_data = dataset[split]
        split_rng = random.Random(random_state + idx)
        sampled_data: List[Tuple[str, str]] = []

        for item in split_data:
            mrn = list(item.keys())[0]
            patient_data = item[mrn]
            for event, event_data in patient_data.items():
                if args.min_sample == -1 or len(event_data) <= args.min_sample:
                    sampled_events = list(event_data)
                else:
                    sampled_events = split_rng.sample(event_data, args.min_sample)

                sampled_data.extend((path, event) for path in sampled_events)

        split_rng.shuffle(sampled_data)
        dataset_event[split] = sampled_data

    with open(os.path.join(dataset_dir, f"dataset_events_{args.min_sample}.pickle"), "wb") as file:
        pickle.dump(dataset_event, file)


if __name__ == "__main__":
    main()
