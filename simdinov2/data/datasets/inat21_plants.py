# inat_plants.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import json
import logging
import os
from enum import Enum
from typing import Callable, Optional, Union

import numpy as np

from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov2")
_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # kept for API compatibility; not used by default

    def __str__(self) -> str:
        return self.value


class INatPlants(ExtendedVisionDataset):
    """
    SimDINOv2-compatible dataset for iNaturalist plants only (~4k classes).
    Builds ImageNet-style extras from iNaturalist JSONs.

    Args:
        split: dataset split (_Split enum or string)
        root: dataset root containing train/val images
        extra: path where generated numpy extras are saved
        train_json / val_json: JSON files (relative to root or absolute)
        train_image_prefix / val_image_prefix: usually 'train' and 'val'
    """
    Target = int
    Split = _Split

    def __init__(
        self,
        *,
        split: Union[str, _Split],
        root: str,
        extra: str,
        train_json: str = "train.json",
        val_json: str = "val.json",
        train_image_prefix: str = "train",
        val_image_prefix: str = "val",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra

        # normalize split to Enum
        if isinstance(split, str):
            split_upper = split.upper()
            if split_upper == "TRAIN":
                self._split = _Split.TRAIN
            elif split_upper == "VAL":
                self._split = _Split.VAL
            elif split_upper == "TEST":
                self._split = _Split.TEST
            else:
                raise ValueError(f"Unknown split '{split}'")
        elif isinstance(split, _Split):
            self._split = split
        else:
            raise TypeError(f"split must be str or _Split, got {type(split)}")

        # Resolve JSON paths
        self._train_json = train_json if os.path.isabs(train_json) else os.path.join(root, train_json)
        self._val_json = val_json if os.path.isabs(val_json) else os.path.join(root, val_json)

        self._train_image_prefix = train_image_prefix
        self._val_image_prefix = val_image_prefix

        # lazy-loaded numpy arrays
        self._entries = None
        self._class_ids = None
        self._class_names = None

    @property
    def split(self) -> _Split:
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        return np.load(self._get_extra_full_path(extra_path), mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(self._get_extra_full_path(extra_path), extra_array)

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    @property
    def _class_ids_path(self) -> str:
        return f"class-ids-{self._split.value.upper()}.npy"

    @property
    def _class_names_path(self) -> str:
        return f"class-names-{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        return self._entries

    def _get_class_ids(self) -> np.ndarray:
        if self._class_ids is None:
            self._class_ids = self._load_extra(self._class_ids_path)
        return self._class_ids

    def _get_class_names(self) -> np.ndarray:
        if self._class_names is None:
            self._class_names = self._load_extra(self._class_names_path)
        return self._class_names

    def find_class_id(self, class_index: int) -> str:
        return str(self._get_class_ids()[class_index])

    def find_class_name(self, class_index: int) -> str:
        return str(self._get_class_names()[class_index])

    def get_image_data(self, index: int) -> bytes:
        entries = self._get_entries()
        image_relpath = str(entries[index]["image_relpath"])
        with open(os.path.join(self.root, image_relpath), "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Optional[_Target]:
        return int(self._get_entries()[index]["class_index"])

    def get_targets(self) -> np.ndarray:
        return self._get_entries()["class_index"]

    def get_class_id(self, index: int) -> str:
        return str(self._get_entries()[index]["class_id"])

    def get_class_name(self, index: int) -> str:
        return str(self._get_entries()[index]["class_name"])

    def __len__(self) -> int:
        return len(self._get_entries())

    def _read_inat_json(self, json_path: str) -> dict:
        with open(json_path, "r") as f:
            return json.load(f)

    def dump_extra(self) -> None:
        """
        Build ImageNet-style extras for TRAIN and VAL, filtering only plant categories.
        Produces:
            - entries-TRAIN.npy / entries-VAL.npy
            - class-ids-TRAIN.npy / class-ids-VAL.npy
            - class-names-TRAIN.npy / class-names-VAL.npy
        """
        logger.info("Building extras for iNaturalist plants")

        # Load JSONs
        j_train = self._read_inat_json(self._train_json)
        j_val = self._read_inat_json(self._val_json)

        # Extract plant categories
        categories = j_train.get("categories", [])
        plant_cat_ids, catid_to_dirname, catid_to_name = [], {}, {}
        for cat in categories:
            kingdom = str(cat.get("kingdom", "")).lower()
            if kingdom == "plantae":
                plant_cat_ids.append(cat["id"])
                safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in cat.get("name", ""))
                catid_to_dirname[cat["id"]] = f"{cat['id']:05d}_{safe_name}"
                catid_to_name[cat["id"]] = cat.get("name", safe_name)

        logger.info(f"Found {len(plant_cat_ids)} plant categories")

        # Helper to process one split
        def process_split(json_obj, image_prefix):
            images = {img["id"]: img for img in json_obj.get("images", [])}
            annotations = json_obj.get("annotations", [])
            entries_tmp, class_id_list, class_name_list = [], [], []
            classid2idx = {}
            missing_images = 0

            for ann in annotations:
                cat_id = ann.get("category_id")
                if cat_id not in plant_cat_ids:
                    continue
                img = images.get(ann["image_id"])
                if img is None:
                    missing_images += 1
                    continue
                file_name = img.get("file_name", "")
                if not file_name:
                    continue
                if file_name.startswith(image_prefix + os.sep) or file_name.startswith(image_prefix + "/"):
                    image_relpath = file_name
                else:
                    if os.path.dirname(file_name):
                        image_relpath = os.path.join(image_prefix, file_name)
                    else:
                        image_relpath = os.path.join(image_prefix, catid_to_dirname[cat_id], file_name)

                class_id = catid_to_dirname[cat_id]
                if class_id not in classid2idx:
                    idx = len(class_id_list)
                    classid2idx[class_id] = idx
                    class_id_list.append(class_id)
                    class_name_list.append(catid_to_name[cat_id])

                class_index = classid2idx[class_id]
                class_name = class_name_list[class_index]

                entries_tmp.append((image_relpath, np.uint32(len(entries_tmp)+1), np.uint32(class_index), class_id, class_name))

            if missing_images:
                logger.warning(f"{missing_images} annotations missing images for prefix {image_prefix}")

            logger.info(f"Collected {len(entries_tmp)} plant images for prefix {image_prefix}")
            return entries_tmp, class_id_list, class_name_list

        entries_train, class_ids_train, class_names_train = process_split(j_train, self._train_image_prefix)
        entries_val, class_ids_val, class_names_val = process_split(j_val, self._val_image_prefix)

        # Union class lists across train+val
        classid2idx, class_id_order, class_name_order = {}, [], []
        for cid, cname in zip(class_ids_train, class_names_train):
            classid2idx[cid] = len(class_id_order)
            class_id_order.append(cid)
            class_name_order.append(cname)
        for cid, cname in zip(class_ids_val, class_names_val):
            if cid not in classid2idx:
                classid2idx[cid] = len(class_id_order)
                class_id_order.append(cid)
                class_name_order.append(cname)

        logger.info(f"Total plant classes (union train+val): {len(class_id_order)}")

        # Save entries
        def save_entries(entries_tmp, split):
            sample_count = len(entries_tmp)
            max_relpath_len = max(len(x[0]) for x in entries_tmp)
            max_class_id_len = max(len(x[3]) for x in entries_tmp)
            max_class_name_len = max(len(x[4]) for x in entries_tmp)
            dtype = np.dtype([
                ("image_relpath", f"U{max_relpath_len}"),
                ("actual_index", "<u4"),
                ("class_index", "<u4"),
                ("class_id", f"U{max_class_id_len}"),
                ("class_name", f"U{max_class_name_len}"),
            ])
            entries_array = np.empty(sample_count, dtype=dtype)
            for i, (relpath, actual_index, class_index, cid, cname) in enumerate(entries_tmp):
                entries_array[i] = (relpath, actual_index, np.uint32(classid2idx[cid]), cid, cname)
            self._save_extra(entries_array, f"entries-{split.value.upper()}.npy")

        save_entries(entries_train, _Split.TRAIN)
        save_entries(entries_val, _Split.VAL)

        # Save class arrays
        class_count = len(class_id_order)
        cid_len = max(len(cid) for cid in class_id_order)
        cname_len = max(len(cn) for cn in class_name_order)
        class_ids_array = np.array(class_id_order, dtype=f"U{cid_len}")
        class_names_array = np.array(class_name_order, dtype=f"U{cname_len}")
        self._save_extra(class_ids_array, "class-ids-TRAIN.npy")
        self._save_extra(class_ids_array, "class-ids-VAL.npy")
        self._save_extra(class_names_array, "class-names-TRAIN.npy")
        self._save_extra(class_names_array, "class-names-VAL.npy")

        logger.info("Finished dumping extras for iNaturalist plants")
