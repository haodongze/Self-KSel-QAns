"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.aok_vqa_datasets import AOKVQADataset, AOKVQAEvalDataset
from lavis.datasets.datasets.coco_vqa_datasets import OKVQADataset, OKVQAEvalDataset, FVQADataset


@registry.register_builder("ok_vqa")
class OKVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = OKVQADataset
    eval_dataset_cls = OKVQAEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/defaults.yaml",
    }

@registry.register_builder("f_vqa")
class FVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = FVQADataset
    eval_dataset_cls = FVQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/fvqa/defaults.yaml",
    }


@registry.register_builder("aok_vqa")
class AOKVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQADataset
    eval_dataset_cls = AOKVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/aokvqa/defaults.yaml"}