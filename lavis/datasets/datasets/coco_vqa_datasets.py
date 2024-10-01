"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import torch
import numpy as np

from PIL import Image

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

from collections import OrderedDict


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )

class OKVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def collater(self, samples):
        image_list, question_list, answer_list, weight_list = [], [], [], []
        gold_answer = []
        passages_list = []
        question_id_list = []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            passages_list.append(sample["passages"])
            question_id_list.append(sample["question_id"])

            answers = sample["answers"]
            gold_answer.append(sample["gold_answer"])

            answer_list.append(answers)

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "passages": passages_list,
            "answers": answer_list,
            "gold_answer": gold_answer,
            "question_id": torch.tensor(question_id_list),
        }

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        #####gs knowledge#####
        passages = ann["passages"][:30]

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        answer = answers[weights.index(max(weights))]


        return {
            "image": image,
            "text_input": question,
            "passages": passages,
            "answers": answers,
            "weights": weights,
            "gold_answer": answer,
            "question_id": ann["question_id"],
        }

class OKVQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))

        answer_list_path = ann_paths[1]
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def collater(self, samples):
        image_list, question_list, question_id_list, instance_id_list = [], [], [], []
        passages_list = []

        num_answers = []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            passages_list.append(sample["passages"])
            question_id_list.append(sample["question_id"])
            instance_id_list.append(sample["instance_id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "passages": passages_list,
            "question_id": torch.tensor(question_id_list),
            "instance_id": instance_id_list,
        }

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        #####gs knowledge#####
        passages = ann["passages"][:30]
        # passages = ' '.join(passages)

        return {
            "image": image,
            "text_input": question,
            "passages": passages,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }

class FVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def collater(self, samples):
        image_list, question_list = [], []
        gold_answer = []
        passages_list = []
        question_id_list = []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            passages_list.append(sample["passages"])
            question_id_list.append(int(sample["question_id"]))
            gold_answer.append(sample["gold_answer"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "passages": passages_list,
            "gold_answer": gold_answer,
            "answer": gold_answer,
            "question_id": torch.tensor(question_id_list),
        }

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        #####concept_net knowledge#####
        passages = ann["passages"][:100]

        answer = ann["answer"]


        return {
            "image": image,
            "text_input": question,
            "passages": passages,
            "gold_answer": answer,
            "question_id": ann["question_id"],
        }
