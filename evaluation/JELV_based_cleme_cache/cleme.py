import copy
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
from tabulate import tabulate
import re
import os, json
from typing import List, Dict, Tuple
from .JELV_utils import (
    init_inference_components,
)

from .chunk import (
    Chunk,
    map_parallel,
    merge_edit,
    convert_edit_into_chunk,
    chunk_list_to_text,
    all_correct, chunk_list_to_text_with_replacement, chunk_list_to_text_with_skip, any_correct
)
from .constants import *
from .data import Dataset
from .scorers import SCORERS
from .utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class LengthWeigher:
    alpha: float = field(
        default=3.0, metadata={"help": "Scale factor"}
    )
    bias: float = field(
        default=0.0, metadata={"help": "Bias factor (default not used)"}
    )
    min_value: float = field(
        default=1.0, metadata={"help": "Clamp factor of minimal value"}
    )
    max_value: float = field(
        default=1.0, metadata={"help": "Clamp factor of maximal value"}
    )
    reverse: bool = field(
        default=False, metadata={"help": "Reverse the value or not"}
    )

    def __call__(self, edit_len: int) -> float:
        if self.reverse:
            weight = self.alpha * (1 / (1 + (self.alpha - 1) * np.exp(edit_len - self.bias)))
        else:
            weight = self.alpha * (1 / (1 + (self.alpha - 1) * np.exp(-edit_len + self.bias)))
        return np.clip(weight, self.min_value, self.max_value).item()

    def __str__(self):
        return f"SigmoidWeigher(alpha={self.alpha}, bias={self.bias}, " \
               f"min_value={self.min_value}, max_value={self.max_value}, reverse={self.reverse})"


class CLEME(ABC):
    """ CLEME: De-biasing Multi-reference Evaluation for Grammatical Error Correction [EMNLP 2023]
        :param scorer: used to compute Precision, Recall and F scores (choices: `corpus`, `sentence`).
        :param weigher_config: configuration of Length Weighting.
    """

    def __init__(self, scorer: str = "corpus", weigher_config: Dict = None):
        super().__init__()
        self.scorer = self.build_scorer(name=scorer)
        self.weigher_config = weigher_config
        self.weigher_tp = None
        self.weigher_fp = None
        self.weigher_fn = None

    @classmethod
    def build_scorer(cls, name: str, *args, **kwargs):
        class_type = SCORERS[name]
        return class_type(*args, **kwargs)

    def evaluate(self, dataset_hyp: Dataset, dataset_ref: Dataset, alpha):
        start_time = time.time()
        self.check_datasets(dataset_hyp, dataset_ref)
        dataset_hyp, dataset_ref = self.prepare_datasets(dataset_hyp, dataset_ref)
        prepare_time = time.time() - start_time

        results = []
        index = 1
        for sample_hyp, sample_ref in zip(dataset_hyp, dataset_ref):
            # print(f"Processing sentence {index}")
            index += 1
            results.append(self.evaluate_sample_correction(sample_hyp, sample_ref, alpha))

        score = self.scorer(results, alpha)
        # LOGGER.info(
        #     "{} Total samples: {}, Total time: {:.3f} seconds; Preparation time: {:.3f}".format(
        #         self.__class__, len(dataset_hyp), time.time() - start_time, prepare_time,
        #     )
        # )
        return score, results

    @abstractmethod
    def evaluate_sample_correction(
            self,
            chunks_hyp: List[Chunk],
            chunks_ref: List[List[Chunk]],
            alpha,
    ) -> List[Dict[str, int]]:
        raise NotImplementedError()

    def prepare_datasets(self, dataset_hyp: Dataset, dataset_ref: Dataset):
        """ Prepare for chunk-level evaluation
            1) Acquire Edits using Errant
            2) Chunk partition
            2) Calculate average chunk length
        """
        # Merge dataset_hyp and dataset_ref into one dataset for convenience
        merge_data = copy.deepcopy(dataset_hyp)
        for sample_idx, sample in enumerate(merge_data):
            sample.target.extend(dataset_ref[sample_idx].target)
            sample.edits[0].extend(dataset_ref[sample_idx].edits[0])

        # Chunk partition
        chunk_dataset = self.chunk_partition(merge_data)
        chunk_dataset_hyp, chunk_dataset_ref = [], []
        for sample_idx in range(len(dataset_hyp)):
            assert len(chunk_dataset[sample_idx]) > 1
            chunk_dataset_hyp.append(chunk_dataset[sample_idx][0])
            chunk_dataset_ref.append(chunk_dataset[sample_idx][1:])

        # Prepare for length weighting
        self.setup_length_weighting(chunk_dataset_ref)
        return chunk_dataset_hyp, chunk_dataset_ref

    def setup_length_weighting(self, chunk_dataset_ref: List[List[List[Chunk]]]):
        def construct_length_weight(config, avg_chunk_len):
            return LengthWeigher(
                alpha=config["alpha"],
                bias=avg_chunk_len,
                min_value=config["min_value"],
                max_value=config["max_value"],
                reverse=config["reverse"],
            )

        # Calculate the average length of correct chunks (i.e., Unchanged Chunks)
        # and incorrect chunks (i.e., Corrected/Dummy Chunks)
        chunk_len_correct, chunk_len_incorrect = [], []

        for chunks_list in chunk_dataset_ref:
            for chunks in chunks_list:
                for chunk in chunks:
                    if chunk.type:  # Corrected/Dummy Chunk
                        chunk_len = max(len(chunk.src_tokens), len(chunk.tgt_tokens))
                        chunk_len_incorrect.append(chunk_len)
                    else:  # Unchanged Chunk
                        chunk_len_correct.append(len(chunk.src_tokens))
        avg_chunk_len_correct = np.average(chunk_len_correct)
        avg_chunk_len_incorrect = np.average(chunk_len_incorrect)
        # LOGGER.info(
        #     f"avg_chunk_len_correct={round(avg_chunk_len_correct, 2)}, "
        #     f"avg_chunk_len_incorrect={round(avg_chunk_len_incorrect, 2)}"
        # )

        self.weigher_tp = construct_length_weight(
            self.weigher_config["tp"],
            avg_chunk_len_incorrect,
        )
        self.weigher_fp = construct_length_weight(
            self.weigher_config["fp"],
            avg_chunk_len_incorrect,
        )
        self.weigher_fn = construct_length_weight(
            self.weigher_config["fn"],
            avg_chunk_len_incorrect,
        )
        # LOGGER.info(f"length_weight_tp: {self.weigher_tp}")
        # LOGGER.info(f"length_weight_fp: {self.weigher_fp}")
        # LOGGER.info(f"length_weight_fn: {self.weigher_fn}")

    @classmethod
    def check_datasets(cls, dataset_hyp: Dataset, dataset_ref: Dataset):
        assert len(dataset_hyp) == len(dataset_ref)
        for sample_hyp, sample_ref in zip(dataset_hyp, dataset_ref):
            assert len(sample_hyp.source) == len(sample_hyp.target) == len(sample_hyp.source) == 1, \
                f"{sample_hyp} || {sample_ref}"
            assert sample_hyp.source[0] == sample_ref.source[0]

    def chunk_partition(self, dataset: Dataset) -> List[List[List[Chunk]]]:
        """ Segment the source, hypothesis and references into chunk sequences
            1) Construct token_mapping
            2) Merge edits with overlapping interval
            3) Convert edit into chunk
        """
        chunk_list_dataset = []
        for sample in dataset:
            for tgt_idx, tgt in enumerate(sample.target):
                if not tgt and len(sample.target) > 2:
                    # LOGGER.warning(f"Ignore empty reference: {sample.source[0]} || {tgt}")
                    del sample.target[tgt_idx]
                    del sample.edits[0][tgt_idx]

            # Segment sentence
            src_tokens = sample.source[0].split()
            tgt_tokens_list = [x.split() for x in sample.target]

            # Construct token_mapping
            edits_list, token_mapping_total = [], []
            for tgt_idx in range(len(sample.target)):
                edits = sample.edits[0][tgt_idx]
                edits = sorted(edits, key=lambda x: x.src_interval[0])
                edits_list.append(edits)
                token_mapping = map_parallel(src_tokens, edits)
                token_mapping_total.append(token_mapping)
            # Merge edits with overlapping interval
            merge_edits_list, shared_interval_list = merge_edit(
                src_tokens,
                tgt_tokens_list,
                edits_list,
                token_mapping_total,
            )
            # Convert edit into chunk
            chunk_list_total = convert_edit_into_chunk(
                src_tokens,
                tgt_tokens_list,
                merge_edits_list,
                shared_interval_list,
                token_mapping_total,
            )
            chunk_list_dataset.append(chunk_list_total)
        return chunk_list_dataset

    def visualize(
            self,
            dataset_ref: Dataset,
            dataset_hyp: Dataset = None,
            sout=sys.stdout,
            tablefmt: str = "fancy_grid",
            display_types: bool = False,
    ):
        """ Visualize the results of chunk partition.
            tabular_data {
                "sentence": [],
                "chunk-0": [],
                "chunk-1": [],
                ...,
                "chunk-N": [],
            }
        """
        if dataset_hyp is None:
            dataset = dataset_ref
        else:
            dataset = copy.deepcopy(dataset_ref)
            dataset.merge(dataset_hyp)

        chunk_results = self.chunk_partition(dataset)
        for chunk_result in chunk_results:
            tabular_data = {
                "sentence": ["source"] + [f"target-{x}" for x in list(range(len(chunk_result)))],
            }
            for chunk_idx in range(len(chunk_result[0])):
                chunks = [" ".join(chunk_result[0][chunk_idx].src_tokens)] + \
                         [" ".join(x[chunk_idx].tgt_tokens) for x in chunk_result]
                if len(set(chunks)) > 1:
                    head_name = f"chunk-{chunk_idx} *"
                else:
                    head_name = f"chunk-{chunk_idx}"
                tabular_data[head_name] = chunks
                if display_types and len(set(chunks)) > 1:
                    types = [""] + [" ".join(x[chunk_idx].type) for x in chunk_result]
                    tabular_data[f"Types-{chunk_idx}"] = types
            table = tabulate(
                tabular_data,
                tablefmt=tablefmt,
                headers="keys",
                floatfmt='.3f',
                missingval='N/A',
                numalign='left',
            )
            sout.write(table + "\n")


class DependentChunkMetric(CLEME):
    def evaluate_sample_correction(
            self,
            chunks_hyp: List[Chunk],
            chunks_refs: List[List[Chunk]],
            alpha,
    ) -> List[Dict[str, int]]:
        result = []
        over_chunks = []

        # Process the chunks in each reference
        # print('=' * 50)
        # position_to_edits = {}
        first = 1
        for ref_id, chunks_ref in enumerate(chunks_refs):

            #     for chunk_ref in chunks_ref:
            #         if chunk_ref.type:
            #             edit_type = ' '.join(chunk_ref.type)
            #             position = ' '.join(chunk_ref.src_tokens)
            #             target = ' '.join(chunk_ref.tgt_tokens)

            #             # Group edits by their positions
            #             if position not in position_to_edits:
            #                 position_to_edits[position] = []

            #             position_to_edits[position].append((edit_type, target))

            # position_edit_counts = {}

            # # Analyze the edits for each position
            # for position, edits in position_to_edits.items():
            #     targets = {edit[1] for edit in edits}  # Unique targets
            #     if len(targets) > 1:
            #         # Multiple edits in the same position with different targets
            #         position_edit_counts[position] = {}

            #         for target in targets:
            #             # Count the edit types for each unique target
            #             edit_types = {edit[0] for edit in edits if edit[1] == target}
            #             for edit_type in edit_types:
            #                 if edit_type not in position_edit_counts[position]:
            #                     position_edit_counts[position][edit_type] = 0
            #                 position_edit_counts[position][edit_type] += 1

            # if position_edit_counts:
            #     with open ('../CLEME/multiple_edit_type.txt', 'a') as f:
            #     # Print the final count of each edit type per position
            #         f.write(f"{position_edit_counts}\n")

            # print('-' * 50)

            src, ref = chunk_list_to_text(chunks_ref)
            LOGGER.debug(f"ref: {ref}")

            # Add the stats of overcorrection
            over_correction = 0
            # over_chunks = []

            tp, fp, fn, tn = 0, 0, 0, 0
            tp_chunks, fp_chunks, fn_chunks, tn_chunks = [], [], [], []
            _, hyp = chunk_list_to_text(chunks_hyp)
            # print(hyp)
            for chunk_idx, chunk_hyp in enumerate(chunks_hyp):

                chunk_len = max(len(chunk_hyp.src_tokens), len(chunk_hyp.tgt_tokens))
                if chunk_hyp.type:
                    if chunk_hyp == chunks_ref[chunk_idx]:
                        weight = self.weigher_tp(chunk_len)
                        tp += weight
                        tp_chunks.append((chunk_hyp, chunks_ref[chunk_idx]))
                        LOGGER.debug(f"{round(weight, 2)} TP: {chunk_hyp} || {chunks_ref[chunk_idx]}")
                    else:
                        weight = self.weigher_fp(chunk_len)
                        # fp += weight
                        # fp_chunks.append((chunk_hyp, chunks_ref[chunk_idx]))
                        # LOGGER.debug(f"{round(weight, 2)} FP: {chunk_hyp} || {chunks_ref[chunk_idx]}")

                        # Add the stats of overcorrection
                        if not chunks_ref[chunk_idx].type:
                            # if first:
                            #     print(f"Src:")
                            #     print(src)
                            #     print(f"Hyp:")
                            #     print(hyp)
                            #     first = 0

                            # print(f"Ref ID: {ref_id}")
                            print(f"Ref:")
                            print(ref)
                            over_correction += weight
                            edit_type = ' '.join(chunk_hyp.type)
                            if edit_type not in over_chunks:
                                # with open ('../CLEME/multiple_edit_type_overcorrection_ind.txt', 'a') as f:
                                #     f.write(f"{edit_type} {weight}\n")
                                # print(f"{edit_type} {weight}")
                                over_chunks.append(edit_type)
                            # over_chunks.append((chunk_hyp, chunks_ref[chunk_idx]))

                            overcorrection_hyp = chunk_list_to_text_with_replacement(chunks_ref, chunk_hyp)
                            print("overcorrection_hyp: ")
                            print(overcorrection_hyp)
                            print(chunk_hyp)
                            print('-' * 50)
                            # print(f"Overcorrection: {round(over_correction, 2)}")

                        else:
                            fp += weight
                            fp_chunks.append((chunk_hyp, chunks_ref[chunk_idx]))
                            LOGGER.debug(f"{round(weight, 2)} FP: {chunk_hyp} || {chunks_ref[chunk_idx]}")

                else:
                    if chunk_hyp != chunks_ref[chunk_idx]:
                        weight = self.weigher_fn(chunk_len)
                        fn += weight
                        fn_chunks.append((chunk_hyp, chunks_ref[chunk_idx]))
                        LOGGER.debug(f"{round(weight, 2)} FN: {chunk_hyp} || {chunks_ref[chunk_idx]}")
                    else:
                        weight = 1.00
                        tn += weight
                        tn_chunks.append((chunk_hyp, chunks_ref[chunk_idx]))
                        # LOGGER.debug(f"{round(weight, 2)} TN: {chunk_hyp}")

            result.append({
                KEY_TP: tp,
                KEY_FP: fp,
                KEY_FN: fn,
                KEY_TN: tn,
                KEY_OC: over_correction,
                KEY_OC_EDIT: over_chunks.copy(),
                KEY_TP_EDIT: tp_chunks.copy(),
                KEY_FP_EDIT: fp_chunks.copy(),
                KEY_FN_EDIT: fn_chunks.copy(),
                KEY_TN_EDIT: tn_chunks.copy(),
            })
            LOGGER.debug(
                f"tp={round(tp, 2)}, fp={round(fp, 2)}, fn={round(fn, 2)}, tn={round(tn, 2)}, over_correction={round(over_correction)}")

        return result


# !/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
IndependentChunkMetric with safe concurrent updates of missing‐predictions JSON.

from filelock import FileLock, Timeout

class IndependentChunkMetric(CLEME):
    在 CLEME 基础上增加对 'missing predictions' 文件的并发安全写入。

    def evaluate_sample_correction(
        self,
        chunks_hyp: List["Chunk"],
        chunks_ref: List[List["Chunk"]],
        alpha: float,
    ) -> List[Dict[str, float]]:

        # 原始预测和缺失预测文件路径
        pred_path = "/data/zyh/Evaluation/CLEME/Cache/GJG15_and_SEEDA_fp_predictions.json"
        missing_pred_path = "/data/zyh/Evaluation/CLEME/Cache/GJG15_and_SEEDA_missing_predictions-for_run_results.json"
        lock = FileLock(missing_pred_path + ".lock", timeout=10)

        # 1) 读取已有 Predictions
        predictions: Dict[Tuple[str, str, str], int] = {}
        if os.path.exists(pred_path):
            with open(pred_path, 'r', encoding='utf-8') as pf:
                for rec in json.load(pf):
                    key = (
                        rec.get("Src", ""),
                        rec.get("Hypo_Chunk", ""),
                        rec.get("FP_noc_hyp", "")
                    )
                    predictions[key] = int(rec.get("Predicted", 0))

        # 2) 在锁保护下读取/更新 missing_records
        try:
            with lock:
                # 2.1 读已有 missing_records（如果文件不存在则初始化空列表）
                if os.path.exists(missing_pred_path):
                    with open(missing_pred_path, 'r', encoding='utf-8') as mf:
                        missing_records = json.load(mf)
                else:
                    missing_records = []

                missing_keys = {
                    (rec.get("Src", ""),
                     rec.get("Hypo_Chunk", ""),
                     rec.get("FP_noc_hyp", ""))
                    for rec in missing_records
                }

                # 3) 核心评估逻辑
                tp = fp = fn = tn = over_correction = 0.0

                for idx, chunk_hyp in enumerate(chunks_hyp):
                    cand_list = [grp[idx] for grp in chunks_ref]
                    length = max(len(chunk_hyp.src_tokens), len(chunk_hyp.tgt_tokens))

                    if chunk_hyp.type:
                        if chunk_hyp in cand_list:
                            w = self.weigher_tp(length)
                            tp += w
                            LOGGER.debug(f"{w:.2f} TP: {chunk_hyp}")
                        else:
                            # 提取可匹配字段
                            s = str(chunk_hyp)
                            m = re.match(
                                r"^Chunk\([^,]+,\s*type=\[.*?\],\s*tgt_idx=\d+,\s*(.*?)\)$", s
                            )
                            hypo_txt = m.group(1) if m else ""

                            # 在各参考组中查找预测
                            is_valid = False
                            for grp in chunks_ref:
                                src_j = chunk_list_to_text_with_skip(grp, chunk_hyp)
                                fnh_j = chunk_list_to_text_with_replacement(grp, chunk_hyp)
                                key = (src_j, hypo_txt, fnh_j)

                                # 新 key，加入 missing_records
                                if key not in predictions and key not in missing_keys:
                                    missing_keys.add(key)
                                    missing_records.append({
                                        "Src":       src_j,
                                        "Hypo_Chunk": hypo_txt,
                                        "FP_noc_hyp": fnh_j,
                                        "Predicted": None
                                    })

                                if predictions.get(key, 0) == 1:
                                    is_valid = True
                                    break

                            w_fp = self.weigher_fp(length)
                            if is_valid:
                                tp += self.weigher_tp(length)
                            else:
                                over_correction += w_fp
                                # 如果存在任一参考 chunk.type，算作 FP
                                if any(item.type for item in cand_list):
                                    fp += w_fp
                                    over_correction = 0.0

                    else:
                        if any_correct(cand_list):
                            w_fn = self.weigher_fn(length)
                            fn += w_fn
                            LOGGER.debug(f"{w_fn:.2f} FN: {chunk_hyp}")
                        else:
                            tn += 1.0

                # 2.2 原子写回 missing_records
                tmp_path = missing_pred_path + ".tmp"
                with open(tmp_path, 'w', encoding='utf-8') as outf:
                    json.dump(
                        missing_records,
                        outf,
                        ensure_ascii=False,
                        indent=4,
                        separators=(',', ': ')
                    )
                    outf.write('\n')
                os.replace(tmp_path, missing_pred_path)

        except Timeout:
            LOGGER.error(f"无法获取文件锁: {missing_pred_path}.lock，跳过 missing_records 更新。")

        # 4) 返回度量结果
        return [{
            KEY_TP: tp,
            KEY_FP: fp,
            KEY_FN: fn,
            KEY_TN: tn,
            KEY_OC: over_correction,
        }]
'''


class IndependentChunkMetric(CLEME):
    def evaluate_sample_correction(
            self,
            chunks_hyp: List[Chunk],
            chunks_ref: List[List[Chunk]],
            alpha,
    ) -> List[Dict[str, int]]:

        pred_path = "Cache-JLEV_judged_edits.json"
        predictions: Dict[Tuple[str, str, str], int] = {}
        if os.path.exists(pred_path):
            with open(pred_path, 'r', encoding='utf-8') as pf:
                # 整体读入一个 JSON 数组
                records = json.load(pf)
                # 构造 (Src, Hypo_Chunk) -> Predicted 映射
                for rec in records:
                    src = rec.get("Src", "")
                    hc = rec.get("Hypo_Chunk", "")
                    fnh = rec.get("FP_noc_hyp", "")
                    predictions[(src, hc, fnh)] = int(rec.get("Predicted", 0))

        result = []
        tp, fp, fn, tn = 0, 0, 0, 0
        over_correction = 0

        for chunk_idx, chunk_hyp in enumerate(chunks_hyp):
            cand_chunk_list = [x[chunk_idx] for x in chunks_ref]
            chunk_len = max(len(chunk_hyp.src_tokens), len(chunk_hyp.tgt_tokens))
            if chunk_hyp.type:
                if chunk_hyp in cand_chunk_list:
                    weight = self.weigher_tp(chunk_len)
                    tp += weight
                    LOGGER.debug(f"{round(weight, 2)} TP: {chunk_hyp} || {cand_chunk_list}")
                else:
                    # weight = self.weigher_fp(chunk_len)
                    # fp += weight
                    s = str(chunk_hyp)
                    pattern = r"^Chunk\([^,]+,\s*type=\[.*?\],\s*tgt_idx=\d+,\s*(.*?)\)$"
                    extracted_chunk_hyp = re.match(pattern, s).group(1)

                    # 遍历每个参考组，只要有一种 (src_j, extracted) 的预测是 1
                    is_valid = False
                    for grp in chunks_ref:
                        src_j = chunk_list_to_text_with_skip(grp, chunk_hyp)
                        fnh_j = chunk_list_to_text_with_replacement(grp, chunk_hyp)
                        key = (src_j, extracted_chunk_hyp, fnh_j)
                        if predictions.get(key, 0) == 1:
                            is_valid = True
                            break

                    weight = self.weigher_fp(chunk_len)

                    if is_valid:
                        tp += self.weigher_tp(chunk_len)
                    else:
                        over_correction += weight
                        for item in cand_chunk_list:
                            if item.type:
                                fp += weight
                                over_correction = 0
                                break

            else:
                if any_correct(cand_chunk_list):
                    weight = self.weigher_fn(chunk_len)
                    fn += weight
                    LOGGER.debug(f"{round(weight, 2)} FN: {chunk_hyp} || {cand_chunk_list}")
                else:
                    weight = 1.00
                    tn += weight
                    # LOGGER.debug(f"{round(weight, 2)} TN: {chunk_hyp}")

        # flush 剩余的 batch 推理
        # flush_inference(self.tokenizer, self.model, self.pred_cache)

        result.append({
            KEY_TP: tp,
            KEY_FP: fp,
            KEY_FN: fn,
            KEY_TN: tn,
            KEY_OC: over_correction,
        })
        LOGGER.debug(f"tp={round(tp, 2)}, fp={round(fp, 2)}, fn={round(fn, 2)}, tn={round(tn, 2)}")
        return result