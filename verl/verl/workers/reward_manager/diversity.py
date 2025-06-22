# Copyright 2025 Individual Contributor: Mert Unsal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import defaultdict

import numpy as np
import openai
import torch
from sklearn.metrics.pairwise import cosine_similarity

from verl import DataProto
from verl.utils.reward_score.math_batch import compute_score_batched

class DiversityRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        prompts_str = [self.tokenizer.decode(p, skip_special_tokens=True) for p in prompt_ids]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        # 1. Compute original accuracy scores
        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get("extra_info", [None] * len(data))

        if self.compute_score is None:
            accuracy_scores = compute_score_batched(
                data_sources=data_sources,
                solution_strs=responses_str,
                ground_truths=ground_truths,
                extra_infos=extras,
                **self.reward_kwargs,
            )
        else:
            accuracy_scores = self.compute_score(
                data_sources=data_sources,
                solution_strs=responses_str,
                ground_truths=ground_truths,
                extra_infos=extras,
                **self.reward_kwargs,
            )

        # 2. Compute diversity scores
        openai.api_key = os.environ.get("OPENAI_API_KEY")

        unique_responses = list(set(responses_str))
        embedding_cache = {}
        if unique_responses:
            response = openai.embeddings.create(input=unique_responses, model="text-embedding-3-small")
            for resp_str, embedding_data in zip(unique_responses, response.data):
                embedding_cache[resp_str] = embedding_data.embedding

        prompt_to_indices = defaultdict(list)
        for idx, prompt in enumerate(prompts_str):
            prompt_to_indices[prompt].append(idx)

        diversity_scores = [0.0] * len(responses_str)
        for indices in prompt_to_indices.values():
            if len(indices) <= 1:
                continue

            group_completions = [responses_str[i] for i in indices]
            embeddings = [embedding_cache[completion] for completion in group_completions]

            sim_matrix = cosine_similarity(embeddings)
            n = len(embeddings)

            for idx, i in enumerate(indices):
                if n > 1:
                    avg_sim = (np.sum(sim_matrix[idx]) - 1) / (n - 1)
                else:
                    avg_sim = 0.0

                if not np.isnan(avg_sim):
                    diversity_scores[i] = 1 - abs(avg_sim)

        del embedding_cache
        # 3. Combine scores
        combined_scores = []
        for i in range(len(accuracy_scores)):
            acc_score = accuracy_scores[i]
            div_score = diversity_scores[i]

            # If accuracy_score is a dictionary, extract the 'score' and add diversity
            if isinstance(acc_score, dict):
                combined_score_dict = acc_score.copy()
                combined_score_dict["score"] = acc_score.get("score", 0.0) + div_score
                combined_score_dict["diversity_score"] = div_score
                combined_scores.append(combined_score_dict)
            else:
                combined_scores.append(acc_score + div_score)

        return combined_scores

    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        scores = self.verify(data)
        rewards = []
        already_printed = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
