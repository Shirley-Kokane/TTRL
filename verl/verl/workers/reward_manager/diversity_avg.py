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
from sympy import div
import openai
import torch
from sklearn.metrics.pairwise import cosine_similarity

from verl import DataProto
from verl.utils.reward_score.math_batch import compute_score_batched

openai.api_key = os.environ.get("OPENAI_API_KEY")


class DiversityAvgRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", mode="train", **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs
        self.mode = mode

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        prompts_str = [self.tokenizer.decode(p, skip_special_tokens=True) for p in prompt_ids]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        embedding_cache = {}
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)
            if response_str not in embedding_cache:
                response = openai.embeddings.create(input=response_str, model="text-embedding-3-small")
                embedding_cache[response_str] = response.data[0].embedding

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
                #**self.reward_kwargs,
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

        prompt_to_indices = defaultdict(list)
        for idx, prompt in enumerate(prompts_str):
            prompt_to_indices[prompt].append(idx)
            
        print("Reaching verify")
        diversity_scores = [0.0] * len(responses_str)
        for indices in prompt_to_indices.values():
            if len(indices) <= 1:
                continue

            group_completions = [responses_str[i] for i in indices]
            embeddings = [embedding_cache[completion] for completion in group_completions]
            
            # Calculate average embedding for all completions of this prompt
            avg_embedding = np.mean(embeddings, axis=0)
            
            # Calculate cosine similarity between each completion and the average embedding
            for idx, i in enumerate(indices):
                completion_embedding = embeddings[idx]
                # Compute cosine similarity between this completion and the average embedding
                similarity = np.dot(completion_embedding, avg_embedding) / (
                    np.linalg.norm(completion_embedding) * np.linalg.norm(avg_embedding)
                )
                
                if not np.isnan(similarity):
                    # Diversity score is 1 minus the similarity (higher similarity = lower diversity)
                    diversity_scores[i] = 1 - abs(similarity)

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
                combined_scores.append( 0*acc_score +  div_score)

        return combined_scores

    def _compute_eval_reward(self, data):
        prompt_ids = data.batch["prompts"]
        prompts_str = [self.tokenizer.decode(p, skip_special_tokens=True) for p in prompt_ids]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        print("Does it reach here? ", len(data))
        responses_str = []
        embedding_cache = {}
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)
            if response_str not in embedding_cache:
                response = openai.embeddings.create(input=response_str, model="text-embedding-3-small")
                embedding_cache[response_str] = response.data[0].embedding

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
                #**self.reward_kwargs,
            )
        else:
            accuracy_scores = self.compute_score(
                data_sources=data_sources,
                solution_strs=responses_str,
                ground_truths=ground_truths,
                extra_infos=extras,
                **self.reward_kwargs,
            )
        
        return accuracy_scores

    def __call__(self, data: DataProto, return_dict=False):
        
        if self.mode == "train":
            scores = self.verify(data)
        elif self.mode == "eval":
            scores = self._compute_eval_reward(data)
        else:
            raise ValueError(f"Mode {self.mode} is not supported for DiversityRewardManager")

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
