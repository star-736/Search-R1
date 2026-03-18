# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Reward function based on F1 score with search and answer penalties.
Reference: The user's pseudocode for token-level F1 reward computation.
"""

import re
import string
import random
from typing import List


def normalize_answer(s: str) -> str:
    """Normalize answer string for fair comparison."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    return normalize_answer(text).split()


def compute_f1(pred_answer: str, gt_answers: List[str]) -> float:
    """
    Compute token-level F1 score between predicted answer and ground truth answers.
    Returns the maximum F1 score among all ground truth answers.
    
    Args:
        pred_answer: Predicted answer string
        gt_answers: List of ground truth answer strings
    
    Returns:
        F1 score (0.0 to 1.0)
    """
    if isinstance(gt_answers, str):
        gt_answers = [gt_answers]
    
    pred_tokens = set(tokenize(pred_answer))
    if len(pred_tokens) == 0:
        return 0.0
    
    max_f1 = 0.0
    
    for gt_answer in gt_answers:
        gt_tokens = set(tokenize(gt_answer))
        
        if len(gt_tokens) == 0:
            continue
        
        # Compute intersection
        common_tokens = pred_tokens & gt_tokens
        
        # Compute precision and recall
        precision = len(common_tokens) / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        recall = len(common_tokens) / len(gt_tokens) if len(gt_tokens) > 0 else 0.0
        
        # Compute F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        max_f1 = max(max_f1, f1)
    
    return max_f1


def count_search_operations(text: str) -> int:
    """Count the number of <search> operations in the text."""
    search_pattern = r"<search>.*?</search>"
    matches = re.findall(search_pattern, text, re.DOTALL)
    return len(matches)


def extract_solution(solution_str: str) -> str:
    """
    Extract the answer from <answer> tags.
    Returns empty string if no answer is found.
    """
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return empty string (no valid answer)
    if len(matches) <= 1:
        return ""
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def is_valid_sequence(text: str) -> bool:
    """
    Validate the format for fast thinking template.
    
    Valid sequence pattern:
    - Can start with free text (implicit thinking)
    - Can search: <search> query </search> -> <information> results </information>
    - Can repeat search multiple times
    - Must end with: <answer> final answer </answer>
    """
    # Find the position of "<|im_start|>assistant" with potential whitespace
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    
    if not assistant_match:
        return False
    
    # Extract the content after the assistant marker
    start_pos = assistant_match.end()
    content = text[start_pos:]
    
    # Check for balanced tags (only search, information, answer - no think)
    tags_to_check = ["search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False
    
    # Check that there are no <think> tags
    think_open = len(re.findall(r"<think>", content))
    if think_open > 0:
        return False
    
    # Check sequence pattern
    split_pattern = r"(</?(?:search|information|answer)>)"
    parts = re.split(split_pattern, content)
    
    state = "start"
    
    for part in parts:
        if not part.strip():
            continue
            
        if re.match(r"</?(?:search|information|answer)>", part):
            if part == "<search>" and state in ["start", "information"]:
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state in ["start", "information"]:
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False
    
    return state == "end"


def compute_score_f1(solution_str, ground_truth, alpha=0.1, beta=0.1, 
                     use_format_penalty=True, use_search_penalty=True):
    """
    Compute reward based on F1 score with penalties.
    
    Formula: reward = F1 - no_search_penalty - no_answer_penalty
    
    Args:
        solution_str: The solution text generated by model
        ground_truth: Dict containing 'target' with list of ground truth answers
        alpha: Penalty for not performing any search (default: 0.1)
        beta: Penalty for not providing an answer (default: 0.1)
        use_format_penalty: Whether to penalize invalid format
        use_search_penalty: Whether to penalize no search
    
    Returns:
        float: Final reward score
    """
    # Extract predicted answer
    pred_answer = extract_solution(solution_str)
    gt_answers = ground_truth['target']
    
    # 1. Compute F1 score
    f1 = compute_f1(pred_answer, gt_answers)
    
    # 2. Count search operations
    num_search = count_search_operations(solution_str)
    
    # 3. Apply penalties
    no_search_penalty = alpha if (use_search_penalty and num_search == 0) else 0.0
    no_answer_penalty = beta if (pred_answer == "") else 0.0
    
    # 4. Compute final reward
    reward = f1 - no_search_penalty - no_answer_penalty
    
    # Optional: additional format penalty
    if use_format_penalty and not is_valid_sequence(solution_str):
        # Invalid format gets additional penalty, but not less than 0
        reward = max(0.0, reward - 0.1)
    
    # Debug print (1/64 chance)
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"F1 Score: {f1:.4f}")
        print(f"Num Search: {num_search}")
        print(f"No Search Penalty: {no_search_penalty}")
        print(f"No Answer Penalty: {no_answer_penalty}")
        print(f"Pred Answer: '{pred_answer}'")
        print(f"GT Answers: {gt_answers}")
        print(f"Final Reward: {reward:.4f}")
    
    return reward


# Alias for backward compatibility
compute_score_em = compute_score_f1
