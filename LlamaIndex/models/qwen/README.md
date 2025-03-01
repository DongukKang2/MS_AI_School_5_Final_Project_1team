---
language:
- zh
- en
license: gpl-3.0
tags:
- qwen
- uncensored
base_model:
- Qwen/Qwen2.5-7B-Instruct
datasets:
- NobodyExistsOnTheInternet/ToxicQAFinal
- anthracite-org/kalo-opus-instruct-22k-no-refusal
- Orion-zhen/dpo-toxic-zh
- unalignment/toxic-dpo-v0.2
- Crystalcareai/Intel-DPO-Pairs-Norefusals
pipeline_tag: text-generation
model-index:
- name: Qwen2.5-7B-Instruct-Uncensored
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: IFEval (0-Shot)
      type: HuggingFaceH4/ifeval
      args:
        num_few_shot: 0
    metrics:
    - type: inst_level_strict_acc and prompt_level_strict_acc
      value: 72.04
      name: strict accuracy
    source:
      url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard?query=Orion-zhen/Qwen2.5-7B-Instruct-Uncensored
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: BBH (3-Shot)
      type: BBH
      args:
        num_few_shot: 3
    metrics:
    - type: acc_norm
      value: 35.83
      name: normalized accuracy
    source:
      url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard?query=Orion-zhen/Qwen2.5-7B-Instruct-Uncensored
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: MATH Lvl 5 (4-Shot)
      type: hendrycks/competition_math
      args:
        num_few_shot: 4
    metrics:
    - type: exact_match
      value: 1.36
      name: exact match
    source:
      url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard?query=Orion-zhen/Qwen2.5-7B-Instruct-Uncensored
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: GPQA (0-shot)
      type: Idavidrein/gpqa
      args:
        num_few_shot: 0
    metrics:
    - type: acc_norm
      value: 7.05
      name: acc_norm
    source:
      url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard?query=Orion-zhen/Qwen2.5-7B-Instruct-Uncensored
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: MuSR (0-shot)
      type: TAUR-Lab/MuSR
      args:
        num_few_shot: 0
    metrics:
    - type: acc_norm
      value: 13.58
      name: acc_norm
    source:
      url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard?query=Orion-zhen/Qwen2.5-7B-Instruct-Uncensored
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: MMLU-PRO (5-shot)
      type: TIGER-Lab/MMLU-Pro
      config: main
      split: test
      args:
        num_few_shot: 5
    metrics:
    - type: acc
      value: 38.07
      name: accuracy
    source:
      url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard?query=Orion-zhen/Qwen2.5-7B-Instruct-Uncensored
      name: Open LLM Leaderboard
---

# Qwen2.5-7B-Instruct-Uncensored

This model is an uncensored fine-tune version of Qwen2.5-7B-Instruct. However, I can still notice that though uncensored, the model fails to generate detailed descriptions on certain extreme scenarios, which might be associated with deletion on some pretrain datasets in Qwen's pretraining stage.

Check out my roleplay&writing enhanced model based on this model: [Orion-zhen/Meissa-Qwen2.5-7B-Instruct](https://huggingface.co/Orion-zhen/Meissa-Qwen2.5-7B-Instruct)

## Traning details

I used SFT + DPO to ensure uncensorment as well as trying to maintain original model's capabilities.

- SFT:
  - NobodyExistsOnTheInternet/ToxicQAFinal
  - anthracite-org/kalo-opus-instruct-22k-no-refusal
- DPO:
  - Orion-zhen/dpo-toxic-zh
  - unalignment/toxic-dpo-v0.2
  - Crystalcareai/Intel-DPO-Pairs-Norefusals
# [Open LLM Leaderboard Evaluation Results](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
Detailed results can be found [here](https://huggingface.co/datasets/open-llm-leaderboard/details_Orion-zhen__Qwen2.5-7B-Instruct-Uncensored)

|      Metric       |Value|
|-------------------|----:|
|Avg.               |27.99|
|IFEval (0-Shot)    |72.04|
|BBH (3-Shot)       |35.83|
|MATH Lvl 5 (4-Shot)| 1.36|
|GPQA (0-shot)      | 7.05|
|MuSR (0-shot)      |13.58|
|MMLU-PRO (5-shot)  |38.07|

