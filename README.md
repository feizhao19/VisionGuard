init# MH_Post_Hoc: Mitigating Image Captioning Hallucinations in Vision-Language Models  
### *A Post-Hoc Test-Time Adaptation Approach with Reinforcement Learning*

This repository contains the code for our paper **"Mitigating Image Captioning Hallucinations in Vision-Language Models: A Post-Hoc Test-Time Adaptation Approach with Reinforcement Learning"**, submitted to **ICME 2025**. Our approach introduces a lightweight, reinforcement learning-based method to dynamically mitigate hallucinations in Vision-Language Models (VLMs) during inference, operating as a post-hoc adaptation mechanism.

## Overview

Vision-Language Models (VLMs) are prone to hallucinations in their generated captions, where descriptions deviate from the actual visual content. This project proposes a novel **post-hoc Test-Time Adaptation (TTA)** framework, leveraging **reinforcement learning (RL)** to iteratively refine captions during inference. Our method eliminates the need for retraining or additional models, offering an efficient solution to improve caption accuracy and factual consistency.

### Key Features

- **Reinforcement Learning-Based Post-Hoc Adaptation:** Dynamically refines captions during inference by treating the VLM as a policy model in an RL framework.
- **Dual Reward Mechanism:**
  - **Semantic Alignment Score (SAS):** Measures alignment between visual and textual modalities.
  - **Non-Hallucination Probability (NHP):** Assesses the factual consistency of captions.
- **Lightweight Design:** Updates only LayerNorm gamma parameters (~0.003% of total model parameters) in the VLMs.
- **Custom CLIP-Based Evaluation Model:** Detects and evaluates hallucinations using learnable query tokens and a triplet loss function.
- **Post-Hoc Nature:** Parameters are reset after processing each sample, ensuring episodic adaptability and computational efficiency.

---


### Details on Folders
- **AMBER:** Contains evaluation scripts for the AMBER dataset, which serves as the benchmark for generative tasks.
- **model:** Includes the TTA implementations for two VLMs:
  - `blip`: TTA adaptation for **InstructBLIP**.
  - `llava`: TTA adaptation for **LLaVA**.
- **temp:** Auxiliary scripts for environment setup and testing.

---

## Methodology

### Reinforcement Learning Framework
Our TTA framework treats the VLM as a policy model in an RL setting. During inference, the model generates multiple captions, which are evaluated using:
- **SAS (Semantic Alignment Score):** Measures alignment between the image and the caption.
- **NHP (Non-Hallucination Probability):** Evaluates the factual consistency of captions.

These rewards guide the model to iteratively refine captions, reducing hallucination rates and improving factual accuracy. The lightweight design ensures only LayerNorm gamma parameters are updated, maintaining computational efficiency.

### Hallucination Detection Model
A CLIP-based hallucination evaluation model is employed to classify captions as hallucinated or non-hallucinated. It utilizes a single learnable query token updated through multi-stage cross-attention, with training guided by a triplet loss function.

---

## Dataset and Models

### Dataset
- **AMBER Dataset:** Used for evaluation, containing 1,004 samples designed for generative benchmarking.
- **PixelProse Dataset:** Curated subset of 30,000 samples for training and testing the hallucination evaluation model.

### Models
- **LLaVA-7B:** Adapted with TTA to improve hallucination mitigation.
- **InstructBLIP:** Enhanced with RL-based TTA for better caption accuracy.

---

## Results

- The **RL-based post-hoc TTA framework** achieves significant improvements in hallucination mitigation:
  - CHAIR scores reduced by **15.4%** on LLaVA-7B and **17.3%** on InstructBLIP.
- The **CLIP\textsubscript{Prompts + Triplet} model** delivers exceptional performance on hallucination detection:
  - **Object:** 86.5%, **Attribute:** 80.7%, **Relation:** 75.5%.

These results validate the effectiveness of our approach in reducing hallucinations and improving caption quality while maintaining computational efficiency.

---
## Acknowledgments
	1. The AMBER dataset authors for providing the evaluation benchmarks.
	2. NVIDIA for their GPU resources enabling efficient experimentation.