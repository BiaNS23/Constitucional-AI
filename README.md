Abstract

This project investigates the application of Constitutional Artificial Intelligence (CAI) to large language models in the Portuguese language, focusing on safety, ethical alignment, and cultural sensitivity. Inspired by Anthropic’s Constitutional AI framework, this work explores how constitutional principles can be adapted to Portuguese-speaking contexts using the Tucano family of language models.
Rather than relying solely on fine-tuning, this research emphasizes a hybrid approach that combines prompt-based critique generation, few-shot learning, rule-based post-processing, and external sociolinguistic resources. The project demonstrates that integrating culturally grounded datasets—such as the HateBR corpus—significantly improves the detection of harmful content in Portuguese, even without full reinforcement learning fine-tuning.

Motivation

Most alignment and safety research for large language models is developed primarily for English. Applying these methods directly to Portuguese often fails to capture linguistic nuance, cultural context, and region-specific ethical concerns.

This project aims to:

- Promote technological sovereignty in Portuguese NLP;

- Explore safe and responsible AI development beyond English-centric approaches;

- Investigate how constitutional principles must be adapted for multilingual and multicultural settings.

Research Objectives

The initial objectives of this research were:

- Design a constitutional framework in Portuguese, adapted to linguistic and cultural contexts;

- Implement an open-source pipeline for Constitutional AI experimentation;

- Evaluate the impact of CAI techniques on Portuguese language models;

- Derive best practices for alignment in non-English languages.
.

Methodology Overview

The research follows a modular and iterative methodology inspired by Constitutional AI:

1. Constitutional Critique Generation

- Use of Tucano-160m and Tucano-2b4-Instruct models;

- Automatic generation of critiques identifying harmful, unethical, or illegal content;

- Iterative prompt engineering to reduce hallucination and unstructured outputs.

2. Prompt Reformulation & Few-Shot Learning

- Reformulation of tasks to critique questions instead of answers;

- Few-shot examples to enforce structured and concise critiques;

- Controlled generation using temperature and token constraints.

3. Hybrid Post-Processing

- Rule-based classification using safety-related keywords;

- Integration of sociolinguistic knowledge from external datasets.

HateBR Integration

- To enhance cultural and linguistic sensitivity, the HateBR corpus was incorporated into the pipeline.

- Over 2,860 offensive and hate-related keywords were extracted;

- Keywords were categorized (racism, sexism, xenophobia, ideology, etc.);

- Critiques containing HateBR terms were automatically flagged as harmful.

This hybrid LLM + rule-based approach significantly improved robustness and contextual awareness.

Evaluation

A custom evaluation pipeline was implemented to measure performance:

- Binary classification: harmful vs neutral;

- Metrics: accuracy, false positives, false negatives;

- Logging and version tracking via CSV files.

Results
Metric	Before HateBR	After HateBR
Accuracy	42.86%	78.57%
False Negatives	High	0
False Positives	Moderate	Low

These results indicate that external sociocultural knowledge is essential for effective alignment in Portuguese.

Key Contributions

- Practical implementation of Constitutional AI concepts in Portuguese;

- Identification of limitations of prompt-only alignment strategies;

- Hybrid alignment pipeline combining LLMs and rule-based filters;

- Open-source scripts for generation, critique, evaluation, and logging;

- Empirical evidence supporting culturally grounded alignment methods.

Limitations & Future Work

This project does not include a completed fine-tuning or RLHF phase.
Major challenges included:

- Distributed training instability with LoRA and gradient checkpointing;

- Limited supervision and time constraints.

Future extensions include:

- Supervised fine-tuning (SFT) of critique models;

- Reinforcement learning from AI feedback (RLAIF);

- Fuzzy matching for obfuscated hate speech;

- Expansion to other Portuguese-speaking regions.

Repository Structure
ia_constitucional/
├── scripts/
│   ├── generate.py
│   ├── critic.py
│   ├── evaluate_model.py
├── data/
│   ├── prompts.json
│   ├── criticized_responses.json
│   ├── hatebr_keywords.json
│   └── evaluation_set.json
├── checkpoints/
├── logs/
└── README.md

Technologies Used

- Python

- Hugging Face Transformers

- PEFT (LoRA)

- Accelerate

- PyTorch

 -HateBR Corpus

Author

Beatriz Nascimento

Undergraduate Researcher in Mechatronics Engineering

University of São Paulo (USP)

Acknowledgments

This project was conducted as part of an undergraduate research initiative focused on AI safety and ethical alignment. The work builds upon publicly available research from Anthropic and the Tucano model ecosystem.
