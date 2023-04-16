# Verifying effectiveness of KeyClass for clinical notes categorization
### Sayar Ghosh Roy and Atharv Chandratre

# Introduction
As a part of the CS 598 Deep Learning for Healthcare Course Final Project (Spring 2023), our aim is to verify the selected paper, titled `Classifying Unstructured Clinical Notes via Automatic Weak Supervision'. The goal of the paper is to alleviate the practice of manual diagnostic coding of clinical notes, a process that is time-consuming, expensive, and error-prone. To address this problem, the paper introduces KeyClass, a weakly-supervised text classification framework. KeyClass learns a text classification model from class label descriptions alone, eliminating the requirement for manually tagged documents. To assign code labels to specific texts, KeyClass leverages pre-trained language models and data programming frameworks. During the classification process, KeyClass creates certain interpretable heuristics based on keywords extracted from the available text data. Through the adoption of domain-specific language models, the KeyClass framework could also be tailored to classification tasks in other highly specialized domains. The comparison between KeyClass and other seminal weakly supervised text classification architectures demonstrates KeyClass's efficiency and adaptability.

# Claims from the original paper that we plan to evaluate
- The KeyClass model mines category-indicative terms from each ICD-9 category
- The KeyClass model achieves an F1-score of 0.62 for multilabel classification of the MIMIC-III clinical notes dataset
- The KeyClass model achieves the category-specific performances reported in the original paper
- (Additional Ablation) The proposed self-training (refining the downstream classifier using the complete training dataset) helps the overall classification performance on the MIMIC-III clinical notes dataset


Link to the original repo - https://github.com/autonlab/KeyClass
