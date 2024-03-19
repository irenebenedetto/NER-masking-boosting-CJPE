# Boosting court judgment prediction and explanation using legal entities
This repository contains the code for the paper **"Boosting court judgment prediction and explanation using legal entities"** published in Artificial Intelligence and Law.

The repository is organized as follows:

- `legal_ner`: contains the code and the data for the **Legal Named Entity Recognition (L-NER)**
- `legal_cpje`: contains the code and the data for the **Court Judgment Prediction and Explanation (CJPE)**

Further details are available in the README files of the subfolders.

## Citation
If you use this code, please cite the following paper:

```bibtex
@inproceedings{Benedetto2024,
    title = "Boosting court judgment prediction and explanation using legal entities",
    author = "Benedetto, Irene  and
      Koudounas, Alkis  and
      Vaiani, Lorenzo  and
      Pastor, Eliana  and
      Baralis, Elena  and
      Cagliero, Luca  and
      Tarasconi, Francesco",
    booktitle = "Artificial Intelligence and Law",
    month = mar,
    year = "2024",
    url = "https://doi.org/10.1007/s10506-024-09397-8",
    doi = "10.18653/v1/2023.semeval-1.194",
    abstract = "The automatic prediction of court case judgments using Deep Learning and Natural Language Processing is challenged by the variety of norms and regulations, the inherent complexity of the forensic language, and the length of legal judgments. Although state-of-the-art transformer-based architectures and Large Language Models (LLMs) are pre-trained on large-scale datasets, the underlying model reasoning is not transparent to the legal expert. This paper jointly addresses court judgment prediction and explanation by not only predicting the judgment but also providing legal experts with sentence-based explanations. To boost the performance of both tasks we leverage a legal named entity recognition step, which automatically annotates documents with meaningful domain-specific entity tags and masks the corresponding fine-grained descriptions. In such a way, transformer-based architectures and Large Language Models can attend to in-domain entity-related information in the inference process while neglecting irrelevant details. Furthermore, the explainer can boost the relevance of entity-enriched sentences while limiting the diffusion of potentially sensitive information. We also explore the use of in-context learning and lightweight fine-tuning to tailor LLMs to the legal language style and the downstream prediction and explanation tasks. The results obtained on a benchmark dataset from the Indian judicial system show the superior performance of entity-aware approaches to both judgment prediction and explanation.",
}

@inproceedings{benedetto-etal-2023-politohfi,
    title = "{P}oli{T}o{HFI} at {S}em{E}val-2023 Task 6: Leveraging Entity-Aware and Hierarchical Transformers For Legal Entity Recognition and Court Judgment Prediction",
    author = "Benedetto, Irene  and
      Koudounas, Alkis  and
      Vaiani, Lorenzo  and
      Pastor, Eliana  and
      Baralis, Elena  and
      Cagliero, Luca  and
      Tarasconi, Francesco",
    editor = {Ojha, Atul Kr.  and
      Do{\u{g}}ru{\"o}z, A. Seza  and
      Da San Martino, Giovanni  and
      Tayyar Madabushi, Harish  and
      Kumar, Ritesh  and
      Sartori, Elisa},
    booktitle = "Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.semeval-1.194",
    doi = "10.18653/v1/2023.semeval-1.194",
    pages = "1401--1411",
    abstract = "The use of Natural Language Processing techniques in the legal domain has become established for supporting attorneys and domain experts in content retrieval and decision-making. However, understanding the legal text poses relevant challenges in the recognition of domain-specific entities and the adaptation and explanation of predictive models. This paper addresses the Legal Entity Name Recognition (L-NER) and Court judgment Prediction (CPJ) and Explanation (CJPE) tasks. The L-NER solution explores the use of various transformer-based models, including an entity-aware method attending domain-specific entities. The CJPE proposed method relies on hierarchical BERT-based classifiers combined with local input attribution explainers. We propose a broad comparison of eXplainable AI methodologies along with a novel approach based on NER. For the L-NER task, the experimental results remark on the importance of domain-specific pre-training. For CJP our lightweight solution shows performance in line with existing approaches, and our NER-boosted explanations show promising CJPE results in terms of the conciseness of the prediction explanations.",
}
``` 

## License
This code is released under the Apache 2.0 license. Please take a look at the [LICENSE](LICENSE) file for more details.

## Contact Information
If you need help or issues using the code, please submit a GitHub issue.

For other communications related to this repository, please contact [Irene Benedetto](mailto:irene.benedetto@polito.it) or [Alkis Koudounas](mailto:alkis.koudounas@polito.it).
