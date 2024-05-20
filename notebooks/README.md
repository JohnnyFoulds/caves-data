# Multi-Label Classification

## Specialized Pre-trained Model

- [COVID-Twitter-BERT: A Natural Language Processing Model to Analyse COVID-19 Content on Twitter](https://arxiv.org/abs/2005.07503)
- [GitHub: COVID-Twitter-BERT](https://github.com/digitalepidemiologylab/covid-twitter-bert)
- [COVID-Twitter-BERT: Pretraining](https://github.com/digitalepidemiologylab/covid-twitter-bert/blob/master/README_pretrain.md)
- [idea: Task-Adaptive Pretraining (TAPT)](https://chatgpt.com/share/539ed230-9381-4bd8-bc52-e21ad3609d8d)

### Adapting to Telecoms Domain

A range of studies have explored the adaptation of RoBERTa, BERT, and LLMs to specific domains and their use in downstream tasks. Zhang 2020 and Ye 2020 both emphasize the importance of domain-specific vocabulary and structure in the unlabeled data for improved performance. Lehecka 2020 and Chen 2023 propose modifications to the pooling layer and the selection of BERT layers for multi-label classification, respectively. Chen 2023 also introduces a label attention mechanism to leverage semantic information in labels. Ardehaly 2016 and Kurmi 2019 focus on domain adaptation, with the former using label proportions and the latter proposing an adversarial discriminator approach. Fallah 2022 evaluates thresholding methods and proposes architectures for classification layers to improve performance in multi-label text classification.

- `doc/references/Elicit - How can RoBERTa _or BERT_ or LLMs_ be adapted to a specific domain and then used to train downstream tasks like multi-label classi.csv`

- [Adopting neural language models for the telecom domain](https://www.ericsson.com/en/blog/2022/1/neural-language-models-telecom-domain)
- [Understanding Telecom Language Through Large Language Models](https://arxiv.org/abs/2306.07933)


### Domain Adapted NLI Model

One possible idea is to train a NLI model for zero-shot prediction specifically for the domain.

1. Start with a Pre-Trained NLI Model: Use facebook/bart-large-mnli or roberta-large-mnli.
2. Prepare Domain-Specific NLI Data: Ensure your dataset contains pairs of sentences with NLI labels.
3. Fine-Tune the Model: Fine-tune the pre-trained NLI model on your domain-specific data.
4. Zero-Shot Classification: Use the fine-tuned model with the zero-shot classification pipeline.

This approach leverages the strengths of an NLI model pre-trained on large-scale NLI data and adapts it to your domain with minimal adjustments.

https://chatgpt.com/share/4043f53e-ca97-4583-9114-23fb7592d028

To create a dataset like the following, I can use the LLM classification work done in [multi-intent-classification](https://github.com/JohnnyFoulds/multi-intent-classification/blob/feature/mutilclass_model/notebooks/01_intent_extraction/01-02_batch_classification.ipynb)

```text
complaint,aspect,label
"The product stopped working after a week.", "Product Quality", 0
"I had to wait two weeks for my order.", "Delivery Issues", 0
"The customer service was unhelpful.", "Customer Service", 0
```

