# Training Runs

## sentence-transformers/paraphrase-mpnet-base-v2

### one-vs-rest 

#### 8 samples per class

```python
model = SetFitModel.from_pretrained(
    model_name,
    multi_target_strategy="one-vs-rest")
```

```text
-- 1 (0.76) --
F1 Score (Macro-Average)   	0.381
F1 Score (Weighted-Average)	0.419
Average Jaccard Similarity 	0.372
Subset Accuracy            	0.300

-- 2 (0.7) --
F1 Score (Macro-Average)   	0.412
F1 Score (Weighted-Average)	0.461
Average Jaccard Similarity 	0.403
Subset Accuracy            	0.322

-- 3 (0.72) --
F1 Score (Macro-Average)   	0.400
F1 Score (Weighted-Average)	0.440
Average Jaccard Similarity 	0.387
Subset Accuracy            	0.306

-- 4 (0.77) --
F1 Score (Macro-Average)   	0.390
F1 Score (Weighted-Average)	0.427
Average Jaccard Similarity 	0.373
Subset Accuracy            	0.301

-- 5 (0.75) --
F1 Score (Macro-Average)   	0.389
F1 Score (Weighted-Average)	0.434
Average Jaccard Similarity 	0.384
Subset Accuracy            	0.301
```


#### 16 samples per class

```text
-- 1 (0.72) --
F1 Score (Macro-Average)   	0.499
F1 Score (Weighted-Average)	0.551
Average Jaccard Similarity 	0.491
Subset Accuracy            	0.397

-- 2 (0.77) --
F1 Score (Macro-Average)   	0.504
F1 Score (Weighted-Average)	0.561
Average Jaccard Similarity 	0.509
Subset Accuracy            	0.409
```


#### differentiable head

##### 8 samples per class

```python
model = SetFitModel.from_pretrained(
    model_name,
    multi_target_strategy="one-vs-rest",
    use_differentiable_head=True,
    head_params={"out_features": len(vocab)},
)
```

```text
-- 1 (0.60) --
F1 Score (Macro-Average)   	0.323
F1 Score (Weighted-Average)	0.389
Average Jaccard Similarity 	0.278
Subset Accuracy            	0.171

-- 2 (0.44) --
F1 Score (Macro-Average)   	0.403
F1 Score (Weighted-Average)	0.436
Average Jaccard Similarity 	0.370
Subset Accuracy            	0.287

-- 5 (0.85) --
F1 Score (Macro-Average)   	0.408
F1 Score (Weighted-Average)	0.446
Average Jaccard Similarity 	0.387
Subset Accuracy            	0.308
```

## Alibaba-NLP/gte-large-en-v1.5

### one-vs-rest 

#### 16 samples per class

```text
-- 2 (0.99) --
F1 Score (Macro-Average)   	0.559
F1 Score (Weighted-Average)	0.619
Average Jaccard Similarity 	0.573
Subset Accuracy            	0.474
```