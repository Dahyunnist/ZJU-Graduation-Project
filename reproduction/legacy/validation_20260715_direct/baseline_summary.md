# Adult Baseline Result

## Setup

- Dataset: UCI Adult
- Generator: CTGAN, epochs=20
- Real rows evaluated: 3000
- Synthetic rows evaluated: 1000
- Classifier: scikit-learn LogisticRegression

## SDMetrics

- Overall quality score: 0.7733
- Column Shapes: 0.8397
- Column Pair Trends: 0.7068

## Classifier Detector

- Accuracy: 0.7600
- AUROC: 0.8490

## Classification Report

```text
              precision    recall  f1-score   support

           0     0.7481    0.7840    0.7656       250
           1     0.7731    0.7360    0.7541       250

    accuracy                         0.7600       500
   macro avg     0.7606    0.7600    0.7599       500
weighted avg     0.7606    0.7600    0.7599       500
```
