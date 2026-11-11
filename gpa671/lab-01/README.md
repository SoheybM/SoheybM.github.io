# Lab 01 — Performance Metrics & Neural Networks

The first GPA671 laboratory introduced model evaluation and progressively built neural-network components from simple perceptrons to a multilayer perceptron (MLP).

## Main topics

- ROC and Precision–Recall curves, AUC and AUPR
- sigmoid perceptron and gradient-based weight updates
- reusable neural-network `Layer` abstraction
- multilayer perceptron with forward propagation and backpropagation
- comparison of architectures and learning rates
- nonlinear decision-boundary visualization
- regression/classification losses
- holdout and k-fold validation utilities

## Selected experiment

Several MLP configurations were compared on the small laboratory dataset:

| Configuration | Final accuracy | Epochs |
| --- | ---: | ---: |
| 2-2-1, lr=0.3 | 87.5% | 10,000 |
| 2-4-1, lr=0.3 | 100% | 3,015 |
| 2-8-1, lr=0.3 | 100% | 4,495 |
| 2-4-4-1, lr=0.3 | 100% | 6,685 |
| 2-8-1, lr=0.1 | 87.5% | 10,000 |
| 2-8-1, lr=0.5 | 100% | 2,777 |

The 2-8-1 network with a learning rate of 0.5 was then used to visualize a **nonlinear decision boundary**.

![MLP decision boundary](../../assets/gpa671/lab1-decision-boundary.jpg)


These figures come from a toy dataset with only eight examples and are presented to illustrate learning behaviour, not as a general performance claim.

## Public source

`source.py` contains the main public implementations extracted from the laboratory:
- `Perceptron`
- `Layer`
- `MLP`
- loss/cost helpers
- holdout and k-fold partition functions

Notebook outputs, teaching statements and datasets are not included.
