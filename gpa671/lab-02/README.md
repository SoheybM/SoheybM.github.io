# Lab 02 — SVM, k-NN & Naive Bayes

The second GPA671 laboratory broadened the classification work to several supervised-learning methods and compared linear/nonlinear decision models on synthetic and image datasets.

## Main topics

- soft-margin linear SVM
- support vectors, margins and the effect of the regularization parameter `C`
- Gaussian/RBF kernels and dual-form decision functions
- manual grid search and cross-validation
- CIFAR-10 SVM experiments
- MNIST binary and 10-class classification
- k-nearest neighbours implemented with NumPy
- Gaussian Naive Bayes implemented with NumPy
- feature scaling and numerical stability

## Selected results

A few representative results from the completed notebook:

- **Two-moons dataset:** RBF SVM test accuracy **95.5%**, compared with **88.5%** for the linear SVM.
- **MNIST, 10 classes:** RBF SVM test accuracy **95.2%**, compared with **91.9%** for the linear SVM.
- **Wine + k-NN:** mean accuracy across 20 stratified splits improved from **71.7% ± 2.7%** without normalization to **95.8% ± 1.4%** after normalization.
- **k-NN vs Gaussian Naive Bayes:** **96.8% ± 1.5%** vs **95.7% ± 1.8%** across the same 20 splits in the laboratory experiment.

The laboratory also explored underfitting/overfitting with different `C` and `gamma` values and the numerical-stability advantage of log-probabilities in Naive Bayes.

## Public source

`source.py` contains the main custom implementations:
- training-set standardization helper
- manual k-fold index generation
- RBF kernel matrix
- dual SVM decision function
- `kNN`
- `NaiveBayes`

Course-provided starter code, datasets, teaching statements and unfinished optional cells are not redistributed.
