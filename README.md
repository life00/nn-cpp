# nn-cpp

A feed-forward neural network written from scratch in C++20 (no linear-algebra or ML libraries), trained with backpropagation and minibatch gradient descent to classify [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist). The network is a 784–128–10 MLP with ReLU then sigmoid. It reaches about 90 % test accuracy in roughly 5 minutes on a single CPU core.

The math behind `src/net.cpp` (forward pass, loss, backpropagation, update rule, and how each symbol maps to the code) is derived in [docs/backprop_derivation.pdf](docs/backprop_derivation.pdf).

## Layout

```
.
├── src/
│   ├── main.cpp        Config and the training pipeline
│   ├── io.{hpp,cpp}    Dataset: CSV loading, validation split, accuracy, prediction export
│   ├── net.{hpp,cpp}   Network: forward, backward, gradient step, predict
│   └── tests/
│       └── xor.cpp     XOR sanity check (not part of the main build)
├── docs/
│   └── backprop_derivation.{tex,pdf}   formulas behind src/net.cpp
├── evaluator/          accuracy script for prediction files
└── data/               Fashion-MNIST CSVs (not tracked)
```

## Usage

Place `fashion_mnist_{train,test}_{vectors,labels}.csv` in `data/`, then:

```bash
g++ -std=c++20 -O3 src/*.cpp -o network
./network   # writes train_predictions.csv and test_predictions.csv
python3 evaluator/evaluate.py test_predictions.csv data/fashion_mnist_test_labels.csv
```
