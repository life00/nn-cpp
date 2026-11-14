#include <iostream>
using namespace std;

struct Config {
  int seed{};
  int epochs{};
  string train_vectors;
  string train_labels;
  string test_vectors;
};

int main() {
  // 0. config: seed, epochs, data paths
  Config cfg = Config{
      .seed = 0, // allowed: 0,1,2,3,42
      .epochs = 10,
      .train_vectors = "data/fashion_mnist_train_vectors.csv",
      .train_labels = "data/fashion_mnist_train_labels.csv",
      .test_vectors = "data/fashion_mnist_test_vectors.csv",
  };

  // 1. load train vectors + labels

  // 2. carve validation slice out of train
  // 3. create network
  // 4. define loss function
  // 5. define optimizer
  // 6. training loop until epoch == n:
  //    shuffle order with the same RNG
  //    for each batch:
  //      reset gradients
  //      forward pass
  //      loss
  //      backpropagate
  //      optimize weights
  //    evaluate on validation
  //    print epoch, val accuracy, seconds
  //    check 600s budget
  // 7. write train predictions
  // 8. write test predictions

  cout << "thinking...";
  return 0;
}
