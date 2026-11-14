#include <iostream>
using namespace std;

int main() {
  // 0. parse args: seed, epochs, paths
  // 1. load train vectors + labels
  // 2. carve validation slice out of train
  // 3. seed the RNG
  // 4. create network
  // 5. define loss function
  // 6. define optimizer
  // 7. training loop until epoch == n:
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
  // 8. write train predictions
  // 9. write test predictions
  cout << "thinking...";
}
