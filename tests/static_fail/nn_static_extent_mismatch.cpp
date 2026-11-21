#include <nn/nn.hpp>

int main() {
  // this should fail to compile: W has 3 columns, x has 4 elements
  [[maybe_unused]] auto y = nn::matvec(nn::Mat<2, 3>{}, nn::Vec<4>{});
  return 0;
}
