#include "../fixtures/functions/st_functions.hpp"
#include "source_transformation/string_compare.hpp"
#include "test_simple_include.hpp"

#include <ad/source_transformation/source_transformation.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
char const results[] =
{
#embed "../results/source_transformation_rws.hpp"
  ,'\0'
};
#pragma clang diagnostic pop

int main(int nargs, char** args) {

  std::string filename = "derivatives.hpp";

  if (nargs >= 2) {
    filename = args[1];
  }

  std::string data = ad::handleNamespaceReverse<^^test_functions>();

  std::ofstream out(filename);
  out << data;
  out.close();

  std::string result_str = results;
  stringCompare(data, result_str);

  TEST_END;
}