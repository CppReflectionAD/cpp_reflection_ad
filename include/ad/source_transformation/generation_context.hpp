#pragma once

#include <ranges>

#include <meta>
#include <string>
#include <vector>

namespace ad {

struct GenerationContext {
  std::string indent = {};
  int temporary_counter = 0;

  std::string statements = {};

  std::vector<std::string> reverse_statements = {};

  std::vector<std::meta::info> required_functions = {};
  std::vector<std::meta::info> handled_functions = {};

  bool reverse_generation = false;
  std::vector<std::string> function_temporary_stack = {};

  constexpr std::string generateTemporary() {
    temporary_counter += 1;
    // TODO: use to_string: return "temp_" + std::to_string(temporary_counter);
    return "temp_" + int_to_string(temporary_counter);
  }

  constexpr std::string format(std::string line) {
    if (line.empty()) {
      return "";
    }
    return indent + line;
  }

  consteval void addStatement(std::string stmt) {
    statements += format(stmt) + "\n";
  }

  consteval void addReverseStatement(std::string stmt) {
    reverse_statements.push_back(format(stmt) + "\n");
  }

  consteval std::string flushStatements() {
    std::string r = {};
    std::swap(r, statements);
    return r;
  }

  consteval std::string flushReverseStatements() {
    std::string r = {};
    for (std::string const& cur : reverse_statements |  std::views::reverse) {
      r += cur;
    }
    reverse_statements.clear();
    return r;
  }

  consteval void addIndent() {
    indent += "  ";
  }

  consteval void removeIndent() {
    indent = indent.substr(2);
  }

  consteval void addRequiredFunction(std::meta::info func) {
    auto pos = std::find(handled_functions.begin(), handled_functions.end(), func);

    if (pos == handled_functions.end()) {
      handled_functions.push_back(func);
      required_functions.push_back(func);
    }
  }

  consteval std::vector<std::meta::info>  getRequiredFunctions() {
    std::vector<std::meta::info> r = {};
    std::swap(r, required_functions);
    return r;
  }

  consteval void pushFunctionTemporary(std::string const& name) {
    function_temporary_stack.push_back(name);
  }

  consteval std::string popFunctionTemporary() {
    std::string r = function_temporary_stack.back();
    function_temporary_stack.pop_back();
    return r;
  }

private:
  constexpr std::string int_to_string(int num)  {
    std::array<char, 12> buf{};
    auto [ptr, ec] = std::to_chars(buf.data(), buf.data() + buf.size() - 1, num);

    if (ec != std::errc{}) {
      throw "conversion failed";
    }

    std::string r = {};
    for (auto cur = buf.begin(); cur != ptr; cur++) {
      r += *cur;
    }

    return r;
  }
};

}