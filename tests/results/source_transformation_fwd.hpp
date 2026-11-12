std::pair<double, double> func_01_d(double a, double a_d) {
  return std::make_pair(a * a, a * a_d + a * a_d);
}

std::pair<double, double> func_02_d(double a, double a_d) {
  double b_d = a * a_d + a * a_d;
  double b = a * a;
  return std::make_pair(b * b, b * b_d + b * b_d);
}

std::pair<double, double> func_03_d(double a, double a_d) {
  a_d = a * a_d + a * a_d;
  a = a * a;
  return std::make_pair(a * a, a * a_d + a * a_d);
}

std::pair<double, double> func_call_d(double a, double a_d) {
  return std::make_pair(a * a, a * a_d + a * a_d);
}

std::pair<double, double> func_04_d(double a, double a_d) {
  auto [temp_1, temp_1_d] = func_call_d(a, a_d);
  return std::make_pair(temp_1 * a, a * temp_1_d + temp_1 * a_d);
}

std::pair<double, double> func_05_d(double a, double a_d) {
  auto [b, b_d] = func_call_d(a, a_d);
  return std::make_pair(b * a, a * b_d + b * a_d);
}

std::pair<double, double> func_06_d(double a, double a_d) {
  double b_d = a_d;
  double b = a;
  [b, b_d] = func_call_d(b, b_d);
  return std::make_pair(b * a, a * b_d + b * a_d);
}

