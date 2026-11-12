void func_01_b(double a, double& a_b, double ret_b) {

  // return a * a;
  a_b += a * ret_b;
  a_b += a * ret_b;
  ret_b = 0.0;
}

void func_02_b(double a, double& a_b, double ret_b) {
  double b_b = 0.0;
  double b = a * a;

  // return b * b;
  b_b += b * ret_b;
  b_b += b * ret_b;
  ret_b = 0.0;

  // double b = a * a;
  a_b += a * b_b;
  a_b += a * b_b;
  b_b = 0.0;
}

void func_03_b(double a, double& a_b, double ret_b) {
  store(a);
  a = a * a;

  // return a * a;
  a_b += a * ret_b;
  a_b += a * ret_b;
  ret_b = 0.0;

  // a = a * a;
  restore(a);
  a_b += a * a_b;
  a_b += a * a_b;
  a_b = 0.0;
}

void func_call_b(double a, double& a_b, double ret_b) {

  // return a * a;
  a_b += a * ret_b;
  a_b += a * ret_b;
  ret_b = 0.0;
}

void func_04_b(double a, double& a_b, double ret_b) {
  double temp_1_b = 0.0;
  double temp_1 = func_call(a);

  // return temp_1 * a;
  a_b += temp_1 * ret_b;
  temp_1_b += a * ret_b;
  ret_b = 0.0;

  // double temp_1 = func_call(a);
  func_call_b(a, a_b, temp_1_b);
  temp_1_b = 0.0;
}

void func_05_b(double a, double& a_b, double ret_b) {
  double b_b = 0.0;
  double b = func_call(a);

  // return b * a;
  a_b += b * ret_b;
  b_b += a * ret_b;
  ret_b = 0.0;

  // double b = func_call(a);
  func_call_b(a, a_b, b_b);
  b_b = 0.0;
}

void func_06_b(double a, double& a_b, double ret_b) {
  double b_b = 0.0;
  double b = a;
  store(b);
  b = func_call(b);

  // return b * a;
  a_b += b * ret_b;
  b_b += a * ret_b;
  ret_b = 0.0;

  // b = func_call(b);
  restore(b);
  func_call_b(b, b_b, b_b);
  b_b = 0.0;

  // double b = a;
  a_b += b_b;
  b_b = 0.0;
}

