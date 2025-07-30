#include "calculator.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

class CalculatorTest : public ::testing::Test {
protected:
  void SetUp() override { calc = std::make_unique<Calculator>(); }

  void TearDown() override { calc.reset(); }

  std::unique_ptr<Calculator> calc;
};

TEST_F(CalculatorTest, AdditionTest) {
  EXPECT_EQ(calc->add(2, 3), 5);
  EXPECT_EQ(calc->add(-1, 1), 0);
  EXPECT_EQ(calc->add(0, 0), 0);
}

TEST_F(CalculatorTest, SubtractionTest) {
  EXPECT_EQ(calc->subtract(5, 3), 2);
  EXPECT_EQ(calc->subtract(0, 5), -5);
  EXPECT_EQ(calc->subtract(10, 10), 0);
}

TEST_F(CalculatorTest, MultiplicationTest) {
  EXPECT_EQ(calc->multiply(4, 5), 20);
  EXPECT_EQ(calc->multiply(-2, 3), -6);
  EXPECT_EQ(calc->multiply(0, 100), 0);
}

TEST_F(CalculatorTest, DivisionTest) {
  EXPECT_DOUBLE_EQ(calc->divide(10, 2), 5.0);
  EXPECT_DOUBLE_EQ(calc->divide(7, 2), 3.5);
  EXPECT_DOUBLE_EQ(calc->divide(0, 5), 0.0);
}

TEST_F(CalculatorTest, DivisionByZeroTest) {
  EXPECT_THROW(calc->divide(5, 0), std::invalid_argument);
}

// Example of parameterized test
class CalculatorParameterizedTest
    : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(CalculatorParameterizedTest, AdditionParameterized) {
  Calculator calc;
  auto [a, b, expected] = GetParam();
  EXPECT_EQ(calc.add(a, b), expected);
}

INSTANTIATE_TEST_SUITE_P(AdditionTests, CalculatorParameterizedTest,
                         ::testing::Values(std::make_tuple(1, 2, 3),
                                           std::make_tuple(0, 0, 0),
                                           std::make_tuple(-1, -1, -2),
                                           std::make_tuple(100, -50, 50)));