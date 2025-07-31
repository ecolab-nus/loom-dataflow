#include "scalein.h"
#include <gtest/gtest.h>

class ScaleInTest : public ::testing::Test {
protected:
  void SetUp() override {
    scalein_instance = std::make_unique<scalein::ScaleIn>();
  }

  void TearDown() override { scalein_instance.reset(); }

  std::unique_ptr<scalein::ScaleIn> scalein_instance;
};

TEST_F(ScaleInTest, ConstructorTest) { EXPECT_NE(scalein_instance, nullptr); }

TEST_F(ScaleInTest, ScaleTest) {
  // This test just verifies the scale method can be called
  // Add more specific tests based on your scale-in logic
  EXPECT_NO_THROW(scalein_instance->scale());
}