#include "scaleout.h"
#include <gtest/gtest.h>

class ScaleOutTest : public ::testing::Test {
protected:
  void SetUp() override {
    scaleout_instance = std::make_unique<scaleout::ScaleOut>();
  }

  void TearDown() override { scaleout_instance.reset(); }

  std::unique_ptr<scaleout::ScaleOut> scaleout_instance;
};

TEST_F(ScaleOutTest, ConstructorTest) { EXPECT_NE(scaleout_instance, nullptr); }

TEST_F(ScaleOutTest, ScaleTest) {
  // This test just verifies the scale method can be called
  // Add more specific tests based on your scale-out logic
  EXPECT_NO_THROW(scaleout_instance->scale());
}