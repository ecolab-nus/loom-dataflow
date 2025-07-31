#include "scalein/scalein.h"
#include "scaleout/scaleout.h"
#include <iostream>

int main() {
  std::cout << "TMD Application" << std::endl;
  std::cout << "Demonstrating scale-in and scale-out libraries:" << std::endl;

  // Create and use scale-out functionality
  scaleout::ScaleOut scaleout_service;
  scaleout_service.scale();

  // Create and use scale-in functionality
  scalein::ScaleIn scalein_service;
  scalein_service.scale();

  std::cout << "Application completed successfully!" << std::endl;

  return 0;
}