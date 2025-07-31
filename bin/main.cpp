#include "scalein/scalein.h"
#include "scaleout/scaleout.h"
#include <iostream>

int main() {
  std::cout << "TMD Application" << std::endl;
  std::cout << "Demonstrating scale-in and scale-out libraries:" << std::endl;
  std::cout << std::endl;

  // Create and use scale-out functionality
  scaleout::ScaleOut scaleout_service;
  scaleout_service.scale();

  // Create and use scale-in functionality
  scalein::ScaleIn scalein_service;
  scalein_service.scale();

  std::cout << std::endl;
  std::cout << "Application completed successfully!" << std::endl;
  std::cout << std::endl;
  std::cout << "Note: For resource management demo, run: ./tmd_resource_demo"
            << std::endl;

  return 0;
}