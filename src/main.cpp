#include <ros/ros.h>
#include <cuda_dasp/dasp_node.hpp>

using namespace cuda_dasp;


int main(int argc, char** argv)
{
  ros::init(argc, argv, "dasp_node");

  DASPNode node;

  node.run();

  return 0;
}
