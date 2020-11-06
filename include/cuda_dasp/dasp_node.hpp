#pragma once

#include <ros/ros.h>
#include <opencv2/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>
#include <message_filters/cache.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <cuda_dasp/dasp.hpp>


namespace cuda_dasp
{

class DASPNode
{

public:
    DASPNode();
    void RGBDCallback(const sensor_msgs::ImageConstPtr& msg_rgb, const sensor_msgs::ImageConstPtr& msg_depth);
    void camInfoCallback(const sensor_msgs::CameraInfo& info);
    void run();

private:
    ros::NodeHandle nh, privateNh;
    float depthScale;
    image_transport::ImageTransport it;
    image_transport::SubscriberFilter rgbSub, depthSub;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> RGBDSyncPolicy;
    message_filters::Synchronizer<RGBDSyncPolicy> sync;
    ros::Subscriber camInfoSub;
    bool camInfoReceived;
    cv::Ptr<DASP> dasp;

};

} // cuda_dasp
