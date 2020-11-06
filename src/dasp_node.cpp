#include <cuda_dasp/dasp_node.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>


namespace cuda_dasp
{

DASPNode::DASPNode()
    : privateNh("~"),
      it(nh),
      rgbSub(it, "/image_color", 1),
      depthSub(it, "/image_depth", 1),
      sync(RGBDSyncPolicy(10), rgbSub, depthSub),
      camInfoReceived(false)
{
    camInfoSub = nh.subscribe("/camera_info", 1, &DASPNode::camInfoCallback, this);
    sync.registerCallback(boost::bind(&DASPNode::RGBDCallback, this, _1, _2));
}

void DASPNode::RGBDCallback(const sensor_msgs::ImageConstPtr& msg_rgb, const sensor_msgs::ImageConstPtr& msg_depth)
{
    if(camInfoReceived)
    {
        cv_bridge::CvImageConstPtr cv_ptr_rgb = cv_bridge::toCvShare(msg_rgb);
        cv_bridge::CvImageConstPtr cv_ptr_depth = cv_bridge::toCvShare(msg_depth);

        cv::Mat rgb = cv_ptr_rgb->image.clone();
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, CV_RGB2BGR);
        cv::Mat depth_16U = cv_ptr_depth->image.clone();
        cv::Mat depth;
        depth_16U.convertTo(depth, CV_32FC1, depthScale);

        dasp->computeSuperpixels(rgb, depth);

        cv::imshow("rgb", bgr);
        cv::imshow("depth", depth_16U * 10);
        cv::waitKey(1);
    }
}

void DASPNode::camInfoCallback(const sensor_msgs::CameraInfo& msg_cam_info)
{
    CamParam cam;
    cam.fx = float(msg_cam_info.K[0]);
    cam.cx = float(msg_cam_info.K[2]);
    cam.fy = float(msg_cam_info.K[4]);
    cam.cy = float(msg_cam_info.K[5]);
    cam.height = msg_cam_info.height;
    cam.width = msg_cam_info.width;

    privateNh.param("depth_scale", depthScale, 0.001f);

    //dasp = new DASP(cam, 0.03f, 0.4f, 0.4f, 10, 3.0f, 700);
    dasp = new DASP(cam, 0.04f, 0.4f, 0.4f, 10, 3.0f, 0);

    camInfoSub.shutdown();

    camInfoReceived = true;

    std::cout<<"Camera info received"<<std::endl;
}

void DASPNode::run()
{
    ros::spin();
}

} // cuda_dasp
