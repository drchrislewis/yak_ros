#include <ros/ros.h>

#include <yak/yak_server.h>
#include <yak/mc/marching_cubes.h>

#include <yak_msgs/ResetParams.h>

#include <gl_depth_sim/interfaces/opencv_interface.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/ply_io.h>

#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <cv_bridge/cv_bridge.h>

#include <std_srvs/Trigger.h>
#include <std_srvs/SetBool.h>
#include <queue>

static const std::double_t DEFAULT_MINIMUM_TRANSLATION = 0.00001;

/**
 * @brief The OnlineFusionServer class. Integrate depth images into a TSDF volume. When requested, mesh the volume using marching cubes.
 * Note that this will work using both simulated and real robots and depth cameras.
 */
class OnlineFusionServer
{
public:
  /**
   * @brief OnlineFusionServer constructor
   * @param nh - ROS node handle
   * @param params - KinFu parameters such as TSDF volume size, resolution, etc.
   * @param fusion_world_to_volume - Transform from world frame to volume origin frame.
   * Note: the fusion server behaves as if the tsdf_frame its world
   * We assume someone moves the tsdf_frame around to place the volume to be scanned relative a world_frame
   * The mesh is returned in the tsdf_frame and needs to be transformed into the world frame
   */
  OnlineFusionServer(ros::NodeHandle &nh, const kfusion::KinFuParams& params, const Eigen::Affine3f& fusion_world_to_volume, const std::string& file_location) :
    params_(params),
    robot_tform_listener_(tf_buffer_),
    fusion_(params, fusion_world_to_volume),
    tsdf_to_camera_prev_(Eigen::Affine3d::Identity()),
    file_location_(file_location)
  {
    // get parameters for this node
    nh.param<std::string>("depth_topic", depth_topic_, "camera/depth/image_raw"); // always use the raw image
    nh.param<std::string>("tsdf_frame_name", tsdf_frame_name_, "yak_frame"); // here we assume someone moves the tsdf frame from place to place

    // Subscribe to depth images published on the topic named by the depth_topic param. Set up callback to integrate images when received.
    point_cloud_sub_ = nh.subscribe(depth_topic_, 1, &OnlineFusionServer::onReceivedDepthImg, this);

    // Advertise service for marching cubes meshing
    generate_mesh_service_ = nh.advertiseService("generate_mesh_service", &OnlineFusionServer::onGenerateMesh, this);

    // Advertise service for clearing the contents of the voxel volume
    clear_volume_service_ = nh.advertiseService("clear_volume_service", &OnlineFusionServer::onClearVolume, this);

    // Advertise service to re-initialize the contents of the voxel volume with new params
    reset_params_service_ = nh.advertiseService("reset_params_service", &OnlineFusionServer::onResetParams, this);

}

private:
  /**
   * @brief onReceivedDepthImg - callback for integrating new depth images into the TSDF volume
   * @param image_in - pointer to new depth image
   */
  void onReceivedDepthImg(const sensor_msgs::ImageConstPtr& image_in)
  {
    image_q_.push(image_in);
    if(image_q_.size() > 2){
      auto next_image = image_q_.front();
      image_q_.pop(); // remove it now

      // Get the camera pose in the tsdf frame at the time when the depth image was generated.
      ROS_INFO_STREAM("Got depth image");
      geometry_msgs::TransformStamped transform_tsdf_to_camera;
      try
	{
	  transform_tsdf_to_camera = tf_buffer_.lookupTransform(tsdf_frame_name_, next_image->header.frame_id, next_image->header.stamp);
	}
      catch(tf2::TransformException &ex)
	{
	  // Abort integration if tf lookup failed
	  ROS_WARN("%s", ex.what());
	  return;
	}
      Eigen::Affine3d tsdf_to_camera = tf2::transformToEigen(transform_tsdf_to_camera);
      
      // Find how much the camera moved since the last depth image. If the magnitude of motion was below some threshold, abort integration.
      // This is to prevent noise from accumulating in the isosurface due to numerous observations from the same pose.
      std::double_t motion_mag = (tsdf_to_camera.inverse() *tsdf_to_camera_prev_).translation().norm();
      ROS_INFO_STREAM(motion_mag);
      if(motion_mag < DEFAULT_MINIMUM_TRANSLATION)
	{
	  ROS_INFO_STREAM("Camera motion below threshold");
	  return;
	}
      
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(next_image, sensor_msgs::image_encodings::TYPE_16UC1);

#if (0)// debug the transform by printing it out every 30'th update
      static int q=0;
      if(q==0){
	ROS_ERROR("f1 %6.3lf %6.3lf %6.3lf %6.3lf",
		  tsdf_to_camera.linear().col(0).x(), tsdf_to_camera.linear().col(1).x(),  tsdf_to_camera.linear().col(2).x(), tsdf_to_camera.translation().x());
	ROS_ERROR("f2 %6.3lf %6.3lf %6.3lf %6.3lf",
		  tsdf_to_camera.linear().col(0).y(), tsdf_to_camera.linear().col(1).y(),  tsdf_to_camera.linear().col(2).y(),  tsdf_to_camera.translation().y());
	ROS_ERROR("f3 %6.3lf %6.3lf %6.3lf %6.3lf\n",
		  tsdf_to_camera.linear().col(0).z(), tsdf_to_camera.linear().col(1).z(),  tsdf_to_camera.linear().col(2).z(),  tsdf_to_camera.translation().z());
      }
      q=(q+1)%30;
#endif      

      // Integrate the depth image into the TSDF volume
      if (!fusion_.fuse(cv_ptr->image, tsdf_to_camera.cast<float>()))
	ROS_WARN_STREAM("Failed to fuse image");
      
      // If integration was successful, update the previous camera pose with the new camera pose
      tsdf_to_camera_prev_ = tsdf_to_camera;
      
    }
    return;
  }
    
    /**
   * @brief onGenerateMesh - Perform marching cubes meshing on the TSDF volume and save the result as a binary .ply file.
   * @param req
   * @param res
   * @return
   */
  bool onGenerateMesh(std_srvs::TriggerRequest &req, std_srvs::TriggerResponse &res)
  {
    yak::MarchingCubesParameters mc_params;
    mc_params.scale = params_.volume_resolution;
    pcl::PolygonMesh mesh = yak::marchingCubesCPU(fusion_.downloadTSDF(), mc_params);

    // save the mash to a file
    ROS_INFO_STREAM("Meshing done, saving ply");
    pcl::io::savePLYFileBinary(file_location_, mesh);
    ROS_INFO_STREAM("Saving done");
    res.success = true;
    return true;
  }

  bool onClearVolume(std_srvs::TriggerRequest &req, std_srvs::TriggerResponse &res)
  {
    ROS_INFO_STREAM("Clearing volume");
    res.success = fusion_.reset();;
    return true;
  }

  bool onResetParams(yak_msgs::ResetParamsRequest &req, yak_msgs::ResetParamsResponse &res)
  {
    // TODO remove the next three lines
    res.success = true;
    return true;// ignoring change in tsdf volume
    //***********************************

    ROS_INFO_STREAM("Resetting volume with new params");

    // Set up new params starting from the current params
    auto new_params = params_;

    // Set up new volume paramaters
    new_params.volume_dims = cv::Vec3i(req.size_x, req.size_y, req.size_z);
    new_params.volume_resolution = req.voxel_size;

    res.success = fusion_.resetWithNewParams(new_params);

    return true;
  }

  const kfusion::KinFuParams params_;
  ros::Subscriber point_cloud_sub_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener robot_tform_listener_;
  ros::ServiceServer generate_mesh_service_;
  ros::ServiceServer clear_volume_service_;
  ros::ServiceServer reset_params_service_;
  ros::ServiceServer set_camera_service_;
  yak::FusionServer fusion_;
  Eigen::Affine3d tsdf_to_camera_prev_;
  std::string file_location_;
  std::string tsdf_frame_name_;
  std::string depth_topic_;
  std::queue<sensor_msgs::ImageConstPtr> image_q_;

};

/**
 * @brief main - Initialize the tsdf_node
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
  ros::init(argc, argv, "tsdf_node");
  ros::NodeHandle nh; // removed "tsdf_node" from node handle to keep service names cleaner

  std::string file_location;
  nh.param<std::string>("output_file", file_location, "cubes.ply");

  kfusion::KinFuParams default_params = kfusion::KinFuParams::default_params();
  default_params.use_pose_hints = true; // use robot forward kinematics to find camera pose relative to TSDF volume
  default_params.use_icp = false; // since we're using robot FK to get the camera pose, don't use ICP (TODO: yet!)
  default_params.update_via_sensor_motion = false;

  // Get camera intrinsics from params
  XmlRpc::XmlRpcValue camera_matrix;
  nh.getParam("camera/camera_matrix/data", camera_matrix);
  default_params.intr.fx = static_cast<double>(camera_matrix[0]);
  default_params.intr.fy = static_cast<double>(camera_matrix[4]);
  default_params.intr.cx = static_cast<double>(camera_matrix[2]);
  default_params.intr.cy = static_cast<double>(camera_matrix[5]);

  ROS_INFO("Camera Intr Params: %f %f %f %f\n", default_params.intr.fx, default_params.intr.fy, default_params.intr.cx, default_params.intr.cy);

  // since we move the yak_frame with the mutable_transform_publisher, world_to_volume is always Identity, only the box size can change
  // The resulting mesh from from the scan server is in the yak_frame, not the world
  Eigen::Affine3f fusion_world_to_volume (Eigen::Affine3f::Identity());

  // Set up TSDF parameters
  // TODO: Autocompute resolution from volume length/width/height in meters
  default_params.volume_dims = cv::Vec3i(32*32, 32*32, 32*32);
  default_params.volume_resolution = 0.005;
  default_params.volume_pose = Eigen::Affine3f::Identity(); // This gets overwritten when Yak is initialized
  default_params.tsdf_trunc_dist = default_params.volume_resolution * 5.0f; //meters;
  default_params.tsdf_max_weight = 50;   //frames
  default_params.raycast_step_factor = 0.25;  //in voxel sizes
  default_params.gradient_delta_factor = 0.25; //in voxel sizes

  // Set up the fusion server with the above parameters;
  OnlineFusionServer ofs(nh, default_params, fusion_world_to_volume, file_location);

  // Do TSDF-type things for the lifetime of the node
  ros::spin();
  return 0;
}
