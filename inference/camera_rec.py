#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import message_filters
import time
import os

class Camera:
    def __init__(self, start_time):
        self.color_image = None
        self.depth_image = None
        self.bridge = CvBridge()
        self.color_writer = None
        self.depth_writer = None
        # self.save_dir = save_dir
        # self.start_time = time.strftime("%Y%m%d_%H%M%S")

        rospy.init_node('realsense_recorder', anonymous=True, log_level=rospy.DEBUG)

        # Subscribers
        color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)

        # Synchronizer
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.image_callback)

        # Video properties
        self.fps = 30
        self.color_video_path = os.path.join("/home/nvidia/hdd/wkn/real/recordings", "rgb",f"rgb_{start_time}.mp4")
        self.depth_video_path = os.path.join("/home/nvidia/hdd/wkn/real/recordings", "dep",f"depth_{start_time}.mp4")

        self.color_frame_size = (640, 480)  # Update based on your camera's resolution
        self.depth_frame_size = (640, 480)  # Same resolution for depth

        self.init_video_writers()

    def init_video_writers(self):
        # Define the codec and create VideoWriter objects
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.color_writer = cv2.VideoWriter(self.color_video_path, fourcc, self.fps, self.color_frame_size)
        self.depth_writer = cv2.VideoWriter(self.depth_video_path, fourcc, self.fps, self.depth_frame_size, isColor=False)

    def image_callback(self, color_msg, depth_msg):
        try:
            # Convert ROS images to OpenCV images
            self.color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

            # Normalize depth image for visualization (convert to 8-bit grayscale)
            depth_vis = cv2.convertScaleAbs(self.depth_image, alpha=0.03)

            # Write frames to video files
            if self.color_writer and self.depth_writer:
                self.color_writer.write(self.color_image)
                self.depth_writer.write(depth_vis)
        except Exception as e:
            rospy.logerr(f"Image processing error: {e}")

    def stop_recording(self):
        # Release video writers
        if self.color_writer:
            self.color_writer.release()
        if self.depth_writer:
            self.depth_writer.release()
        rospy.loginfo(f"Videos saved")
    
    def get_rgb_obs(self):
        return self.color_image

    def get_dep_obs(self):
        return self.depth_image

if __name__ == "__main__":
    save_directory = "/path/to/save/videos"  # Replace with your desired directory
    os.makedirs(save_directory, exist_ok=True)

    recorder = CameraRecorder(save_directory)
    try:
        rospy.loginfo("Recording started. Press Ctrl+C to stop.")
        rospy.spin()  # Keep the node running to record
    finally:
        recorder.stop_recording()


if __name__ == "__main__":
    camera = Camera()
    # Wait for images to be received
    rospy.sleep(2)

    # Main loop to retrieve and process images
    while not rospy.is_shutdown():
        rgb_image = camera.get_rgb_obs()
        image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_image = Image.fromarray(image_rgb)
        pil_image.save("converted_image.jpg")

        depth_image = camera.get_dep_obs()
        # camera_info = camera.get_camera_info()
        # print(camera_info.K)
        if rgb_image is not None and depth_image is not None:
            print("RGB Image Shape:", rgb_image.shape)
            print("Depth Image Shape:", depth_image.shape)
            break
        else:
            rospy.loginfo("Waiting for images...")
            time.sleep(0.5)
