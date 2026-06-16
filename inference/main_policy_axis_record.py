import sys
import time
sys.path.append('/home/nvidia/simulators/fr_grasp/frcobot_python_sdk/linux/lib_x86_64-linux-gnu')
import frrpc

from scooping.object_grounding import Ball
from policy.infer import diffusion_policy_predictor
from scooping.camera_rec import Camera
import rospy

import signal
def signal_handler(signum, frame):
    raise KeyboardInterrupt

if __name__=='__main__':
    signal.signal(signal.SIGINT, signal_handler) 


    robot = frrpc.RPC("192.168.1.102")
    ret = robot.GetSDKVersion() 
    if ret[0] == 0:
        print("SDK version is:", ret[1])
    else:
        print("the errcode is: ", ret[0])

    start_time = time.strftime("%Y%m%d_%H%M%S")
    ball = Ball()    
    cam = Camera(start_time)

    while cam.get_rgb_obs() is None or cam.get_dep_obs() is None:
        rospy.sleep(0.1)
    
    cam.init_video_writers()

    try:
        predictor = diffusion_policy_predictor()
        predictor.collect_data_scoop_infer(ball, cam, robot,start_time)

    finally:
        cam.stop_recording()


