import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import torch
import math
from omnigibson.object_states import *

def align_bowl_to_direction(delta):
    forward = delta / np.linalg.norm(delta)
    up = np.array([0, 0, 1])
    right = np.cross(forward,up)
    up_new = np.cross(forward,right)
    # print(up_new / np.linalg.norm(up_new))
    return up_new / np.linalg.norm(up_new)


def find_perpendicular_vector(a, b):
    dot_product = np.dot(a, b)
    norm_a_squared = np.dot(a, a)
    projection = dot_product / norm_a_squared * a
    c = b - projection
    c_normalized = c / np.linalg.norm(c)
 
    return c_normalized

import numpy as np
from scipy.spatial.transform import Rotation as R

def direction_to_quaternion_sim(direction):
    # Normalize direction vector
    direction = direction / np.linalg.norm(direction)
    
    # Reference direction (z-axis)
    ref_direction = np.array([0, 0, 1])
    
    # Calculate rotation axis (cross product)
    axis = np.cross(ref_direction, direction)
    axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) != 0 else np.array([0, 0, 1])
    
    # Calculate rotation angle (dot product and arccos)
    theta = np.arccos(np.dot(ref_direction, direction))
    
    # Calculate quaternion
    q_w = np.cos(theta / 2)
    q_xyz = axis * np.sin(theta / 2)
    
    # Return quaternion as [w, x, y, z]
    # return np.array([q_w, q_xyz[0], q_xyz[1], q_xyz[2]])
    return np.array([q_xyz[0], q_xyz[1], q_xyz[2], q_w])


def random_value():
    if np.random.rand() < 0.5:
        return np.random.uniform(-0.03, -0.01)  
    else:
        return np.random.uniform(0.01, 0.03)   
    
import random
import math


def generate_variables(r, row):
    i=0
    while True:
        i+=1
        z_relat = random.uniform(0.01, 0.2) 
        h = random.uniform(row/math.sqrt(100) + z_relat, row*math.sqrt(100) + z_relat)
        
        l = math.sqrt(row**2 + (h - z_relat)**2)
        d = l - h
        delta_d = l - h - r

        if l > h + r and h > z_relat and delta_d > r and delta_d < 2*r:
            return row, h, z_relat, l, d, delta_d
        if i>1000000:
            row = random.uniform(0.05, 0.12)
            # return None,None,None,None,None,None


def collect_data_scoop_closeloop(env, ball, ball_r, row, bowl, container, cam, action):
    row, h, z_relat, l, d, delta_d = generate_variables(ball_r, row)
    if row is None:
        return [],[],[],[],[],[], [],[]
    # print(row, z_relat, h)
    variables_data = (row, h, z_relat, l, d, delta_d)
    print(variables_data)
    
    pre_obs = []
    
    for i in range(4):
        env.step(action)
        # print(ball.get_position())
        pre_obs.append(ball.get_position()) 
    
    ball_position = ball.get_position()  # 获取当前球的位置

    x = np.random.normal(0,0.5)*0.02
    y = np.random.normal(0,0.5)*0.02
    z = np.random.normal(0,0.5)*0.02



    init_bowl_position = np.array([ball_position[0]+row+x, ball_position[1]+y, ball_position[2]+z_relat+z])
    bowl2ball_direction = ball_position - init_bowl_position

    bd2 = ball_position[2] - init_bowl_position[2]+h
    bowl2ball_direction[2] = bd2
    bowl_orientation = direction_to_quaternion_sim(bowl2ball_direction) ##### different from real

    bowl.set_position_orientation(
        position=init_bowl_position,
        orientation=bowl_orientation
    )
    bowl.keep_still()

    step_size = 0.01
    
    last_ball_position =  ball_position

    new_bowl_position = bowl.get_position()
    env.step(action)
    # env.step(action)

    image_seq = [] 
    pos_seq = []
    ball_pos_seq = []
    ori_seq = []
    action_seq = []
    ballmotion_seq = []

    i=0
    ball_movement = 0
    backwards_count = 0
    
    
    colli_count = 0

    for i in range(60):
        bowl.keep_still()
        container.keep_still()

        obs, _ = cam.get_obs()
        frame = obs["rgb"][...,:3]

        curr_ball_pos = ball.get_position()
        ball_movement = curr_ball_pos - last_ball_position
        last_ball_position = curr_ball_pos

        bd2 = curr_ball_pos[2] - bowl.get_position()[2]+h
        bowl2ball_direction = curr_ball_pos - bowl.get_position()
        bowl2ball_direction[2] = bd2

        
        random_array_3 = np.random.randn(3) * 0.01
        random_array_4 = np.random.randn(4) * 0.02
        bowl_orientation = direction_to_quaternion_sim(bowl2ball_direction) + random_array_4
        bowl_moving_direction = align_bowl_to_direction(bowl2ball_direction) + random_array_3
        bowl_orientation = bowl_orientation / np.linalg.norm(bowl_orientation)
        bowl_moving_direction = bowl_moving_direction / np.linalg.norm(bowl_moving_direction)

        frame = cv2.resize(frame, (320,180))

        image_seq.append(frame)

        last_bowl_pos = bowl.get_position()
        bowl_motion = ball_movement + bowl_moving_direction * step_size
        new_bowl_position = last_bowl_pos +  bowl_motion
        pos_seq.append(last_bowl_pos)
        ball_pos_seq.append(ball.get_position())
        action_seq.append(bowl_moving_direction)
        ori_seq.append(bowl_orientation)

        ballmotion_seq.append(ball_movement)
        
        bowl.set_position_orientation(
            position=new_bowl_position,
            orientation=bowl_orientation
        )

        env.step(action)

        test_bowl_pos = bowl.get_position()

        condition1 = np.sqrt(np.sum((test_bowl_pos[:3]-last_bowl_pos[:3])**2)) < 0.5*np.sqrt(np.sum(bowl_motion[:3]**2))
        condition2 = abs(new_bowl_position[0]-curr_ball_pos[0])<0.01 and abs(new_bowl_position[1]-curr_ball_pos[1])<0.01
        
        if condition1 and condition2:
            backwards_count+=1
            bowl.set_position_orientation(
                position=bowl.get_position() - bowl_motion,
                orientation=bowl_orientation
            )

            pos_seq.append(test_bowl_pos)
            ball_pos_seq.append(ball.get_position())
            action_seq.append(-bowl_moving_direction)
            ori_seq.append(bowl_orientation)

            ballmotion_seq.append(ball.get_position() - last_ball_position)

            last_ball_position = curr_ball_pos
            
            env.step(action)
            print("back----------------{}".format(i))
            break
        
        if condition2:
            break

        if bowl.states[Touching].get_value(ball):
            colli_count+=1

        if colli_count > 3:
            return [],[],[],[],[],[], [],[]




    for j in range(20+59-i-backwards_count):
        # print(i)
        bowl.keep_still()
        container.keep_still()
        # ball.keep_still()
        obs, _ = cam.get_obs()
        frame = obs["rgb"][...,:3]

        curr_ball_pos = ball.get_position()
        ball_movement = curr_ball_pos - last_ball_position
        last_ball_position = curr_ball_pos

        bowl2ball_direction = curr_ball_pos - bowl.get_position()
        # bd2 = curr_ball_pos[2] + h
        bd2 = curr_ball_pos[2] - bowl.get_position()[2]+h
        bowl2ball_direction[2] = bd2
        
        
        # bowl_orientation = direction_to_quaternion_sim(bowl2ball_direction)
        bowl_orientation = direction_to_quaternion_sim(np.array([0,0,1]))
        bowl_moving_direction = align_bowl_to_direction(bowl2ball_direction)
        frame = cv2.resize(frame, (320,180))
        
        image_seq.append(frame)

        new_bowl_position = bowl.get_position()  + (bowl_moving_direction *step_size + np.array([0, 0, step_size])) * (0.4+0.6*j/(20+59-i-backwards_count)) ####
        pos_seq.append(bowl.get_position())
        ball_pos_seq.append(ball.get_position())
        bowl_moving_direction = bowl_moving_direction + np.array([0,0,(0.4+0.6*j/(20+59-i-backwards_count))])
        up_direction = bowl_moving_direction / np.linalg.norm(bowl_moving_direction)
        action_seq.append(up_direction)
        ori_seq.append(bowl_orientation)
        
        # 更新勺子的位置
        bowl.set_position_orientation(
            position=new_bowl_position,
            orientation=bowl_orientation
        )

        env.step(action)
        if curr_ball_pos[2]-bowl.get_position()[2] > 0.:
            pass
        else:
            return [],[],[],[],[],[],[],[]
    # print(ball.get_position())
    final_z = ball.get_position()[2]
    if final_z-bowl.get_position()[2] > 0.:
        return image_seq,pos_seq,ball_pos_seq, ori_seq, action_seq,ballmotion_seq, variables_data, pre_obs
    else:
        return [],[],[],[],[],[], [],[]
