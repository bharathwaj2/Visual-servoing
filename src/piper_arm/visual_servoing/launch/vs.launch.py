#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Get the directory where your Python files are located
    script_dir = '/home/bharath/visual_servoing/src/piper_arm/visual_servoing/src' 

    i = 0
    
    if i==0:
        script = 'PBVS'
    elif i==1:
        script = 'IBVS'
    elif i==2:
        script = 'HEC'
    elif i==3:
        script = 'IBVS_RL'


    return LaunchDescription([
        
        ExecuteProcess(
            cmd=['python3', os.path.join(script_dir, f'{script}.py')],
            output='screen',
            # parameters=[{'use_sim_time': True}],
            shell=False,
            cwd=script_dir
        ),
    ])