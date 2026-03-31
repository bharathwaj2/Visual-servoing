#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription
import os

def generate_launch_description():

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(FindPackageShare('piper_arm_bringup').find('piper_arm_bringup'), 'launch', 'gazebo.launch.py')
        )
    )

    servo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(FindPackageShare('piper_arm_moveit_config').find('piper_arm_moveit_config'), 'launch', 'servo.launch.py')
        )
    )


    script_dir = '/home/bharath/visual_servoing/src/piper_arm/ibvs/src'

    return LaunchDescription([
        
        gazebo_launch,
        servo_launch,
        ExecuteProcess(
            cmd=['ros2', 'service', 'call', '/servo_node/start_servo', 'std_srvs/srv/Trigger', '{}'],
            output='screen'
        ),
        TimerAction(
            period=1.5,
            actions=[
                ExecuteProcess(
                    cmd=['python3', os.path.join(script_dir, 'set_initial_pose_IBVS.py')],
                    output='screen',
                    shell=False,
                    cwd=script_dir
                ),
            ]
        ),
    ])