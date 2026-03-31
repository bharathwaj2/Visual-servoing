#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription
import os

def generate_launch_description():

    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(FindPackageShare('piper_arm_bringup').find('piper_arm_bringup'), 'launch', 'rviz.launch.py')
        )
    )

    servo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(FindPackageShare('piper_arm_moveit_config').find('piper_arm_moveit_config'), 'launch', 'servo.launch.py')
        )
    )

    script_dir = '/home/bharath/visual_servoing/src/piper_arm/visual_servoing/src'

    return LaunchDescription([
        
        rviz_launch,
        servo_launch,
        ExecuteProcess(
            cmd=['ros2', 'service', 'call', '/servo_node/start_servo', 'std_srvs/srv/Trigger', '{}'],
            output='screen'
        ),
        TimerAction(
            period=2.5,
            actions=[
                ExecuteProcess(
                    cmd=['python3', os.path.join(script_dir, 'set_initial_pose_visual_servoing.py')],
                    output='screen',
                    shell=False,
                    cwd=script_dir
                ),
            ]
        ),
    ])