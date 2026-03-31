#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import RegisterEventHandler
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command
from launch.substitutions import FindExecutable
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    x_pos = 0.4
    y_pos = 0.15
    pitch = 0.0

    pos = {'x': LaunchConfiguration('x_pose', default=x_pos),
           'y': LaunchConfiguration('y_pose', default=y_pos)}
    
    
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'start_rviz',
            default_value='true',
            description='Whether execute rviz2'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'prefix',
            default_value='""',
            description='Prefix of the joint and link names'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'use_sim',
            default_value='true',
            description='Start robot in Gazebo simulation.'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'use_fake_hardware',
            default_value='true',
            description='Start robot with fake hardware mirroring command to its states.'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'fake_sensor_commands',
            default_value='false',
            description='Enable fake command interfaces for sensors used for simple simulations. \
            Used only if "use_fake_hardware" parameter is true.'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'port_name',
            default_value='/dev/ttyUSB0',
            description='The port name to connect to hardware.'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'x',
            default_value=pos['x'],
            description='Cube position in x-axis.'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'y',
            default_value=pos['y'],
            description='Cube position in y-axis.'
        )
    )

    start_rviz = LaunchConfiguration('start_rviz')
    prefix = LaunchConfiguration('prefix')
    use_sim = LaunchConfiguration('use_sim')

    urdf_file = Command(
        [
            PathJoinSubstitution([FindExecutable(name='xacro')]),
            ' ',
            PathJoinSubstitution(
                [
                    FindPackageShare('piper_arm_description'),
                    'urdf',
                    'piper_arm_sim.xacro'
                ]
            ),
            ' ',
            'prefix:=',
            prefix,
            ' ',
            'use_sim:=',
            use_sim,
        ]
    )

    cube_urdf_file = Command(
        [
            PathJoinSubstitution([FindExecutable(name='xacro')]),
            ' ',
            PathJoinSubstitution(
                [
                    FindPackageShare('piper_arm_description'),
                    'urdf',
                    'cube_marker.xacro'
                ]
            ),
            ' ',
            'prefix:=',
            prefix,
            ' ',
            'use_sim:=',
            use_sim,
        ]
    )

    controller_manager_config = PathJoinSubstitution(
        [
            FindPackageShare('piper_arm_bringup'),
            'config',
            'gazebo_controller_manager.yaml',
        ]
    )

    rviz_config_file = PathJoinSubstitution(
        [
            FindPackageShare('piper_arm_bringup'),
            'rviz',
            'piper_arm.rviz'
        ]
    )

    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            {'robot_description': urdf_file},
            controller_manager_config
        ],
        output='both',
        condition=UnlessCondition(use_sim))

    robot_state_pub_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': urdf_file, 'use_sim_time': use_sim}],
        output='screen'
    )

    cube_state_pub_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='cube_state_publisher',
        parameters=[{
            'robot_description': cube_urdf_file,
            # 'frame_prefix': 'cube_' 
        }],
        remappings=[
            ('/robot_description', '/cube_description')
        ]
    )

    cube_static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_cube',
        arguments=[str(x_pos), str(y_pos), '0', '0', str(pitch), '0', 'world', 'cube_link']
    )

    cube_spawn_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-file', '/home/bharath/visual_servoing/src/piper_arm/piper_arm_description/models/cube_with_marker/cube_with_marker.sdf',
                '-entity', 'cube_with_marker',
                '-x', str(x_pos), '-y', str(y_pos), '-z', '0.0',
                '-R', '0.0','-P', str(pitch),'-Y', '0.0'],
        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen',
        condition=IfCondition(start_rviz)
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller'],
        output='screen',
    )

    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller'],
        output='screen',
    )

    delay_rviz_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[rviz_node],
        )
    )

    delay_arm_controller_spawner_after_joint_state_broadcaster_spawner = \
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[arm_controller_spawner],
            )
        )

    delay_gripper_controller_spawner_after_joint_state_broadcaster_spawner = \
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[gripper_controller_spawner],
            )
        )
    
    
    nodes = [
        control_node,
        robot_state_pub_node,
        cube_state_pub_node,
        cube_spawn_node,
        cube_static_tf,
        joint_state_broadcaster_spawner,
        delay_rviz_after_joint_state_broadcaster_spawner,
        delay_arm_controller_spawner_after_joint_state_broadcaster_spawner,
        delay_gripper_controller_spawner_after_joint_state_broadcaster_spawner,
    ]

    return LaunchDescription(declared_arguments + nodes)
