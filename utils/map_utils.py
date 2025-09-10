import math
import os
import random
import subprocess
import time
from os import path
import actionlib  
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion,Twist
from nav_msgs.msg import Odometry,OccupancyGrid
from sensor_msgs.msg import PointCloud2,LaserScan
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal  
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import tf
# 判断点是否在三角形内部的函数
def point_in_triangle(point, vertex_a, vertex_b, vertex_c):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(point, vertex_a, vertex_b)
    d2 = sign(point, vertex_b, vertex_c)
    d3 = sign(point, vertex_c, vertex_a)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)

#旋转地图函数,k代表顺时针旋转多少个90度
def costmap_rot(map_data,map_width,map_height,k):
    
    costmap_array = np.array(map_data).reshape(map_height, map_width)

    # 顺时针旋转90度
    rotated_costmap = np.rot90(costmap_array, k)

    # 将旋转后的二维数组转回一维数组
    rotated_costmap_data = rotated_costmap.flatten()
    return rotated_costmap_data

def costmap_cut(map_data,map_width, map_height,triangle_base_width,triangle_height):
    #前侧(绿轴正向)
    vertex_a = (0, 0)
    vertex_b = (-triangle_base_width / 2, triangle_height)
    vertex_c = (triangle_base_width / 2, triangle_height)
    triangle_values = []
    height_list=range(int(map_height/2),map_height)
    width_list=range(map_width)
    # 遍历栅格地图
    for y in height_list:
        for x in width_list:
            # 计算栅格单元格的中心坐标
            cell_center = (x - map_width / 2 + 0.5, y - map_height / 2 + 0.5)
            
            # 检查是否在等腰三角形内部
            if point_in_triangle(cell_center, vertex_a, vertex_b, vertex_c):
                index = y * map_width + x
                triangle_values.append(map_data[index])
            else:
                triangle_values.append(100)
    # print(len(triangle_values))
    return np.array(triangle_values)

def quaternion2yaw(quaternion):
    # 使用tf库将四元数转换为欧拉角（RPY）
    euler = tf.transformations.euler_from_quaternion(quaternion)

    # 从欧拉角中提取偏航角
    yaw = euler[2]
    return yaw

def goal2option(x,y):
    if y >= abs(x):
        return 0
    elif abs(y) <= (-x) :
        return 1
    elif y <= (-abs(x)) :
        return 2
    elif abs(y) <= x :
        return 3
    
def rotate_point( x , y, angle_degrees):
# 将角度转换为弧度
    angle_radians = math.radians(angle_degrees)
    # 计算旋转后的坐标
    x_rotated = x * math.cos(angle_radians) + y * math.sin(angle_radians)
    y_rotated = -x * math.sin(angle_radians) + y * math.cos(angle_radians)
    
    return x_rotated,y_rotated

def rotate_yaw( yaw, angle_degrees):
# 将角度转换为弧度
    angle_radians = math.radians(angle_degrees)
        # 更新yaw角
    new_yaw = yaw + angle_radians
    
    # 保持yaw角在 -π 到 π 的范围内
    while new_yaw > math.pi:
        new_yaw -= 2 * math.pi
    while new_yaw < -math.pi:
        new_yaw += 2 * math.pi
    
    return new_yaw

def process_option(option):
    
    result=np.zeros(4)
    result[option]=1


    return result

def find_neighbors(map_width, map_height , map_data ,x, y, threshold, distance):
    flag =False
    for i in range(-distance, distance+1):
        for j in range(-distance, distance+1):
            # 添加条件，只遍历距离为3的格子
            if abs(i) == distance or abs(j) == distance:
                if 0 <= x + i < map_width and 0 <= y + j < map_height:
                    if map_data[x + i + (y + j) * map_width] <= threshold:
                        x = x+i
                        y = y+j
                        return x,y
    return x,y

def is_line_cost_above_threshold(cost_map_data, costmap_width, start_point, end_point, threshold):
    def get_line_points(start, end):
        x1, y1 = start
        x2, y2 = end
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = -1 if x1 > x2 else 1
        sy = -1 if y1 > y2 else 1
        err = dx - dy
        index = y1 * costmap_width + x1
        while True:
            points.append(index)

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err

            if e2 > -dy:
                err -= dy
                x1 += sx

            if e2 < dx:
                err += dx
                y1 += sy
            index = y1 * costmap_width + x1
        return points

    # 获取两点之间的栅格线
    line_points = get_line_points(start_point, end_point)

    # 检查每个栅格点的值是否超过阈值
    for idx in line_points:
        if cost_map_data[idx] > threshold:
            return True

    # 如果没有超过阈值的点，返回False
    return False

def get_obstacle_info():
    try:
        # 使用rospy.wait_for_message等待单个消息
        msg = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=5.0)

        obstacles = []
        # 提取障碍物的位置信息
        for i in range(len(msg.name)):
            model_name = msg.name[i]
            model_pose = msg.pose[i]

            # 在这里添加逻辑以确定哪些模型是障碍物
            # if "obstacle" in model_name:
            obstacle_info = {'name': model_name, 'x': model_pose.position.x, 'y': model_pose.position.y}
            print(obstacle_info)   
                # obstacles.append(obstacle_info)

        return obstacles

    except rospy.ROSException as e:
        rospy.logerr("Failed to receive model_states message: %s", str(e))
        return None
    
def generate_discrete_points(discrete_row, discrete_column, max_width, max_height, min_width, min_height):
    # 计算行和列的步长
    row_step = max_height / (discrete_row - 1)
    col_step = max_width / (discrete_column - 1)

    # 生成二维点的坐标
    points = []
    for i in range(discrete_row):
        for j in range(discrete_column):
            x = j * col_step - max_width / 2.0
            y = i * row_step - max_height / 2.0
            points.append((x, y))

    # 将结果转换为 NumPy 数组
    points_array = np.array(points)

    return points_array

def filter_points(points, d, padding):
    filtered_points = []

    for i in range(len(points)):
        include_point = True
        if (abs(points[i][0]) < padding and abs(points[i][1]) < padding):
            include_point = False
        else:
            for j in range(i + 1, len(points)):
                distance = math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
                if distance < d:
                    include_point = False
                    break
        if include_point:
            filtered_points.append(points[i])

    return filtered_points

def points_sampling(vector, target_point, min_distance ,k=1):
    probabilities = []

    for point in vector:
        distance = np.linalg.norm(np.array(point) - np.array(target_point))

        # Adjust the probability based on the distance using an exponential function
        probability = np.exp(-(distance - min_distance) )

        probabilities.append(probability)

    # Normalize probabilities to sum to 1
    probabilities = np.array(probabilities) / np.sum(probabilities)
    # Perform sampling based on computed probabilities
    sampled_index = np.random.choice(len(vector),size = k, replace=False,p=probabilities).astype(int)
    return np.array(vector)[sampled_index]

def calculate_angle(delta_x, delta_y ):


    # 使用atan2计算角度
    angle_radians = math.atan2(delta_y, delta_x)


    return angle_radians

