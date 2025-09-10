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
import threading
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion,Twist,PoseStamped
from nav_msgs.msg import Odometry,OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from sensor_msgs.msg import PointCloud2,LaserScan
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal  
from gazebo_msgs.msg import ModelStates
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from actionlib_msgs.msg import GoalStatus,GoalStatusArray
from utils.map_utils import *
from preference_model import HumanPreference
from preferences_buffer import PreferencesBuffer
GOAL_REACHED_DIST = 0.1
COLLISION_DIST = 0.35
TIME_DELTA = 0.1


# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True

    # 左上角非室内区域
    if -9 <= x <= -2 and 2.36 <= y <= 5.32:
        goal_ok = False

    # 右上角非室内区域
    if 4.26 <= x <= 9 and 2.9 <= y <= 5.32:
        goal_ok = False

    # 左上角桌椅垃圾桶
    if -9 < x < -7.77 and 0.46 < y < 2.36:
        goal_ok = False

    # 左上角床
    if -7.7 < x < -4.81 and 0.7 < y < 2.36:
        goal_ok = False

    # 左上角箱子和墙
    if -3.83 < x < -1.92 and 0.2 < y < 2.36:
        goal_ok = False

    # 左侧屏风
    if -8.67 < x < -3.35 and -1.76 < y < -0.9:
        goal_ok = False

    # 左侧椅子和球
    if -8.65 < x < -6.26 and -4.7 < y < -3.4:
        goal_ok = False
        
    # 左下墙
    if -3 < x < -2 and -5.1 < y < -2.22:
        goal_ok = False
                
    # 下侧各种
    if -1.99 < x < 9 and -5.1 < y < -3.78:
        goal_ok = False
        
    # 右侧橱柜
    if 8 < x < 9 and -3.77 < y < 0.05:
        goal_ok = False
        
    # 右侧橱柜
    if 8 < x < 9 and -3.77 < y < 0.05:
        goal_ok = False
                
    # 右侧桌椅
    if 5.21 < x < 7.9 and -0.2 < y < 2.4:
        goal_ok = False
    
    # 上侧健身器材
    if 1.95 < x < 5.22 and 1.8 < y < 4.87:
        goal_ok = False
    
    # 上侧椅子
    if -1.61 < x < 0.53 and 3.6 < y < 4.7:
        goal_ok = False
            
    # 中间沙发等
    if -1.26 < x < 3.05 and -3.41 < y < -0.13:
        goal_ok = False
        
    # 外围边界
    if x > 9 or x < -9 or y > 5.3 or y < -5.1:
        goal_ok = False

    return goal_ok


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, **kwargs):
        self.scan_dim = 36
        self.robot_dim = 3
        self.observation_dim = self.scan_dim + self.robot_dim
        self.action_dim = 2
        self.action_high = np.array(kwargs["max_limits"])
        self.action_low = np.array(kwargs["low_limits"])
        self.manual = kwargs["manual"]
        self.preferences_train = kwargs["preference_training"]
        if self.preferences_train:
            self.preferences_model = kwargs["preferences_model"]
            self.Stage = kwargs["stage"]
            self.neighbor_radius = 0.8  # 不确定性搜索半径
            self.uncertainty_threshold = 0.2  # 不确定性阈值
        self.max_episode_steps = 30
        self.steps = 0
        # self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0
        self.goal_x = 1
        self.goal_y = 0.0
        #局部代价地图尺寸
        self.max_width = 2.8
        self.max_height = 2.8
        self.local_distance = np.linalg.norm(
            [self.max_width/2, self.max_height/2]
        )
        #环境尺寸
        self.env_scale_left = -9.2
        self.env_scale_right = 9.2
        self.env_scale_button = -5.3
        self.env_scale_top = 5.5
        #全局目标距离
        self.global_distance = 0
        # self.velodyne_data = np.ones(self.environment_dim) * 10
        self.scan_data = np.ones(self.scan_dim) * 10
        self.last_odom = None
        self.last_costmap = None
        
        self.achive_flag = False
        self.aborted_flag = False
        self.out_time_flag = False
        
        self.guidance_option = 0
        
        self.in_triangle = None
        
        self.guidance_x = 0
        self.guidance_y = 0
        self.guidance_yaw = 0
       
        self.distance = 0
        self.first = True
        self.done = False
        self.cross = False
        
        self.fond_global_goal = False #用于标志全局点有没有找到
        
        self.set_self_state = ModelState()
        self.set_self_state.model_name = "p3dx"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        rospy.init_node("gym", anonymous=True)

        # Set up the ROS publishers and subscribers
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.marker_pub1 = rospy.Publisher('/marker', MarkerArray, queue_size=3)
        self.marker_pub2 = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        # self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        # self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        # self.velodyne = rospy.Subscriber(
        #     "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        # )
        self.scan = rospy.Subscriber(
            "/scan", LaserScan, self.scan_callback, queue_size=1
        )
        self.odom = rospy.Subscriber(
            "/odom", Odometry, self.odom_callback, queue_size=1
        )
        self.costmap = rospy.Subscriber(
            "/move_base/local_costmap/costmap_updates", OccupancyGridUpdate, self.costmap_callback, queue_size=1
        )
        self.guidance = rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self.guidance_callback, queue_size=1
        )
        
    def seed(self, seed_value):
        random.seed(seed_value)
        
    def get_reward(self,goal_invalid,cross,d_distance,target,theta,steps,distance,action):
        # reward = 1 -int(goal_invalid)*150+ 10*distance +int(target)*100-abs(theta)*3
        # print('reward:',reward)
        # return reward
        # if target:
        #     reward = 100.0
        # elif goal_invalid:
        #     reward = -100.0
        # else:
        #     reward = 0
        # d = np.linalg.norm(
        #     [action[0], action[1]]
        # )
        reward=0
        if d_distance > 0:
            reward = reward + 3.6*d_distance
        else:
            reward = reward + 1.7*d_distance
            
        if cross:
            reward = reward - 15
        # if target :
        #     reward = reward +10 
        # if d < 0.4:
        #     reward = reward -3
        max_steps = math.ceil(self.global_distance / self.local_distance)
        if steps > max_steps:
            reward = reward-0.7*steps
        # reward = reward - 1*abs(theta)
        # print(reward)
        return reward
    
    def reset_flags(self):
        self.achive_flag = False
        self.aborted_flag = False
        self.out_time_flag = False        
        
    def scan_callback(self, scan):
        data = scan.ranges
        sampled_indices = np.linspace(0, len(data) - 1, self.scan_dim, dtype=int)
        self.scan_data = [data[i] for i in sampled_indices]
        
    def odom_callback(self, od_data):
        self.last_odom = od_data
        
    def costmap_callback(self, cm_data):
        map_datas = list(cm_data.data)
        if len(map_datas) < 3600:
        # 计算需要补多少个0
            num_zeros_to_add = 3600 - len(map_datas)
        # 补0
            map_datas.extend([0] * num_zeros_to_add)
            cm_data.data = tuple(map_datas)
        # print('get costmap!')
        self.last_costmap = cm_data    #60*60
        
    def guidance_callback(self,msg):
        quaternion = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        self.guidance_yaw = quaternion2yaw(quaternion)  
        # print(self.guidance_yaw ) 
        self.guidance_x = msg.pose.position.x
        self.guidance_y = msg.pose.position.y
        # # 订阅move_base服务器的消息  
        # self.goal = MoveBaseGoal()  
        # #action[0]:x_offset  action[1]:y_offset action[2]:yaw
        # # qw, qx, qy, qz=yaw_to_quaternion(action[2])
        # self.goal.target_pose.pose = Pose(Point(self.guidance_x, self.guidance_y, 0),  Quaternion.from_euler(0.0, 0.0, self.guidance_yaw))  
        # self.goal.target_pose.header.frame_id = 'odom'  
        # self.goal.target_pose.header.stamp = rospy.Time.now()
        # self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)  
        # self.move_base.send_goal(self.goal,done_cb = self.move_base_result_callback)
        
        
    def move_base_result_callback(self,status,result):
        if status == GoalStatus.SUCCEEDED:
            self.achive_flag=True
        if status == GoalStatus.ABORTED:
            self.aborted_flag=True
            
    def timer_function(self):
        #定时任务
        self.out_time_flag =True
        print('out_time_flag')               
        
    # Perform an action and read a new state
    def is_goal_valid(self,goal_x, goal_y):
        if self.last_costmap is None :
            print('last_costmap is None')
            # return False
            

        # 获取代价地图信息
        costmap_resolution = 0.05
        costmap_width = self.last_costmap.width
        costmap_height = self.last_costmap.height
        # costmap_origin = self.last_costmap.info.origin


        # 将目标点的坐标转换为代价地图坐标系
        costmap_x = int(goal_x / costmap_resolution + costmap_width/2 ) -1
        costmap_y = int(goal_y / costmap_resolution + costmap_height/2 ) -1
        # 检查目标点是否在代价地图范围内
        if costmap_x < 0 or costmap_x >= costmap_width or costmap_y < 0 or costmap_y >= costmap_height:
            print('lpg out of the costmap')
            # return False
        map_data = self.last_costmap.data
        # 获取代价地图中目标点的值
        costmap_value = map_data[costmap_x + costmap_y * costmap_width]
        
        # if (abs(goal_x+self.last_odom.pose.pose.position.x)>4.7) or (abs(goal_y+self.last_odom.pose.pose.position.y)>4.7) :
        #     return False
        
        # 检查代价地图中目标点的值是否合法（根据实际情况定义合法的阈值）
        # print(costmap_value)
        # if costmap_value <= 65:
        #     return True
        # else:
        #     return False
        self.done = False
        self.cross = False
        self.fond_global_goal = False

        threshold = 54
        d = 1
        max_d = max(costmap_width, costmap_height)
        # flag = check_pos(goal_x,goal_y)  

        x = costmap_x
        y = costmap_y       
        while d <= max_d and costmap_value > threshold :
            # self.done = True
            x,y=find_neighbors(costmap_width,costmap_height,map_data,costmap_x,costmap_y,threshold,d)
            costmap_value = map_data[x + y * costmap_width]
            d=d+1
            if d == max_d +1:
                print("valid goal not find")
        goal_x = (x+1-costmap_width/2)*costmap_resolution
        goal_y = (y+1-costmap_height/2)*costmap_resolution
        #如果目标点已经在代价地图内
        if abs(self.goal_x-self.last_odom.pose.pose.position.x) < self.max_width/2 and abs(self.goal_y-self.last_odom.pose.pose.position.y) < self.max_height/2 :
            #将全局目标点转化为is_line_cost_above_threshold需要的格式（代价地图坐标系）
            goal_x1 = self.goal_x-self.last_odom.pose.pose.position.x
            goal_y1 = self.goal_y-self.last_odom.pose.pose.position.y
            x1 = int(goal_x1 / costmap_resolution + costmap_width/2 ) - 1
            y1 = int(goal_y1 / costmap_resolution + costmap_height/2 ) - 1
            #且到达目标点不发生横跨
            if not is_line_cost_above_threshold(map_data, costmap_width,[int(costmap_width/2-1),int(costmap_height/2-1)], [x1,y1], 99):
                goal_x = goal_x1
                goal_y = goal_y1
                x = x1
                y = y1
                self.fond_global_goal = True
                
        self.cross = is_line_cost_above_threshold(map_data, costmap_width,[int(costmap_width/2-1),int(costmap_height/2-1)], [x,y], 99)
        #lpf超出场景
        if not self.manual:
            if (goal_x+self.last_odom.pose.pose.position.x)<self.env_scale_left \
            or (goal_x+self.last_odom.pose.pose.position.x)>self.env_scale_right \
            or (goal_y+self.last_odom.pose.pose.position.y)<self.env_scale_button \
            or (goal_y+self.last_odom.pose.pose.position.y)>self.env_scale_top :
                # print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                self.done = True
            # print('cost:',costmap_value)
        return goal_x,goal_y
        

         

    def step(self,action):
        self.steps += 1
        need_prefer = True
        target = False
        done = False
        goal_invalid = False
        guidance = None
        next_info =dict()
        next_info["TimeLimit.truncated"] = False
        # timer = threading.Timer(20, self.timer_functaion)
        action[0],action[1] = self.is_goal_valid(action[0],action[1])
        dx = self.goal_x - (action[0]+self.last_odom.pose.pose.position.x)
        dy = self.goal_y - (action[1]+self.last_odom.pose.pose.position.y)
        beta=math.atan2(dy, dx)
        if not (self.preferences_train) :
            '''
            # 订阅move_base服务器的消息  
            self.goal = MoveBaseGoal()  
            #action[0]:x_offset  action[1]:y_offset action[2]:yaw
            # qw, qx, qy, qz=yaw_to_quaternion(action[2])
            self.goal.target_pose.pose = Pose(Point(action[0]+self.last_odom.pose.pose.position.x, action[1]+self.last_odom.pose.pose.position.y, 0),  Quaternion.from_euler(0.0, 0.0, action[2]))  
            self.goal.target_pose.header.frame_id = 'odom'  
            self.goal.target_pose.header.stamp = rospy.Time.now()
            self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)  
            # self.move_base.wait_for_server()
            # rospy.wait_for_service("/gazebo/unpause_physics")
            # try:
            #     self.unpause()
            # except (rospy.ServiceException) as e:
            #     print("/gazebo/unpause_physics service call failed")

            # propagate state for TIME_DELTA seconds
            #time.sleep(TIME_DELTA)

            self.pub_goal_maker(self.goal)
            # self.move_base.send_goal(self.goal)  
            self.move_base.send_goal(self.goal,done_cb = self.move_base_result_callback)
            time.sleep(1)
            if self.first : 
                self.move_base.send_goal(self.goal,done_cb = self.move_base_result_callback)
                self.first =False
            
            # print('wait for guidance')
            # rospy.wait_for_message('/move_base_simple/goal', PoseStamped)
            timer.start()
            # time.sleep(0.3)
            print('send goal')
            while not (self.achive_flag or self.aborted_flag or self.out_time_flag):
                pass
            timer.cancel()
            if (self.aborted_flag or self.out_time_flag):
                # print(achive_flag)
                # print(aborted_flag)
                # print(out_time_flag)
                # self.move_base.cancel_goal()  
                done = True    
                rospy.logwarn("Navigation done")
            '''
            pass
            


            # rospy.wait_for_service("/gazebo/pause_physics")
            # try:
            #     pass
            #     self.pause()
            # except (rospy.ServiceException) as e:
            #     print("/gazebo/pause_physics service call failed")

        elif  not self.fond_global_goal:#全局目标已经在局部代价地图内且不发生横跨，无需介入偏好
            # done = True
            #step1：在代价地图中均匀采样备选点(n乘m)
            points_array = generate_discrete_points(7, 7, self.max_width, self.max_height,self.max_width/2, self.max_height/2)
            #step2：对所有备选点进行目标邻域扩展，把所有备选点变为合法点
            for i in range(len(points_array)):
                points_array[i][0],points_array[i][1] = self.is_goal_valid(points_array[i][0],points_array[i][1])
            #step3：剔除距离过近的点和距离机器人过近的点
            points_array = filter_points(points_array,0.27,0.7)
            #step4：偏好模型预测最优行动点
            self.pub_candidate_points(points_array)  # Show all candidate points
            # 生成 k 个不重复的随机数
            # random_point = points_array[random.randint(0, len(points_array)-1)]  #0-49
            # 计算每个候选点的模型得分
            scores = []
            for point in points_array:
                
                odom_x = self.last_odom.pose.pose.position.x
                odom_y = self.last_odom.pose.pose.position.y
                quaternion = Quaternion(
                    self.last_odom.pose.pose.orientation.w,
                    self.last_odom.pose.pose.orientation.x,
                    self.last_odom.pose.pose.orientation.y,
                    self.last_odom.pose.pose.orientation.z,
                )
                euler = quaternion.to_euler(degrees=False)
                angle = round(euler[2], 4)
                # Calculate distance to the goal from the robot
                distance = np.linalg.norm(
                    [odom_x - self.goal_x, odom_y - self.goal_y]
                )

                # Calculate the relative angle between the robots heading and heading toward the goal
                skew_x = self.goal_x - self.odom_x
                skew_y = self.goal_y - self.odom_y
                dot = skew_x * 1 + skew_y * 0
                mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2))
                if skew_y < 0:
                    if skew_x < 0:
                        beta = -beta
                    else:
                        beta = 0 - beta
                theta = beta - angle
                if theta > np.pi:
                    theta = np.pi - theta
                    theta = -np.pi - theta
                    
                if theta < -np.pi:
                    theta = -np.pi - theta
                    theta = np.pi - theta
                    
                # 构建状态向量
                laser_state = np.array(self.scan_data)
                laser_state[np.isinf(laser_state)] = 10  # 处理Inf
                laser_state = laser_state / 5  # 归一化
                robot_state = [distance, theta, angle]
                state = np.concatenate([laser_state, robot_state, point])
                
                # 模型预测得分
                score = self.preferences_model.predict(state)
                scores.append(score.item())  # 假设返回的是单元素数组
            # 选择最高分候选点
            max_idx = np.argmax(scores)
            optimal_point = points_array[max_idx]
            optimal_action = [optimal_point[0], optimal_point[1]]
            
            # self.pub_makers2(points_array)
            if (self.Stage == 1):#阶段一：使用强化学习输出点和偏好模型最优点作比较
                next_info['compare_action1'] = np.array([action[0],action[1]])
                next_info['compare_action2'] = np.array([optimal_action[0],optimal_action[1]])
                self.pub_makers(action,optimal_action)
            elif (self.Stage == 2):#阶段二：使用其邻域不确定性点和偏好模型最优点作比较
                # 筛选最优行动点的邻域点集
                neighbor_points = []
                neighbor_scores = []
                
                # 计算最优行动点到每个点的距离
                for i, point in enumerate(points_array):
                    dist = np.linalg.norm(np.array(point) - np.array(optimal_action))
                    if dist <= self.neighbor_radius:
                        neighbor_points.append(point)
                        neighbor_scores.append(scores[i])
                # 如果有邻域点存在
                if neighbor_points:
                    # 找到最接近决策边界的点（得分最接近0.5）
                    uncertainty_idx = np.argmin(np.abs(np.array(neighbor_scores) - 0.5))
                    uncertainty_point = neighbor_points[uncertainty_idx]
                    uncertainty_score = neighbor_scores[uncertainty_idx]
                    
                    # 计算不确定性值
                    uncertainty_value = np.abs(uncertainty_score - 0.5)
                     # 如果高于阈值，进行偏好比较
                    if uncertainty_value < self.uncertainty_threshold:
                        # 输出两个比较点
                        next_info['compare_action1'] = np.array([uncertainty_point[0], uncertainty_point[1]])
                        next_info['compare_action2'] = np.array([optimal_action[0],optimal_action[1]])
                        self.pub_makers(uncertainty_point, optimal_action)
                    else:
                        # 低不确定性，直接使用最优行动点
                        action = optimal_action
                        user_input = None
                else:#如果没有邻域点存在，认为当前点即是最优点
                    need_prefer = False
                    action = optimal_action
                    print("警告：未找到有效邻域点，默认使用最优行动点")
                    
            else:
                print('需要在偏好训练模式下指定stage 1 or 2')
                exit(0)
            while(need_prefer):
                user_input = input("waiting for human preference on stage "+str(self.Stage)+":" )
                if user_input == '0':#介入指导
                    break
                elif user_input == '1': #动作1更好
                    if (self.Stage == 1):
                        break
                    else:
                        action[0] = uncertainty_point[0]
                        action[1] = uncertainty_point[1]#这里赋值为邻域不确定性点
                        break
                elif user_input == '2': #动作2更好
                    action[0] = optimal_action[0]
                    action[1] = optimal_action[1]
                    break

                next_info['preference'] = user_input
            # rospy.wait_for_service("/gazebo/unpause_physics")
            # try:
            #     self.unpause()
            # except (rospy.ServiceException) as e:
            #     print("/gazebo/unpause_physics service call failed")
            if user_input == '0':   
                print('waiting for human guidance ...')
                rospy.wait_for_message('/move_base_simple/goal', PoseStamped)   
                # self.in_triangle = True
                '''
                timer.start()
                while not (self.achive_flag or self.aborted_flag or self.out_time_flag):
                    pass
                timer.cancel()
                if (self.aborted_flag or self.out_time_flag):
                    done = True    
                    rospy.logwarn("Navigation done")
                '''
                time.sleep(0.5)
                action[0] = self.guidance_x - self.last_odom.pose.pose.position.x
                action[1] = self.guidance_y - self.last_odom.pose.pose.position.y
                # action[2] = self.guidance_yaw
                next_info['guidance'] = np.array([action[0],action[1]])
            
                # object_state = self.set_self_state
                # object_state.pose.position.x = action[0]+self.last_odom.pose.pose.position.x
                # object_state.pose.position.y = action[1]+self.last_odom.pose.pose.position.y
                # q = Quaternion.from_euler(0, 0, self.guidance_yaw)
                # # object_state.pose.position.z = 0.
                # object_state.pose.orientation.x = q.x
                # object_state.pose.orientation.y = q.y
                # object_state.pose.orientation.z = q.z
                # object_state.pose.orientation.w = q.w
                # self.set_state.publish(object_state)      
            

            # rospy.wait_for_service("/gazebo/pause_physics")
            # try:
            #     pass
            #     self.pause()
            # except (rospy.ServiceException) as e:
            #     print("/gazebo/pause_physics service call failed")
        object_state = self.set_self_state
        object_state.pose.position.x = action[0]+self.last_odom.pose.pose.position.x
        object_state.pose.position.y = action[1]+self.last_odom.pose.pose.position.y
        # q = tf.transformations.quaternion_from_euler(0, 0, action[2])
        q = Quaternion.from_euler(0, 0, beta)
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = q.x
        object_state.pose.orientation.y = q.y
        object_state.pose.orientation.z = q.z
        object_state.pose.orientation.w = q.w
        self.set_state.publish(object_state)  
        self.reset_flags()
        # read velodyne laser state
        #done, collision, min_laser = self.observe_collision(self.scan_data)
        time.sleep(0.6)
        costmap_state = [self.last_costmap.data]
        laser_state = [self.scan_data]
        #####################代价地图数据###################
        next_state_costmap = np.array(costmap_state)
        ######################激光数据######################
        next_state_laser = np.array(laser_state)
        # 使用np.isinf()函数找到所有的inf值
        inf_indices = np.isinf(next_state_laser)
        # 使用布尔索引将inf值替换为10
        next_state_laser[inf_indices] = 10
        next_state_laser = next_state_laser/5
        #################### 机器人状态####################
        # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
         # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
            
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True
        if self.done:
            done = True
            goal_invalid = True
        elif not self.steps < self.max_episode_steps:
            next_info["TimeLimit.truncated"] = True
            # print('time out')
        if target:
            next_info["Target"] = True
        else:
            next_info["Target"] = False

        robot_state = [distance, theta, angle]
        state = np.append(next_state_laser, robot_state)
        reward = self.get_reward(goal_invalid,self.cross,self.distance-distance,target,theta,self.steps,distance,action)
        self.distance = distance
        #return state, reward, done, target

        return action, state, reward, done , next_info 

    def reset(self,**kwargs):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state
        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(self.env_scale_left, self.env_scale_right)
            y = np.random.uniform(self.env_scale_button, self.env_scale_top)
            position_ok = check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # set a random goal in empty space in environment
        self.change_goal()
        # randomly scatter boxes in the environment
        ###########################################################
        # self.random_box()
        ############################################################
        # self.publish_markers([0.0, 0.0])
        # rospy.wait_for_service("/gazebo/unpause_physics")
        # try:
        #     self.unpause()
        # except (rospy.ServiceException) as e:
        #     print("/gazebo/unpause_physics service call failed")

        time.sleep(1)

        # rospy.wait_for_service("/gazebo/pause_physics")
        # try:
        #     self.pause()
        # except (rospy.ServiceException) as e:
        #     print("/gazebo/pause_physics service call failed")
        #####################代价地图数据###################
        costmap_state = self.last_costmap.data
        ######################激光数据######################
        laser_state = self.scan_data
        #########################机器人状态#################
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )
        self.distance = distance
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, angle]
        ###################################################
        state_laser = np.array(laser_state)
        # 使用np.isinf()函数找到所有的inf值
        inf_indices = np.isinf(state_laser)

        # 使用布尔索引将inf值替换为10
        state_laser[inf_indices] = 10
        #归一化
        state_laser = state_laser/5 
        
        state_costmap = np.array(costmap_state)
        robot_state = np.array(robot_state)
        state = np.append(state_laser, robot_state)
        #激光数据，机器人状态（当前位置和朝向，与目标的距离和角度），代价地图数据
        info =dict()
        info["TimeLimit.truncated"] = False
        info["Target"] = False
        self.steps = 0
        return state,info
    
    def change_goal(self):
        if self.manual:
            msg = rospy.wait_for_message("/move_base_simple/goal", PoseStamped)
            self.goal_x = msg.pose.position.x
            self.goal_y = msg.pose.position.y
        else:
            # Place a new goal and check if its location is not on one of the obstacles
            # if self.upper < 10:
            #     self.upper += 0.004
            # if self.lower > -10:
            #     self.lower -= 0.004

            goal_ok = False

            while not goal_ok:
                # self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
                # self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
                self.goal_x = np.random.uniform(self.env_scale_left, self.env_scale_right)
                self.goal_y = np.random.uniform(self.env_scale_button, self.env_scale_top)
                goal_ok = check_pos(self.goal_x, self.goal_y)
                if (abs(self.odom_x - self.goal_x)<(self.max_width/2) ) or (abs(self.odom_y - self.goal_y)<(self.max_height/2)) :
                    goal_ok = False
                    
        self.global_distance = np.linalg.norm([self.goal_x - self.odom_x, self.goal_y - self.odom_y])
        
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0
        markerArray.markers.append(marker)
        self.marker_pub1.publish(markerArray)

    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < (self.max_width/2) or distance_to_goal < (self.max_height/2):
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)
    # def pub_goal_maker(self,goal):
    #     goal_x = goal.target_pose.pose.position.x
    #     goal_y = goal.target_pose.pose.position.y
    #     markerArray = MarkerArray()
    #     marker = Marker()
    #     marker.header.frame_id = "odom"
    #     marker.type = marker.CYLINDER
    #     marker.action = marker.ADD
    #     marker.scale.x = 0.1
    #     marker.scale.y = 0.1
    #     marker.scale.z = 0.01
    #     marker.color.a = 1.0
    #     marker.color.r = 1.0
    #     marker.color.g = 0.0
    #     marker.color.b = 0.0
    #     marker.pose.orientation.w = 1.0
    #     marker.pose.position.x = goal_x
    #     marker.pose.position.y = goal_y
    #     marker.pose.position.z = 0

    #     markerArray.markers.append(marker)
    #     self.publisher.publish(markerArray)
        
    def pub_makers(self, action1, action2):
            # Publish visual data in Rviz
            markerArray = MarkerArray()
            
            # Marker 1 - Sphere
            marker1 = Marker()
            marker1.header.frame_id = "odom"
            marker1.ns = "markers"
            marker1.id = 0
            marker1.header.stamp = rospy.Time.now()
            marker1.type = Marker.SPHERE
            marker1.action = Marker.ADD
            marker1.scale.x = 0.3  # 3x original size (0.1 → 0.3)
            marker1.scale.y = 0.3  # 3x original size
            marker1.scale.z = 0.3  # 3x original size
            marker1.color.a = 1.0
            marker1.color.r = 0.0
            marker1.color.g = 1.0
            marker1.color.b = 1.0
            marker1.pose.position.x = self.last_odom.pose.pose.position.x + action1[0]
            marker1.pose.position.y = self.last_odom.pose.pose.position.y + action1[1]
            marker1.pose.position.z = 0
            markerArray.markers.append(marker1)
            
            # Text marker for number 1
            text1 = Marker()
            text1.header.frame_id = "odom"
            text1.ns = "text_markers"
            text1.id = 10
            text1.header.stamp = rospy.Time.now()
            text1.type = Marker.TEXT_VIEW_FACING
            text1.action = Marker.ADD
            text1.scale.z = 0.3  # 3x original text size (0.1 → 0.3)
            text1.color.a = 1.0
            text1.color.r = 0.0  # Black text
            text1.color.g = 0.0  # Black text
            text1.color.b = 0.0  # Black text
            text1.text = "1"
            # For bold text, we can't directly set bold in RViz markers, but increasing size helps visibility
            text1.pose.position.x = self.last_odom.pose.pose.position.x + action1[0]
            text1.pose.position.y = self.last_odom.pose.pose.position.y + action1[1]
            text1.pose.position.z = 0.3  # Adjusted for larger sphere (original 0.1 → 0.3)
            markerArray.markers.append(text1)

            if not action2 == None:
                # Marker 2 - Sphere
                marker2 = Marker()
                marker2.header.frame_id = "odom"
                marker2.ns = "markers"
                marker2.id = 1
                marker2.header.stamp = rospy.Time.now()
                marker2.type = Marker.SPHERE
                marker2.action = Marker.ADD
                marker2.scale.x = 0.3  # 3x original size
                marker2.scale.y = 0.3  # 3x original size
                marker2.scale.z = 0.3  # 3x original size
                marker2.color.a = 1.0
                marker2.color.r = 1.0
                marker2.color.g = 0.0
                marker2.color.b = 1.0
                marker2.pose.position.x = self.last_odom.pose.pose.position.x + action2[0]
                marker2.pose.position.y = self.last_odom.pose.pose.position.y + action2[1]
                marker2.pose.position.z = 0
                markerArray.markers.append(marker2)
                
                # Text marker for number 2
                text2 = Marker()
                text2.header.frame_id = "odom"
                text2.ns = "text_markers"
                text2.id = 11
                text2.header.stamp = rospy.Time.now()
                text2.type = Marker.TEXT_VIEW_FACING
                text2.action = Marker.ADD
                text2.scale.z = 0.3  # 3x original text size
                text2.color.a = 1.0
                text2.color.r = 0.0  # Black text
                text2.color.g = 0.0  # Black text
                text2.color.b = 0.0  # Black text
                text2.text = "2"
                text2.pose.position.x = self.last_odom.pose.pose.position.x + action2[0]
                text2.pose.position.y = self.last_odom.pose.pose.position.y + action2[1]
                text2.pose.position.z = 0.3  # Adjusted for larger sphere
                markerArray.markers.append(text2)
            
            self.marker_pub2.publish(markerArray)

    def pub_makers2(self, markers):
            # Publish visual data in Rviz
            markerArray = MarkerArray()
            for i in range (len(markers)):
                marker1 = Marker()
                marker1.header.frame_id = "odom"
                marker1.ns = "markers"
                marker1.id = i
                marker1.header.stamp = rospy.Time.now()
                marker1.type = marker1.CYLINDER
                marker1.action = marker1.ADD
                marker1.scale.x = 0.12
                marker1.scale.y = 0.12
                marker1.scale.z = 0.01
                marker1.color.a = 1.0
                marker1.color.r = 0.0
                marker1.color.g = 1.0
                marker1.color.b = 1.0
                marker1.pose.orientation.x = 0
                marker1.pose.orientation.y = 0
                marker1.pose.orientation.z = 0
                marker1.pose.orientation.w = 1
                marker1.pose.position.x = self.last_odom.pose.pose.position.x+markers[i][0]
                marker1.pose.position.y = self.last_odom.pose.pose.position.y+markers[i][1]
                marker1.pose.position.z = 0
                markerArray.markers.append(marker1)
            self.marker_pub2.publish(markerArray)
    
    def pub_candidate_points(self, points_array):
        """发布候选点标记（持久显示，直到下次更新）"""
        markerArray = MarkerArray()
        
        # ================== 清除旧标记 ==================
        delete_all_marker = Marker()
        delete_all_marker.header.frame_id = "odom"     # 必须与后续标记坐标系一致
        delete_all_marker.ns = "candidate_points"      # 指定要清除的命名空间
        delete_all_marker.action = Marker.DELETEALL    # 清除操作指令
        markerArray.markers.append(delete_all_marker)

        # ================== 生成新标记 ==================
        for idx, (x_offset, y_offset) in enumerate(points_array):
            # 计算全局坐标
            global_x = self.last_odom.pose.pose.position.x + x_offset
            global_y = self.last_odom.pose.pose.position.y + y_offset

            # --- 主标记（球体）---
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.ns = "candidate_points"
            marker.id = idx  # ID只需保证本次唯一性
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            # 视觉设置
            marker.scale.x = 0.15  # 直径X
            marker.scale.y = 0.15  # 直径Y
            marker.scale.z = 0.1  # 高度Z
            marker.color.r = 0.2
            marker.color.g = 0.8
            marker.color.b = 0.2
            marker.color.a = 0.9  # 不透明度
            # 位置设置
            marker.pose.position.x = global_x
            marker.pose.position.y = global_y 
            marker.pose.position.z = 0.05  # 稍微抬升避免Z-fighting
            marker.pose.orientation.w = 1.0
            # 持久化设置
            marker.lifetime = rospy.Duration(0)  # 0表示无限持续时间

            # 添加至数组
            markerArray.markers.append(marker)

        # ================== 发布标记 ==================
        try:
            self.marker_pub2.publish(markerArray)
        except rospy.ROSException as e:
            rospy.logerr("Marker publish failed: {}".format(str(e)))
                
    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    
            

