import argparse
import datetime
import glob
import os
import rospy
from typing import Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import pandas as pd
from gym import wrappers
from copy import copy
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion,Twist,PoseStamped
#from gops.create_pkg.create_env_model import create_env_model
from initialization import create_env
from plot_evaluation import cm2inch
from common_utils import get_args_from_json, mp4togif
#from gops.sys_simulator.opt_controller import OptController
from preference_model import HumanPreference
from preferences_buffer import PreferencesBuffer


default_cfg = dict()
default_cfg["fig_size"] = (12, 9)
default_cfg["dpi"] = 300
default_cfg["pad"] = 0.5

default_cfg["tick_size"] = 8
default_cfg["tick_label_font"] = "Times New Roman"
default_cfg["legend_font"] = {
    "family": "Times New Roman",
    "size": "8",
    "weight": "normal",
}
default_cfg["label_font"] = {
    "family": "Times New Roman",
    "size": "9",
    "weight": "normal",
}

default_cfg["img_fmt"] = "png"


class PreferencesTrainer:
    """Plot module for trained policy

    :param list log_policy_dir_list: directory of trained policy.
    :param list trained_policy_iteration_list: iteration of trained policy.
    :param bool save_render: save environment animation or not.
    :param list plot_range: customize plot range.
    :param bool is_init_info: customize initial information or not.
    :param dict init_info: initial information.
    :param list legend_list: legends of figures.
    :param dict opt_args: arguments of optimal solution solver.
    :param bool constrained_env: constraint environment or not.
    :param bool is_tracking: tracking problem or not.
    :param bool use_dist: use adversarial action or not.
    :param float dt: time interval between steps.
    :param str obs_noise_type: type of observation noise, "normal" or "uniform".
    :param list obs_noise_data: Mean and
        Standard deviation of Normal distribution or Upper
        and Lower bounds of Uniform distribution.
    :param str action_noise_type: type of action noise, "normal" or "uniform".
    :param list action_noise_data: Mean and
        Standard deviation of Normal distribution or Upper
        and Lower bounds of Uniform distribution.
    """

    def __init__(
        self,
        log_policy_dir_list: list,
        trained_policy_iteration_list: list,
        save_render: bool = False,
        plot_range: list = None,
        is_init_info: bool = False,
        init_info: dict = None,
        legend_list: list = None,
        opt_args: dict = None,
        constrained_env: bool = False,
        is_tracking: bool = False,
        use_dist: bool = False,
        manual: bool = False,
        stage: int = 1,
        dt: float = None,
        obs_noise_type: str = None,
        obs_noise_data: list = None,
        action_noise_type: str = None,
        action_noise_data: list = None,
    ):
        self.log_policy_dir_list = log_policy_dir_list
        self.trained_policy_iteration_list = trained_policy_iteration_list
        self.save_render = save_render
        self.args = None
        self.plot_range = plot_range
        if is_init_info:
            self.init_info = init_info
        else:
            self.init_info = {}
        self.legend_list = legend_list
        self.opt_args = opt_args
        self.constrained_env = constrained_env
        self.use_dist = use_dist
        self.is_tracking = is_tracking
        self.manual = manual
        self.stage = stage
        self.dt = dt
        self.policy_num = len(self.log_policy_dir_list)
        if self.policy_num != len(self.trained_policy_iteration_list):
            raise RuntimeError(
                "The length of policy number is not equal to that of policy iteration"
            )
        self.obs_noise_type = obs_noise_type
        self.obs_noise_data = obs_noise_data
        self.action_noise_type = action_noise_type
        self.action_noise_data = action_noise_data
        self.ref_state_num = 0

        # data for plot
        self.args_list = []
        self.eval_list = []
        self.env_id_list = []
        self.algorithm_list = []
        self.tracking_list = []
        
        self.suc = 0
        self.num = 0

        self.__load_all_args()
        self.env_id = self.get_n_verify_env_id()

        # save path
        path = os.path.join(os.path.dirname(__file__), "..",  "figures")
        path = os.path.abspath(path)

        algs_name = ""
        for item in self.algorithm_list:
            algs_name = algs_name + item + "-"
        self.save_path = os.path.join(
            path,
            algs_name + self.env_id,
            datetime.datetime.now().strftime("%y%m%d-%H%M%S"),
        )
        os.makedirs(self.save_path, exist_ok=True)
        
        self.preferences_buffer = PreferencesBuffer(buffer_size = 500000, filename = 'buffer_data.pkl',random_seed = 0)
        self.preferences_buffer.load_from_file()
        self.preferences_model = HumanPreference(input_dim = 41)
    def run_an_episode(
        self,
        env: Any,
        controller: Any,
        init_info: dict,
        is_opt: bool,
        render: bool = True,
    ) -> Tuple[dict, dict]:
        state_list = []
        action_list = []
        reward_list = []
        constrain_list = []
        obs_list = []
        step = 0
        step_list = []
        info_list = [init_info]
        obs, info = env.reset(**init_info)
        obs, info = env.reset(**init_info)
        # state = env.state
        state = obs
        print("Initial state: ")
        print(self.__convert_format(state))
        # plot tracking
        state_with_ref_error = {}
        
        while self.num < 50000:
            done = False
            info["TimeLimit.truncated"] = False
            self.num = self.num + 1
            
                
            while not (done or info["TimeLimit.truncated"]):
                state_list.append(state)
                obs_list.append(obs)
                if is_opt:
                    action = controller(obs, info)
                else:
                    action = self.compute_action(obs, controller)
                    action = self.__action_noise(action)
                if self.use_dist:
                    action = np.hstack((action, env.dist_func(step * env.tau)))
                if self.constrained_env:
                    constrain_list.append(info["constraint"])
                if self.is_tracking:
                    state_num = len(info["ref"])
                    self.ref_state_num = sum(x is not None for x in info["ref"])
                    if step == 0:
                        for i in range(state_num):
                            if info["ref"][i] is not None:
                                state_with_ref_error["state-{}".format(i)] = []
                                state_with_ref_error["ref-{}".format(i)] = []
                                state_with_ref_error["state-{}-error".format(i)] = []

                    for i in range(state_num):
                        if info["ref"][i] is not None:
                            state_with_ref_error["state-{}".format(i)].append(
                                info["state"][i]
                            )
                            state_with_ref_error["ref-{}".format(i)].append(info["ref"][i])
                            state_with_ref_error["state-{}-error".format(i)].append(
                                info["ref"][i] - info["state"][i]
                            )
                # print(action)
                action, next_obs, reward, done, info = env.step(action)
                # if info["Target"]:
                #     self.suc = self.suc + 1
                if (not done) and ('preference' in info):
                    if info['preference'] == '0':
                        self.preferences_buffer.add(
                            np.hstack((np.array(obs), info['guidance'])),
                            np.hstack((np.array(obs), info['compare_action1'])),
                            np.array([1,0])
                        )
                        self.preferences_buffer.add(
                            np.hstack((np.array(obs), info['guidance'])),
                            np.hstack((np.array(obs), info['compare_action2'])),
                            np.array([1,0])
                        )
                    elif info['preference'] == '1':
                        self.preferences_buffer.add(
                            np.hstack((np.array(obs), info['compare_action1'])),
                            np.hstack((np.array(obs), info['compare_action2'])),
                            np.array([1,0])
                        )
                    else:
                        self.preferences_buffer.add(
                            np.hstack((np.array(obs), info['compare_action2'])),
                            np.hstack((np.array(obs), info['compare_action1'])),
                            np.array([1,0])
                        )
                    
                n =  1 + int(self.preferences_buffer.size()/20 * 0.3)   
                for i in range(n):
                    self.preferences_model.train(self.preferences_buffer,batch_size = 20)

                action_list.append(action)
                step_list.append(step)
                reward_list.append(reward)
                info_list.append(info)

                obs = next_obs
                # state = env.state
                state = obs
                step = step + 1
                # print("step:", step)

                if "TimeLimit.truncated" not in info.keys():
                    info["TimeLimit.truncated"] = False
                # Draw environment animation
                if render:
                    env.render()
                if done or info["TimeLimit.truncated"]:
                    obs, info = env.reset()
                    break
            # suc_rate = self.suc / self.num *100
            # print("第%d轮测试,成功%d次,成功率为%.1f%%" % (self.num,self.suc,suc_rate))

        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "state_list": state_list,
            "step_list": step_list,
            "obs_list": obs_list,
            "info_list": info_list,
        }
        if self.constrained_env:
            eval_dict.update(
                {"constrain_list": constrain_list,}
            )

        if self.is_tracking:
            tracking_dict = state_with_ref_error
        else:
            tracking_dict = {}

        return eval_dict, tracking_dict

    def compute_action(self, obs: np.ndarray, networks: Any) -> np.ndarray:
        batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
        logits = networks.policy(batch_obs)
        action_distribution = networks.create_action_distributions(logits)
        action = action_distribution.mode()#***************
        # print(action)
        action = action.detach().numpy()[0]
        return action

    @staticmethod
    def __load_args(log_policy_dir: str):
        json_path = os.path.join(log_policy_dir, "config.json")
        parser = argparse.ArgumentParser()
        args_dict = vars(parser.parse_args())
        args = get_args_from_json(json_path, args_dict)
        return args

    def __load_all_args(self):
        for i in range(self.policy_num):
            log_policy_dir = self.log_policy_dir_list[i]
            args = self.__load_args(log_policy_dir)
            self.args_list.append(args)
            env_id = args["env_id"]
            self.env_id_list.append(env_id)
            self.algorithm_list.append(args["algorithm"])

    def __load_env(self):
        env_args = {
            **self.args,
            "obs_noise_type": self.obs_noise_type,
            "obs_noise_data": self.obs_noise_data,
            "action_noise_type": self.action_noise_type,
            "action_noise_data": self.action_noise_data,
            "manual":self.manual,
            "stage":self.stage,
            "preference_training": True,
            "preferences_model": self.preferences_model
        }
        env = create_env(**env_args)
        if self.save_render:
            video_path = os.path.join(self.save_path, "videos")
            env = wrappers.RecordVideo(
                env, video_path, name_prefix="{}_video".format(self.args["algorithm"])
            )
        self.args["action_high_limit"] = env.action_high
        self.args["action_low_limit"] = env.action_low
        return env

    def __load_policy(self, log_policy_dir: str, trained_policy_iteration: str):
        # Create policy
        alg_name = self.args["algorithm"]
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, "ApproxContainer")
        networks = ApproxContainer(**self.args)

        # Load trained policy
        log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(
            trained_policy_iteration
        )
        networks.load_state_dict(torch.load(log_path))
        return networks

    def __convert_format(self, origin_data_list: list):
        data_list = copy(origin_data_list)
        for i in range(len(origin_data_list)):
            if isinstance(origin_data_list[i], list) or isinstance(
                origin_data_list[i], np.ndarray
            ):
                data_list[i] = self.__convert_format(origin_data_list[i])
            else:
                data_list[i] = "{:.2g}".format(origin_data_list[i])
        return data_list

    def __run_data(self):
        for i in range(self.policy_num):
            log_policy_dir = self.log_policy_dir_list[i]
            trained_policy_iteration = self.trained_policy_iteration_list[i]

            self.args = self.args_list[i]
            print("===========================================================")
            print("*** Begin to run policy {} ***".format(i + 1))
            env = self.__load_env()
            if hasattr(env, "set_mode"):
                env.set_mode("test")

            if hasattr(env, "train_space") and hasattr(env, "work_space"):
                print("Train space: ")
                print(self.__convert_format(env.train_space))
                print("Work space: ")
                print(self.__convert_format(env.work_space))
            networks = self.__load_policy(log_policy_dir, trained_policy_iteration)

            # Run policy
            eval_dict, tracking_dict = self.run_an_episode(
                env, networks, self.init_info, is_opt=False, render=False
            )
            print("Successfully run policy {}".format(i + 1))
            print("===========================================================\n")
            # mp4 to gif
            self.eval_list.append(eval_dict)
            self.tracking_list.append(tracking_dict)


    def __action_noise(self, action: np.ndarray) -> np.ndarray:
        if self.action_noise_type is None:
            return action
        elif self.action_noise_type == "normal":
            return action + np.random.normal(
                loc=self.action_noise_data[0], scale=self.action_noise_data[1]
            )
        elif self.action_noise_type == "uniform":
            return action + np.random.uniform(
                low=self.action_noise_data[0], high=self.action_noise_data[1]
            )


    def get_n_verify_env_id(self):
        env_id = self.env_id_list[0]
        for i, eid in enumerate(self.env_id_list):
            assert (
                env_id == eid
            ), "GOPS: policy {} is not trained in the same environment".format(i)
        return env_id

    def run(self):
        self.__run_data()
