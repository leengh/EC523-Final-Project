from typing import Tuple
from gym.envs.mujoco import mujoco_env
from gym import utils, spaces
import numpy as np
from pyquaternion import Quaternion
import time
from pathlib import Path
import os
import cv2 as cv

from sympy import Integer
from Camera import Camera

from Environment import Environment
from UR5_Arm import UR5

from termcolor import colored
import copy
from scipy.spatial.distance import cdist


class GraspingEnvironment(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, model_type="binary", train=False) -> None:
        self.model_type = model_type
        self.number_of_objects = 20 if train else 6  # objects to pick
        self.total_picked_objects = 0  # used for training
        self.training = train
        self.render_environment = not train

        self.IMAGE_WIDTH = 200
        self.IMAGE_HEIGHT = 200
        self.grasp_counter = 0
        self.rotations = {0: 0, 1: 30, 2: 60, 3: 90, 4: -30, 5: -60, 6: -90}
        utils.EzPickle.__init__(self)

        if self.training:
            model_path = "/environment/training_env.xml"
        else:
            model_path = "/environment/testing_env.xml"

        full_path = self.get_full_path(model_path)
        self.initialized = False
        self.environment = Environment()
        self.camera = Camera()
        self.robot = UR5()
        self.controller = self.robot.controller
        self.action_space_type = "multidiscrete"
        self.pick_box = np.array([[0.0, 0.0, 0.91],
                                  [0.23, 0.0, 0.91],
                                  [-0.23, 0.0, 0.91],
                                  [0.0, 0.3, 0.91],
                                  [0.0, -0.3, 0.91]])

        mujoco_env.MujocoEnv.__init__(self, model_path=full_path, frame_skip=1)

        if self.render_environment:
            self.render()

    def _set_action_space(self):
        if self.action_space_type == "discrete":
            size = self.IMAGE_WIDTH * self.IMAGE_HEIGHT
            self.action_space = spaces.Discrete(size)
        elif self.action_space_type == "multidiscrete":
            self.action_space = spaces.MultiDiscrete(
                [self.IMAGE_HEIGHT * self.IMAGE_WIDTH, len(self.rotations)]
            )

        return self.action_space

    def get_full_path(self, model_path):
        path = os.path.realpath(__file__)
        full_path = str(Path(path).parent.parent.parent) + model_path
        return full_path

    def step(self, action=None) -> Tuple[object, Integer, bool, object]:

        done = False
        info = {}
        observation = None
        reward = 0

        if not self.initialized:
            # set observation_space in MujoCo
            self.camera.initialize_camera(model=self.model)
            self.controller.set_actuators(sim=self.sim, model=self.model)
            observation = {}
            observation["rgb"] = np.zeros(
                (self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3))
            observation["depth"] = np.zeros(
                (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
            self.current_observation = observation
            self.initialized = True
            return observation, reward, done, info

        observation = self.get_observation()

        x = action[0] % self.IMAGE_WIDTH
        y = action[0] // self.IMAGE_WIDTH
        rotation = action[1]

        depth = observation["depth"][y][x]
        coordinates = self.camera.convert_pixel_to_world_coordinates(
            pixel_x=x, pixel_y=y, depth=depth, height=self.IMAGE_HEIGHT, width=self.IMAGE_WIDTH
        )

        print(
            colored(
                "Action: Pixel X: {}, Pixel Y: {}, Rotation: {} ({} deg)".format(
                    x, y, rotation, self.rotations[rotation]
                ),
                color="blue",
                attrs=["bold"],
            )
        )
        print(
            colored(
                "Transformed into world coordinates: {}".format(
                    coordinates[:2]),
                color="blue",
                attrs=["bold"],
            )
        )

        if coordinates[2] < 0.8 or coordinates[1] > -0.3:
            success = False
            reward = 0
            print(
                colored(
                    "Skipping execution because coordinate is not a point on the table", color="red", attrs=["bold"]
                )
            )
        else:
            if self.model_type == "binary":
                success, reward = self.binary_reward(coordinates, rotation)
            elif self.model_type == "categorical":
                success, reward = self.categorical_reward(
                    coordinates, rotation)
            elif self.model_type == "euclidean":
                success, reward = self.euclidean_reward(coordinates, rotation)
            elif self.model_type == "manhattan":
                success, reward = self.manhattan_reward(coordinates, rotation)

        if success:
            self.total_picked_objects = self.total_picked_objects + 1
            if self.total_picked_objects == self.number_of_objects:
                done = True

        observation = self.get_observation()
        self.current_observation = observation

        info = {"success": success}

        return observation, reward, done, info

    def binary_reward(self, coordinates, rotation):
        success = self.grasp_object(
            target_coordinates=coordinates, rotation=rotation)

        reward = 1 if success else 0

        return success, reward

    def categorical_reward(self, coordinates, rotation):
        z = coordinates[2]

        if z > 1.1:
            print(
                colored(
                    "Skipping execution because robot cant reach table!", color="red", attrs=["bold"]
                )
            )
            return False, 0

        if z <= 1.1 and z > 1:
            reward = 0.25
            print(
                colored(
                    "Skipping execution because robot cant reach table!", color="red", attrs=["bold"]
                )
            )
            return False, reward

        if z <= 1 and z > 0.975:
            reward = 0.5
            print(
                colored(
                    "Skipping execution because robot cant reach table!", color="red", attrs=["bold"]
                )
            )
            return False, reward

        success = self.grasp_object(
            target_coordinates=coordinates, rotation=rotation)

        if success:
            reward = 2
        elif z <= 0.975 and z > 0.8:
            reward = 0.75

        return success, reward

    def euclidean_reward(self, coordinates, rotation):
        success = self.grasp_object(
            target_coordinates=coordinates, rotation=rotation)

        if not success:
            reward = -1 * \
                min(cdist(self.pick_box, coordinates.reshape(1, 3), 'euclidean'))
            reward = reward[0]
        else:
            reward = 1

        return success, reward

    def manhattan_reward(self, coordinates, rotation, reward_type=2):
        success = self.grasp_object(
            target_coordinates=coordinates, rotation=rotation)

        delta = 0.05
        reward = 0
        ee_pos = self.robot.get_end_effector_position()
        distance = abs(ee_pos - coordinates)
        if reward_type == 0:
            # Prioritized Distance from Z
            weight = np.array([1, 1, 5])
            reward = -sum(distance * weight)
        elif reward_type == 1:
            # Prioritized Distance from XYZ
            weight = np.array([10, 5, 1])
            reward = -sum(distance * weight)
        elif reward_type == 2:
            # anhattan Trajectory
            weight = [i <= delta for i in distance]
            fixed_penalty = np.array([5, 2.5, 1])
            reward = -sum(fixed_penalty * weight)

        return success, reward

    def grasp_object(self, target_coordinates, rotation) -> bool:
        success = False
        # Move end effector above the target
        target = copy.deepcopy(target_coordinates)
        target[2] = 1.1
        is_ee_above_target, max_steps_reached = self.robot.move_end_effector(
            target=target, tolerance=0.05, max_steps=1000, sim=self.sim, model=self.model, viewer=self.viewer, render=self.render_environment)

        if not is_ee_above_target and not max_steps_reached:
            # Try to move it back to its initial position
            _, max_steps_reached = self.robot.move_end_effector(
                target=[0.0, -0.6, 1.1], tolerance=0.05, max_steps=1000, sim=self.sim, model=self.model, viewer=self.viewer, render=self.render_environment)

        if not max_steps_reached:
            # Rotate gripper according to second action dimension
            rotation_success, _ = self.robot.rotate_wrist_3_joint(
                self.rotations[rotation], sim=self.sim, viewer=self.viewer, render=self.render_environment)
            self.robot.toggle_gripper(
                open=True, half=True, sim=self.sim, viewer=self.viewer, render=self.render_environment)
            # Move to grasping height
            target = copy.deepcopy(target_coordinates)
            target[2] = max(self.environment.TABLE_HEIGHT, target[2] - 0.01)
            _, max_steps_reached = self.robot.move_end_effector(
                target=target, tolerance=0.01, max_steps=300, sim=self.sim, model=self.model, viewer=self.viewer, render=self.render_environment)
            if max_steps_reached:
                print("Max steps reaches, can't reach target")
                success = False
            else:
                success = self.robot.grasp(
                    sim=self.sim, viewer=self.viewer, render=self.render_environment)
        else:
            success = False

        self.controller.actuators[0][3].Kp = 10.0

        self.robot.move_end_effector(
            target=self.environment.initial_position, tolerance=0.05, max_steps=1000, sim=self.sim, model=self.model, viewer=self.viewer, render=self.render_environment)

        # Check if object is in gripper
        object_in_gripper = self.robot.is_object_in_gripper(
            sim=self.sim, viewer=self.viewer, render=self.render_environment)

        drop_position = self.environment.goal_1

        if success and object_in_gripper:
            self.stay(1000)
            self.grasp_counter += 1

            print(colored("Successful grasp!", color="green", attrs=["bold"]))

            shape, _ = self.get_grasped_item_info()
            if shape == 'cube':
                drop_position = self.environment.goal_1
            else:
                drop_position = self.environment.goal_2
        elif not object_in_gripper:
            print(colored("Did not grasp anything.",
                  color="red", attrs=["bold"]))

        self.robot.move_end_effector(
            target=drop_position, tolerance=0.01, max_steps=1200, sim=self.sim, model=self.model, viewer=self.viewer, render=self.render_environment)

        # drop item
        self.robot.toggle_gripper(
            open=True, sim=self.sim, viewer=self.viewer, render=self.render_environment)
        # Move back to zero rotation
        self.robot.rotate_wrist_3_joint(
            0, sim=self.sim, viewer=self.viewer, render=self.render_environment)
        self.controller.actuators[0][3].Kp = 20.0

        return success and object_in_gripper

    def get_grasped_item_info(self) -> str:
        capture_rgb, _ = self.camera.capture_photo(
            width=1000, height=1000, camera="side", sim=self.sim
        )
        img_name = "./grasps/" + self.model_type + \
            "/Grasp_{}.png".format(self.grasp_counter)
        cv.imwrite(img_name, cv.cvtColor(capture_rgb, cv.COLOR_BGR2RGB))
        img = cv.imread(img_name)
        shape = self.camera.get_object_shape(img)
        return shape, img

    def get_observation(self) -> object:
        """
        """
        rgb, depth = self.camera.capture_photo(
            camera="top_down", sim=self.sim, height=self.IMAGE_HEIGHT, width=self.IMAGE_WIDTH)
        depth = self.environment.convert_depth_to_meters(
            depth=depth, model=self.model)

        # img_name = "observation.png"
        # cv.imwrite(img_name, cv.cvtColor(rgb, cv.COLOR_BGR2RGB))

        return {"rgb": rgb, "depth": depth}

    def reset_model(self):
        actuated_joint_ids = self.controller.get_actuated_joint_ids()

        qpos = self.data.qpos
        qvel = self.data.qvel

        qpos[actuated_joint_ids] = [
            0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.3]

        n_objects = self.number_of_objects

        for i in range(n_objects):
            joint_name = f"free_joint_{i}"
            q_adr = self.model.get_joint_qpos_addr(joint_name)
            start, end = q_adr
            qpos[start] = np.random.uniform(low=-0.25, high=0.25)
            qpos[start + 1] = np.random.uniform(low=-0.77, high=-0.43)
            qpos[start + 2] = np.random.uniform(low=1.0, high=1.5)
            qpos[start + 3: end] = Quaternion.random().unit.elements

        self.set_state(qpos, qvel)

        for index in range(6):
            self.controller.current_target_joint_values[index] = qpos[actuated_joint_ids][index]

        self.stay(8000)
        self.total_picked_objects = 0

        return self.get_observation()

    def stay(self, duration=1000):
        starting_time = time.time()
        elapsed = 0
        while elapsed < duration:
            elapsed = (time.time() - starting_time) * 1000
