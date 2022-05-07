

import math
from typing import Tuple
import ikpy as ik
import ikpy.chain
from Controller import Controller
import numpy as np


class UR5():

    def __init__(self) -> None:
        self.chain = ik.chain.Chain.from_urdf_file(
            "./environment/ur5_gripper.urdf")
        self.controller = Controller()

    def inverse_kinematics(self, target_position: Tuple[float, float], sim=None, model=None) -> Tuple:
        """
        Returns the angle of each joint
        """
        base_link = model.body_name2id("base_link")
        ee_position_base = (
            target_position - sim.data.body_xpos[base_link]
        )
        gripper_center_position = ee_position_base + [0, -0.005, 0.16]
        joint_angles = self.chain.inverse_kinematics(
            gripper_center_position, [0, 0, -1], orientation_mode="X")
        prediction = (
            self.forward_kinematics(joint_angles)[:3, 3]
            + sim.data.body_xpos[model.body_name2id("base_link")]
            - [0, -0.005, 0.16]
        )
        diff = abs(prediction - target_position)
        error = np.sqrt(diff.dot(diff))
        joint_angles = joint_angles[1:-2]
        if error <= 0.02:
            return joint_angles

        # Failed to find IK solution.
        return [] 

    def forward_kinematics(self, angles: Tuple) -> Tuple:
        return self.chain.forward_kinematics(angles)
    
    def get_end_effector_position(self):
        current_joint_values = np.array(np.zeros(8))
        current_joint_values[1:] = self.controller.current_target_joint_values
        end_effector_position = self.chain.forward_kinematics(current_joint_values)[:3, 3]

        return end_effector_position

    def move_end_effector(self, target: list, max_steps: int = 10000, tolerance: float = 0.0, sim=None, model=None, viewer=None, render=True) -> Tuple:
        target_joint_values = self.inverse_kinematics(
            target, sim=sim, model=model)
        if len(target_joint_values) == 0:
            return False, False

        # Move arm only
        ids = [0, 1, 2, 3, 4]

        reached_target = False
        sim_data = sim.data
        actuators = self.controller.actuators
        success = True
        max_steps_reached = False

        deltas = np.zeros(len(sim_data.ctrl))
        steps = 1

        for i, v in enumerate(ids):
            self.controller.current_target_joint_values[v] = target_joint_values[i]

        for j in range(len(sim_data.ctrl)):
            actuators[j][3].setpoint = self.controller.current_target_joint_values[j]

        actuated_joint_ids = np.array([actuators[i][1] for i in actuators])

        while not reached_target:
            current_joint_values = sim_data.qpos[actuated_joint_ids]
            for j in range(len(sim_data.ctrl)):
                sim_data.ctrl[j] = actuators[j][3](current_joint_values[j])

            for i in ids:
                deltas[i] = abs(
                    self.controller.current_target_joint_values[i] - current_joint_values[i])

            if max(deltas) < tolerance:
                success = True
                reached_target = True
            elif steps > max_steps:
                success = False
                max_steps_reached = True
                break

            steps += 1
            sim.step()
            if render:
                viewer.render()

        return success, max_steps_reached

    def toggle_gripper(self, open: bool = True, max_steps: int = 300, half: bool = False, tolerance: float = 0.05, sim=None, viewer=None, render=True) -> Tuple[bool, bool]:
        """
        Open or close the gripper
        """
        id = 6  # Move gripper only

        if half:
            target = 0.0
        else:
            target = 0.4 if open else -0.4

        reached_target = False
        sim_data = sim.data
        actuators = self.controller.actuators
        success = True
        max_steps_reached = False

        deltas = np.zeros(len(sim_data.ctrl))

        steps = 1

        self.controller.current_target_joint_values[id] = target

        for j in range(len(sim_data.ctrl)):
            actuators[j][3].setpoint = self.controller.current_target_joint_values[j]

        actuated_joint_ids = np.array([actuators[i][1] for i in actuators])

        while not reached_target:
            current_joint_values = sim_data.qpos[actuated_joint_ids]
            for j in range(len(sim_data.ctrl)):
                sim_data.ctrl[j] = actuators[j][3](current_joint_values[j])

            deltas[id] = abs(
                self.controller.current_target_joint_values[id] - current_joint_values[id])

            if max(deltas) < tolerance:
                success = True
                reached_target = True
            elif steps > max_steps:
                success = False
                max_steps_reached = True
                break

            steps += 1
            sim.step()
            if render:
                viewer.render()

        return success, max_steps_reached

    def grasp(self, sim, viewer, render) -> bool:
        success, _ = self.toggle_gripper(
            open=False, max_steps=300, sim=sim, viewer=viewer, render=render)
        return success == False

    def is_object_in_gripper(self, sim, viewer, render):
        _, max_steps_reached = self.toggle_gripper(
            open=False, max_steps=1000, tolerance=0.01, sim=sim, viewer=viewer, render=render)
        return max_steps_reached

    def rotate_wrist_3_joint(self, degrees, max_steps: int = 500, tolerance: float = 0.05, sim=None, viewer=None, render=True) -> Tuple[bool, bool]:
        self.controller.current_target_joint_values[5] = math.radians(degrees)
        # Move arm only

        reached_target = False
        sim_data = sim.data
        actuators = self.controller.actuators
        success = True

        ids = list(range(len(sim_data.ctrl)))

        deltas = np.zeros(len(sim_data.ctrl))

        steps = 1

        for j in range(len(sim_data.ctrl)):
            actuators[j][3].setpoint = self.controller.current_target_joint_values[j]

        actuated_joint_ids = np.array([actuators[i][1] for i in actuators])

        while not reached_target:
            current_joint_values = sim_data.qpos[actuated_joint_ids]
            for j in range(len(sim_data.ctrl)):
                sim_data.ctrl[j] = actuators[j][3](current_joint_values[j])

            for i in ids:
                deltas[i] = abs(
                    self.controller.current_target_joint_values[i] - current_joint_values[i])

            if max(deltas) < tolerance:
                success = True
                reached_target = True
            elif steps > max_steps:
                success = False
                break

            steps += 1
            sim.step()
            if render:
                viewer.render()

        return success, max_steps <= steps
