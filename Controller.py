from typing import List, Set
from simple_pid import PID
import numpy as np



class Controller():
    def __get_PID_controllers(self) -> Set:
        controllers = {}
        # Values for training
        sample_time = 0.0001
        p_scale = 3
        i_scale = 0.0
        i_gripper = 0
        d_scale = 0.1

        # Shoulder Pan Joint
        controllers[0] = PID(
            7 * p_scale,
            0.0 * i_scale,
            1.1 * d_scale,
            setpoint=0,
            output_limits=(-2, 2),
            sample_time=sample_time,
        )

        # Shoulder Lift Joint
        controllers[1] = PID(
            10 * p_scale,
            0.0 * i_scale,
            1.0 * d_scale,
            setpoint=-1.57,
            output_limits=(-2, 2),
            sample_time=sample_time,
        )

        # Elbow Joint
        controllers[2] = PID(
            5 * p_scale,
            0.0 * i_scale,
            0.5 * d_scale,
            setpoint=1.57,
            output_limits=(-2, 2),
            sample_time=sample_time,
        )

        # Wrist 1 Joint
        controllers[3] = PID(
            7 * p_scale,
            0.0 * i_scale,
            0.1 * d_scale,
            setpoint=-1.57,
            output_limits=(-1, 1),
            sample_time=sample_time,
        )

        # Wrist 2 Joint
        controllers[4] = PID(
            5 * p_scale,
            0.0 * i_scale,
            0.1 * d_scale,
            setpoint=-1.57,
            output_limits=(-1, 1),
            sample_time=sample_time,
        )

        # Wrist 3 Joint
        controllers[5] = PID(
            5 * p_scale,
            0.0 * i_scale,
            0.1 * d_scale,
            setpoint=0.0,
            output_limits=(-1, 1),
            sample_time=sample_time,
        )

        # Gripper Joint
        controllers[6] = PID(
            2.5 * p_scale,
            i_gripper,
            0.00 * d_scale,
            setpoint=0.0,
            output_limits=(-1, 1),
            sample_time=sample_time,
        )

        return controllers

    def set_actuators(self, sim, model) -> None:
        actuators = {}
        pid_controllers = self.__get_PID_controllers()
        current_target_joint_values = [
            pid_controllers[i].setpoint for i in range(len(sim.data.ctrl))
        ]
        self.current_target_joint_values = np.array(
            current_target_joint_values)

        for i in range(len(sim.data.ctrl)):
            actuator_trnid = model.actuator_trnid[i][0]
            actuators[i] = [model.actuator_id2name(
                i), actuator_trnid, model.joint_id2name(actuator_trnid), pid_controllers[i]]
        self.actuators = actuators

    def get_actuated_joint_ids(self) -> List:
        # try np.array
        return [self.actuators[key][1] for key in self.actuators.keys()]

    
