EC523 Final Project - Pick & Place using Deep Reinforcement learning

We re implemented https://github.com/PaulDanielML/MuJoCo_RL_UR5 

The Modules.py file, transform_observation and transform_action functions are taken from that project (with some modifications). Everything else was re-written. This project only uses binary rewards, we testing the model with 3 different rewards: Categorical, Euclidean and Manhattan.

## Installation

1- Install mujoco 210 and setup the license.

2- Run 'pip install -r requirements.txt' to install the required packages

## Train

Run the train.py file to train the model. This by default trains the model with binary rewards.

If you want to use another reward, you can uncomment the line that initializes the model with that reward and run train.py again.


## Test

Run the test.py file to test the model. This by default test the model with binary rewards.

If you want to use another reward, you can uncomment the line that initializes the model with that reward and run test.py again.


