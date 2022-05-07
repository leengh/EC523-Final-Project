from gym.envs.registration import register

register(
    id="Pick_and_Place-v0",
    entry_point="gym_pick_and_place.envs:GraspingEnvironment",
    kwargs={'model_type': 'binary', 'train': False} 
)
