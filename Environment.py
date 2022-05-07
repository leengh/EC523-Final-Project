class Environment():
    def __init__(self) -> None:
        self.goal_1 = [0.6, -0.1, 1.15]
        self.goal_2 = [0, 0.6, 1.15]
        self.initial_position = [0.0, -0.6, 1.1]
        self.TABLE_HEIGHT = 0.91

 
    def convert_depth_to_meters(self, model, depth: list) -> list:
        extend = model.stat.extent
        near = model.vis.map.znear * extend
        far = model.vis.map.zfar * extend
        return near / (1 - depth * (1 - near / far))

   
