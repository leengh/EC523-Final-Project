import numpy as np
import cv2
import copy


class Camera():
    def initialize_camera(self, model: None = None, width=200, height=200, camera: str = "top_down") -> None:
        cam_id = model.camera_name2id(camera)
        fovy = model.cam_fovy[cam_id]
        f = 0.5 * height / np.tan(fovy * np.pi / 360)
        self.cam_matrix = np.array(
            ((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
        self.cam_rot_mat = model.cam_mat0[cam_id]
        self.cam_rot_mat = np.reshape(self.cam_rot_mat, (3, 3))
        self.cam_pos = model.cam_pos0[cam_id]

    def convert_pixel_to_world_coordinates(self, pixel_x, pixel_y, depth, width=200, height=200, camera="top_down"):
        # Create coordinate vector
        pixel_coord = np.array([pixel_x, pixel_y, 1]) * (-depth)
        # Get position relative to camera
        pos_c = np.linalg.inv(self.cam_matrix) @ pixel_coord
        # Get world position
        pos_w = np.linalg.inv(self.cam_rot_mat) @ (pos_c + self.cam_pos)

        return pos_w

    def capture_photo(self, camera: str = "top_down", sim: None = None, width: int = 200, height: int = 200) -> object:
        rgb, depth = copy.deepcopy(sim.render(
            camera_name=camera, width=width, height=height, depth=True)
        )
        rgb = np.fliplr(np.flipud(rgb))
        depth = np.fliplr(np.flipud(depth))

        return np.array(rgb), np.array(depth)

    def get_object_shape(self, img) -> str:
        img = img[0:250, 400:700]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 127, 255, 1)

        contours, _ = cv2.findContours(thresh, 1, 2)

        options = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            if len(approx) == 5:
                options.append("pentagon")
                cv2.drawContours(img, [cnt], 0, 255, -1)
            elif len(approx) == 3:
                options.append("triangle")
                cv2.drawContours(img, [cnt], 0, (0, 255, 0), -1)
            elif len(approx) == 4:
                options.append("square")
                cv2.drawContours(img, [cnt], 0, (0, 0, 255), -1)
            elif len(approx) == 9:
                options.append("half-circle")
                cv2.drawContours(img, [cnt], 0, (255, 255, 0), -1)
            elif len(approx) > 15:
                options.append("circle")
                cv2.drawContours(img, [cnt], 0, (0, 255, 255), -1)
        if "circle" in options:
            return "circle"
        else:
            return "square"
