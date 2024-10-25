# With this sample program, you can obtain and save 2D images, depth maps and point clouds
# simultaneously from multiple cameras.

import sys, os

import threading
import time

import numpy as np
import open3d as o3d

import cv2
from MechEye import Device
from mecheye_python_samples.source import Common

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(BASE_DIR)
# print(BASE_DIR)

save_dir_root_path = os.getcwd()
save_dir = save_dir_root_path + "/data/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


class CaptureThread (threading.Thread):
    def __init__(self, device):
        threading.Thread.__init__(self)
        self.device = device

    def run(self):
        device_info = self.device.get_device_info()
        t_name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        print("Camera {} start capturing.".format(device_info.id))
        # ColorMap
        color_map = self.device.capture_color()
        color_file = save_dir + t_name + "ColorMap.png"
        cv2.imencode('.png', color_map.data())[1].tofile(color_file)

        # DepthMap
        depth_map = self.device.capture_depth()
        depth_file = save_dir + t_name + "DepthMap.tiff"
        cv2.imencode('.tiff', depth_map.data())[1].tofile(depth_file)

        # PointCloudXYZ
        points_xyz = self.device.capture_point_xyz()
        points_xyz_data = points_xyz.data()
        points_xyz_o3d = o3d.geometry.PointCloud()
        points_xyz_o3d.points = o3d.utility.Vector3dVector(points_xyz_data.reshape(-1, 3) * 0.001)

        o3d.io.write_point_cloud(save_dir + t_name + "PointCloudXYZ.ply", points_xyz_o3d)

        # PointCloudXYZRGB
        points_xyz_bgr = self.device.capture_point_xyz_bgr().data()
        points_reshape = points_xyz_bgr.reshape(-1, 6)
        points_xyz_rgb_points = points_reshape[:, :3] * 0.001
        point_xyz_rgb_colors = points_reshape[:, 3:6][:, ::-1] / 255
        points_xyz_rgb_o3d = o3d.geometry.PointCloud()
        points_xyz_rgb_o3d.points = o3d.utility.Vector3dVector(points_xyz_rgb_points.astype(np.float64))
        points_xyz_rgb_o3d.colors = o3d.utility.Vector3dVector(point_xyz_rgb_colors.astype(np.float64))
        o3d.io.write_point_cloud(save_dir + t_name + "PointCloudXYZRGB.ply", points_xyz_rgb_o3d)

        self.device.disconnect()
        print("Disconnected from the Mech-Eye device successfully.")


class CaptureSimultaneouslyMultiCamera(object):
    def __init__(self):
        self.device = Device()

    def connect_device_and_capture(self):
        devices = []
        for index in self.indices:
            device = Device()
            error_status = device.connect(self.device_list[index])
            if not error_status.ok():
                print(error_status.description())
                quit()
            devices.append(device)
            print(111111111)
            print(device)
            print(devices)

        for device in devices:
            capture_thread = CaptureThread(device)
            capture_thread.start()
            capture_thread.join()

    def main(self):
        Common.find_camera_list(self)
        Common.choose_multi_camera(self)
        if len(self.indices) != 0:
            self.connect_device_and_capture()
        else:
            print("No camera was selected.")


if __name__ == '__main__':
    a = CaptureSimultaneouslyMultiCamera()
    a.main()
