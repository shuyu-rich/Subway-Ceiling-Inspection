#


import os
import sys
import threading
import time

import cv2
import numpy as np
import open3d as o3d
from MechEye import Device

from mecheye_python_samples.source import Common

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

save_dir_root_path = os.getcwd()
save_dir = save_dir_root_path + "/data/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


class CaptureAllData(object):
    def __init__(self):
        self.device = Device()

    def capture_color_map(self):
        color_map = self.device.capture_color()
        color_file = save_dir + t_name + "ColorMap.png"
        print(color_map)
        print(color_map.data())

        cv2.imencode('.png', color_map.data())[1].tofile(color_file)
        print("Capture and save color image : {}".format(color_file))

        # self.device.disconnect()
        print("Disconnected from the Mech-Eye device successfully.")

    def capture_depth_map(self):
        depth_map = self.device.capture_depth()
        depth_file = save_dir + t_name + "DepthMap.tiff"
        cv2.imencode('.tiff', depth_map.data())[1].tofile(depth_file)
        print("Capture and save depth image : {}".format(depth_file))

        # self.device.disconnect()
        print("Disconnect from the Mech-Eye device successfully.")

    def capture_point_cloud(self):
        points_xyz = self.device.capture_point_xyz()
        points_xyz_data = points_xyz.data()
        points_xyz_o3d = o3d.geometry.PointCloud()
        points_xyz_o3d.points = o3d.utility.Vector3dVector(points_xyz_data.reshape(-1, 3) * 0.001)
        # 下面注释了可视化代码
        # o3d.visualization.draw_geometries([points_xyz_o3d])
        o3d.io.write_point_cloud(save_dir + t_name + "PointCloudXYZ.ply", points_xyz_o3d)
        print("Point cloud saved to path" + save_dir + t_name + "PointCloudXYZ.ply")

    def capture_color_point_cloud(self):
        points_xyz_bgr = self.device.capture_point_xyz_bgr().data()

        points_reshape = points_xyz_bgr.reshape(-1, 6)
        points_xyz_rgb_points = points_reshape[:, :3] * 0.001
        point_xyz_rgb_colors = points_reshape[:, 3:6][:, ::-1] / 255

        points_xyz_rgb_o3d = o3d.geometry.PointCloud()
        points_xyz_rgb_o3d.points = o3d.utility.Vector3dVector(points_xyz_rgb_points.astype(np.float64))
        points_xyz_rgb_o3d.colors = o3d.utility.Vector3dVector(point_xyz_rgb_colors.astype(np.float64))
        # 下面注释了可视化代码
        # o3d.visualization.draw_geometries([points_xyz_rgb_o3d])
        o3d.io.write_point_cloud(save_dir + t_name + "PointCloudXYZRGB.ply", points_xyz_rgb_o3d)
        print("Color point cloud saved to path" + save_dir + t_name + "PointCloudXYZRGB.ply")

    def main(self):
        Common.find_camera_list(self)
        if Common.choose_camera_and_connect(self):
            self.capture_color_map()
            # self.capture_depth_map()
            # self.capture_point_cloud()
            # self.capture_color_point_cloud()
            self.device.disconnect()
            print("Disconnected from the Mech-Eye device successfully.")


if __name__ == '__main__':
    a = CaptureAllData()
    num_collect = 0
    while True:
        user_input = input("是否采集（输入任意数字采集，输入其他内容则退出）：")
        if user_input.isdigit():
            num_collect = num_collect + 1
            t_name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
            a.main()
            print("采集一次")
            print("第%d次数据采集" % num_collect)
        else:
            break
