# With this sample program, you can connect to a camera and obtain the 2D image, depth map and point
# cloud data.

import sys, os

from MechEye import Device
from mecheye_python_samples.source import Common
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


class ConnectAndCaptureImage(object):
    def __init__(self):
        self.device = Device()

    def connect_and_capture(self):
        device_intrinsic = self.device.get_device_intrinsic()
        Common.print_dist_coeffs("CameraDistCoeffs", device_intrinsic.texture_camera_intrinsic())
        Common.print_matrix("CameraMatrix", device_intrinsic.texture_camera_intrinsic())

        Common.print_dist_coeffs("DepthDistCoeffs", device_intrinsic.depth_camera_intrinsic())
        Common.print_matrix("DepthMatrix", device_intrinsic.depth_camera_intrinsic())

        row, col = 222, 222
        color_map = self.device.capture_color()
        print("Color map size is width: {}, height: {}".format(color_map.width(), color_map.height()))

        color_data = color_map.data()
        RGB = [color_data[int(row)][int(col)][i] for i in range(3)]
        print("Color map element at ({},{}) is R:{},G:{},B{}\n".
              format(row, col, RGB[0], RGB[1], RGB[2]))

        depth_map = self.device.capture_depth()
        print("Depth map size is width: {}, height: {}".format(depth_map.width(), depth_map.height()))
        print("Depth map element at ({},{}) is depth :{}mm\n".
              format(row, col, depth_map.data()[int(row)][int(col)]))

        point_xyz_map = self.device.capture_point_xyz()
        print("PointXYZ map size is width: {}, height: {}".format(point_xyz_map.width(), point_xyz_map.height()))

        point_xyz_data = point_xyz_map.data()
        print("PointXYZ map element at ({},{}) is X: {}mm , Y: {}mm, Z: {}mm\n".
              format(row, col, point_xyz_data[int(row)][int(col)][0],
                     point_xyz_data[int(row)][int(col)][1],
                     point_xyz_data[int(row)][int(col)][2]))

        self.device.disconnect()
        print("Disconnected from the Mech-Eye device successfully.")

    def main(self):
        Common.find_camera_list(self)
        if Common.choose_camera_and_connect(self):
            self.connect_and_capture()


if __name__ == '__main__':
    a = ConnectAndCaptureImage()
    a.main()
