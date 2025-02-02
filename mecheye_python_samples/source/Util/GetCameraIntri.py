# With this sample program, you can get and print a camera's intrinsic parameters.

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from MechEye import Device
from mecheye_python_samples.source import Common


class GetCameraIntri(object):
    def __init__(self):
        self.device = Device()

    def get_device_intrinsic(self):
        device_intrinsic = self.device.get_device_intrinsic()
        Common.print_dist_coeffs("CameraDistCoeffs", device_intrinsic.texture_camera_intrinsic())
        Common.print_dist_coeffs("DepthDistCoeffs", device_intrinsic.depth_camera_intrinsic())
        Common.print_matrix("CameraMatrix", device_intrinsic.texture_camera_intrinsic())
        Common.print_matrix("DepthMatrix", device_intrinsic.depth_camera_intrinsic())

        self.device.disconnect()
        print("Disconnected from the Mech-Eye device successfully.")

    def main(self):
        Common.find_camera_list(self)
        if Common.choose_camera_and_connect(self):
            self.get_device_intrinsic()


if __name__ == '__main__':
    a = GetCameraIntri()
    a.main()
