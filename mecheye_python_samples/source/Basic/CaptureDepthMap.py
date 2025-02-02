# With this sample program, you can obtain and save the depth map in OpenCV format from a camera.

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from MechEye import Device
import cv2
from mecheye_python_samples.source import Common


class CaptureDepthMap(object):
    def __init__(self):
        self.device = Device()

    def capture_depth_map(self):
        depth_map = self.device.capture_depth()
        depth_file = "DepthMap.tiff"
        cv2.imencode('.tiff', depth_map.data())[1].tofile(depth_file)
        print("Capture and save depth image : {}".format(depth_file))

        self.device.disconnect()
        print("Disconnect from the Mech-Eye device successfully.")

    def main(self):
        Common.find_camera_list(self)
        if Common.choose_camera_and_connect(self):
            self.capture_depth_map()


if __name__ == '__main__':
    a = CaptureDepthMap()
    a.main()
