# With this sample program, you can set specified parameters to a camera.

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from MechEye import Device
from mecheye_python_samples.source import Common


class GetAndSetParameter(object):
    def __init__(self):
        self.device = Device()

    def set_parameters(self):
        print("All user sets : ", end='')
        user_sets = self.device.get_all_user_sets()
        for user_set in user_sets:
            print(user_set, end=' ')

        current_user_set = self.device.get_current_user_set()
        print("\ncurrent_user_set: " + str(current_user_set))

        Common.show_error(self.device.set_scan_3d_exposure([1.0, 32.1, 99.0]))
        exposure_sequence = self.device.get_scan_3d_exposure()
        print("\nThe 3D scanning exposure multiplier: {}".format(
            len(exposure_sequence)))
        for i in exposure_sequence:
            print("3D scanning exposure time: {}".format(i))

        Common.show_error(self.device.set_depth_range(1, 2))
        depth_range = self.device.get_depth_range()
        print("\n3D scanning depth Lower Limit: {} mm,".format(depth_range.lower()),
              "depth upper limit: {} mm\n".format(depth_range.upper()))

        self.device.set_scan_3d_roi(20, 20, 1000, 1000)
        scan_3d_roi = self.device.get_scan_3d_roi()
        print("3D scanning ROI topLeftX: {}, topLeftY: {}, width: {}, height: {}\n".
              format(scan_3d_roi.x(), scan_3d_roi.y(), scan_3d_roi.width(), scan_3d_roi.height()))

        Common.show_error(self.device.set_scan_2d_exposure_mode("Auto"))
        Common.show_error(self.device.set_scan_2d_exposure_time(999.0))
        exposure_mode_2d = self.device.get_scan_2d_exposure_mode()
        scan_2d_exposure_time = self.device.get_scan_2d_exposure_time()
        print("2D scanning exposure mode enum: {}, exposure time: {}\n".
              format(exposure_mode_2d, scan_2d_exposure_time))

        Common.show_error(self.device.set_cloud_surface_smoothing_mode("Normal"))
        Common.show_error(self.device.set_cloud_noise_removal_mode("Normal"))
        Common.show_error(self.device.set_cloud_outlier_removal_mode("Normal"))
        Common.show_error(self.device.set_cloud_edge_preservation_mode("Normal"))

        cloud_surface_smoothing_mode = self.device.get_cloud_surface_smoothing_mode()
        cloud_noise_removal_mode = self.device.get_cloud_noise_removal_mode()
        cloud_outlier_removal_mode = self.device.get_cloud_outlier_removal_mode()
        cloud_edge_preservation_mode = self.device.get_cloud_edge_preservation_mode()

        print("Cloud surface smoothing mode enum : {}, cloud edge preservation mode enum : {}.".
              format(cloud_surface_smoothing_mode, cloud_edge_preservation_mode))
        print("Cloud noise removal mode enum : {}, cloud outlier removal mode enum :{}.".
                    format(cloud_noise_removal_mode, cloud_outlier_removal_mode))

        Common.show_error(self.device.add_user_set("iii"))
        Common.show_error(self.device.delete_user_set('iii'))

        # Parameter of laser camera, please comment out when connecting non-laser camera.
        Common.show_error(self.device.set_laser_settings("Accurate", 2, 50, 4, 80))
        laser_settings = self.device.get_laser_settings()
        print("\nlaser_mode: {}, range_start:{}, range_end:{}, partition_count:{}, power_level:{}".
              format(laser_settings.fringe_coding_mode(), laser_settings.frame_range_start(), laser_settings.frame_range_end(),
                     laser_settings.frame_partition_count(), laser_settings.power_level()))

        self.device.save_all_settings_to_user_set()

        self.device.disconnect()
        print("Disconnect Mech-Eye Success.")

    def main(self):
        Common.find_camera_list(self)
        if Common.choose_camera_and_connect(self):
            self.set_parameters()


if __name__ == '__main__':
    a = GetAndSetParameter()
    a.main()
