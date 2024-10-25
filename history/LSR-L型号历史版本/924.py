import json
import shutil
import struct
import sys
import zipfile

import cv2
import time
import sqlite3
import os
import pygame
import serial.tools.list_ports
import re

from datetime import datetime
from zipfile import ZipFile
# import qdarkstyle
#
# from qt_material import apply_stylesheet
from paddleocr import PaddleOCR, draw_ocr
from io import BytesIO
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QComboBox, QDialog, QMessageBox, QSpacerItem, QSizePolicy, QTextEdit, QLineEdit, QSlider,
    QCheckBox, QStyledItemDelegate, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFileDialog
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QCoreApplication, QDateTime, QObject
from PySide6.QtGui import QPixmap, QImage, QColor, QFont, QPalette
import numpy as np
from MechEye import Device
from dash.html import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from mapping2depth import *
from mecheye_python_samples.source import Common
from now_demo_loop import CaptureAllData
import open3d as o3d
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from openpyxl import Workbook, load_workbook
from collections import defaultdict

# 创建全局变量
global_lc_lf_conditions = []
# 创建全局变量
global_lc_cf_conditions = []


class DraculaColors:
    background = "#282a36"
    current_line = "#44475a"
    selection = "#44475a"
    foreground = "#f8f8f2"
    comment = "#6272a4"
    cyan = "#8be9fd"
    green = "#50fa7b"
    orange = "#ffb86c"
    pink = "#DCDCDC"
    purple = "#bd93f9"
    red = "#ff5555"
    yellow = "#1E90FF"
    blue = "＃0000FF "
    white = "#FFFFFF"


# 自定义委托类用于绘制下拉列表项
class ComboBoxDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        if index.data(Qt.UserRole):
            pixmap = QPixmap(index.data(Qt.UserRole))
            painter.drawPixmap(option.rect.x(), option.rect.y(), pixmap)
        else:
            super().paint(painter, option, index)


class CustomComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setItemDelegate(ComboBoxDelegate())


class YourClass(QObject):
    relay1_state_changed_signal = Signal(bool)
    relay2_state_changed_signal = Signal(bool)
    relay3_state_changed_signal = Signal(bool)
    relay4_state_changed_signal = Signal(bool)

    def __init__(self):
        super().__init__()
        self.connected = False  # 初始化连接状态为未连接
        self.ser = None
        self.connect_to_serial("COM5")  # 自动连接到指定的串口

    def connect_to_serial(self, selected_port):
        try:
            self.ser = serial.Serial(selected_port, 9600, timeout=0.5)
            print(f"已连接至串口 {self.ser.port}, 波特率 9600")
            self.connected = True
        except Exception as e:
            print(f"连接串口 {selected_port} 时发生错误: {e}")
            self.connected = False

    def send_command(self, relay_number, action_code):
        if not self.connected:
            print("串口未连接")
            return

        # 计算继电器的起始地址
        coil_address = relay_number - 1
        # 构造指令
        command = struct.pack(">BBHH", 1, 0x05, coil_address, action_code)
        crc = self.calculate_crc(command)
        command += struct.pack("<H", crc)

        try:
            self.ser.write(command)
            print(f"已发送指令: {command.hex()}")
        except Exception as e:
            print(f"发送指令时发生错误: {e}")

    @staticmethod
    def calculate_crc(data):
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        return crc

    def relay1_state_changed(self, state):
        print("Relay 1 State Changed:", state)
        self.send_command(1, 0xFF00 if state else 0x0000)

    def relay2_state_changed(self, state):
        print("Relay 2 State Changed:", state)
        self.send_command(2, 0xFF00 if state else 0x0000)

    def relay3_state_changed(self, state):
        print("Relay 3 State Changed:", state)
        self.send_command(3, 0xFF00 if state else 0x0000)

    def relay4_state_changed(self, state):
        print("Relay 4 State Changed:", state)
        self.send_command(4, 0xFF00 if state else 0x0000)


class ImageCaptureThread(QThread):
    stopCapture = Signal()
    resumeCapture = Signal()
    imagesCaptured = Signal(QImage, QImage)
    imagesCapturedAndSaved = Signal(str, str)
    analysisCompleted = Signal()
    detectionCompleted = Signal(object)
    depthvalue = Signal(str)
    depthvalue1 = Signal(str)
    imageCaptured = Signal(QImage, QImage)
    # 新增信号定义，用于传递LC最大值
    lcMaxUpdated = Signal(str)
    alarmStatusChanged = Signal(bool)

    def __init__(self, device, parent_window):

        super().__init__()
        self.device = device
        self.parent_window = parent_window
        self.is_running = True  # 标志来控制线程的运行状态
        self.model_cache = {}  # 字典用于存储加载的模型
        self.selected_model = None
        self.selected_model = "错缝"
        pygame.mixer.init()
        self.processing_functions = {
            '补丁铝板': self.process_common_model,
            '铝板': self.process_common_model,
            '栅格': self.process_grille,
            '反光冲孔板': self.process_common_model,
            '冲孔板': self.process_common_model,
            '拉网': self.process_common_model,
            '铝条板': self.process_common_model,
            '栅格加网': self.process_common_model,
            '铝方通': self.process_aluminum_square_tube,
            '矿棉板': self.process_common_model,
            '勾搭铝板': self.process_common_model,

            '垂片': self.process_aluminum_square_tube,
            '高铝板': self.process_common_model,
        }
        self.detected_labels = []
        # self.capture_interval = 10000
        self.imageCaptured.connect(self.on_images_captured)

    def run(self):
        timer = QTimer()
        timer.timeout.connect(self.capture_images)
        timer.start(self.capture_interval)  # 8000 毫秒（8秒）触发一次
        self.stopCapture.connect(lambda: timer.stop())
        self.resumeCapture.connect(lambda: timer.start(self.capture_interval))

        while self.is_running:
            QCoreApplication.processEvents()  # 处理事件，确保界面响应
            time.sleep(0.1)  # 避免CPU占用过高
        timer.stop()

    def set_capture_interval(self, interval):
        self.capture_interval = interval

    def set_selected_model(self, model_name):
        self.selected_model = model_name
        if model_name not in self.model_cache:
            self.model_cache[model_name] = self.load_model(model_name)

    def load_model(self, model_name):
        # 小电脑模型路径
        model_path = 'D:\PythonCode\yolov5'
        weight_paths = {
            '补丁铝板': 'D:\\PythonCode\\yolov5\\runs\\train\\all\\716fanglvban.pt',
            '铝板': 'D:\\PythonCode\\yolov5\\runs\\train\\all\\902fanglvban.pt',
            '反光冲孔板': 'D:/PythonCode/yolov5/runs/train/all/523fanguang.pt',
            '冲孔板': 'D:/PythonCode/yolov5/runs/train/all/620ck.pt',
            '拉网': 'D:/PythonCode/yolov5/runs/train/all/506heibai.pt',
            '铝条板': 'D:/PythonCode/yolov5/runs/train/all/902ltb.pt',
            '栅格加网': 'D:/PythonCode/yolov5/runs/train/all/shangejiawang.pt',
            '矿棉板': 'D:/PythonCode/yolov5/runs/train/all/kuangmian.pt',
            '栅格': 'D:/PythonCode/yolov5/runs/train/all/717lvfangtong.pt',
            '铝方通': 'D:/PythonCode/yolov5/runs/train/all/830lvfangtong.pt',
            '勾搭铝板': 'D:/PythonCode/yolov5/runs/train/all/830gouda.pt',
            '垂片': 'D:/PythonCode/yolov5/runs/train/all/511lvfangtong.pt',
            '高铝板': 'D:/PythonCode/yolov5/runs/train/all/910gaolvban.pt'
        }
        weight_path = weight_paths.get(model_name, 'D:\PythonCode\yolov5/runs/train/all')
        return torch.hub.load(model_path, 'custom', weight_path, source='local')

    def capture_images(self):
        if not self.is_running:
            return
        color_image = self.device.capture_color().data()
        depth_image = self.device.capture_depth().data()
        self.depth_image = depth_image

        self.height, self.width = color_image.shape[:2]
        if self.height != 1500 or self.width != 2000:
            print("change size!")
            color_image = cv2.resize(color_image, (2000, 1500), interpolation=cv2.INTER_AREA)
            depth_image = cv2.copyMakeBorder(depth_image, 70, 200, 90, 200, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            depth_image = cv2.resize(depth_image, (2048, 1536), interpolation=cv2.INTER_AREA)
            self.depth_image = depth_image
        color_image_width, color_image_height = color_image.shape[1], color_image.shape[0]
        depth_image_width, depth_image_height = depth_image.shape[1], depth_image.shape[0]

        print(f"Color Image Resolution: {color_image_width} x {color_image_height}")
        print(f"Depth Image Resolution: {depth_image_width} x {depth_image_height}")

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        self.depth_img = QImage(depth_colormap.data, depth_colormap.shape[1], depth_colormap.shape[0],
                                QImage.Format_RGB888)
        self.color_img = QImage(color_image.data, color_image.shape[1], color_image.shape[0], QImage.Format_RGB888)
        self.current_color_image = self.color_img
        self.current_depth_image = self.depth_img
        self.imagesCaptured.emit(self.color_img, self.depth_img)
        self.imagesCapturedAndSaved.emit(self.color_img, self.depth_img)

        if self.selected_model in self.processing_functions:
            # detection_results = self.processing_functions[self.selected_model](color_image)
            # self.detectionCompleted.emit(detection_results)
            # self.imageCaptured.emit(self.color_img, self.depth_img)

            print("烦死了"*100)
            detection_results, self.tankuang = self.processing_functions[self.selected_model](color_image)

            # 发送检测结果信号
            self.detectionCompleted.emit(detection_results)
            if self.tankuang:
                self.imageCaptured.emit(self.color_img, self.depth_img)

    def calculate_vertical_distance(self, depth_value1, depth_value2):
        # 深度值以毫米为单位，直接相减得到垂直距离
        vertical_distance = abs(depth_value2 - depth_value1)
        return vertical_distance



    def process_common_model(self, color_frame):
        color_frame1 = color_frame
        # 旋转图像180°
        (h, w) = color_frame.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        color_frame = cv2.warpAffine(color_frame, M, (w, h))

        self.tankuang = False
        # 引用全局变量
        global global_lc_lf_conditions , depth_value_at_x2_y2, depth_value_at_x1_y1, roy1
        global_lc_lf_conditions = []
        self.line_threshold = 67  # 基于您的图像尺寸和视野，您可能需要调整这个值
        self.column_threshold = 60
        # QTimer.singleShot(500, lambda: self.detectionCompleted.emit(detection_img))
        self.distance_to_object_data = {
            1200: (1200, 1200),  # 相机距离拍摄物1.2米，视野宽度1.2米，视野高度1米
            1300: (1300, 1070),  # 相机距离拍摄物1.3米，视野宽度1.3米，视野高度1.07米
            1400: (1400, 1130),  # 相机距离拍摄物1.4米，视野宽度1.4米，视野高度1.13米
            1500: (1500, 1200),  # 相机距离拍摄物1.5米，视野宽度1.5米，视野高度1.2米
            1600: (1600, 1280),  # 相机距离拍摄物1.6米，视野宽度1.6米，视野高度1.28米
            1700: (1700, 1360),  # 相机距离拍摄物1.7米，视野宽度1.7米，视野高度1.36米
            1800: (1800, 1440),  # 相机距离拍摄物1.8米，视野宽度1.8米，视野高度1.44米
            1900: (1900, 1520),  # 相机距离拍摄物1.9米，视野宽度1.9米，视野高度1.52米
            2000: (2000, 1600),  # 相机距离拍摄物2.0米，视野宽度2.0米，视野高度1.60米
            2100: (2100, 1680),  # 相机距离拍摄物2.1米，视野宽度2.1米，视野高度1.68米
            2200: (2200, 1760),  # 相机距离拍摄物2.2米，视野宽度2.2米，视野高度1.76米
            2300: (2300, 1840),  # 相机距离拍摄物2.3米，视野宽度2.3米，视野高度1.84米
            2400: (2400, 1920),  # 相机距离拍摄物2.4米，视野宽度2.4米，视野高度1.92米
            2500: (2500, 2000),  # 相机距离拍摄物2.5米，视野宽度2.5米，视野高度2.0米
            2600: (2600, 2080),  # 相机距离拍摄物2.6米，视野宽度2.6米，视野高度2.08米
            2700: (2700, 2160),  # 相机距离拍摄物2.7米，视野宽度2.7米，视野高度2.16米
            2800: (2800, 2240),  # 相机距离拍摄物2.8米，视野宽度2.8米，视野高度2.24米
            2900: (2900, 2320),  # 相机距离拍摄物2.9米，视野宽度2.9米，视野高度2.32米
            3000: (3000, 2400)  # 相机距离拍摄物3.0米，视野宽度3.0米，视野高度2.40米
        }
        if self.selected_model not in self.model_cache:
            # print(f"模型 {self.selected_model} 尚未加载")
            return



        model = self.model_cache[self.selected_model]
        # print("ModelName:", self.selected_model)
        results = model(color_frame1)  # 假设model是已加载并准备好的模型实例
        # ocr = PaddleOCR(use_angle_cls=True, lang="en")
        # img = color_frame
        # result = ocr.ocr(img, cls=True)
        #
        # if result is None:
        #     print("未检测到文本或OCR返回None")
        #     return
        #
        # # 迭代OCR结果
        # for res in result:
        #     if res is None:
        #         continue  # 如果某个特定结果是None，则跳过
        #     for line in res:
        #         print("文字识别")
        #         print(line, "\tresult:", line[1][0])

        # 获取彩色图像宽高、深度图宽高
        image_width, image_height = color_frame1.shape[1], color_frame1.shape[0]  # 彩色图像的宽度和高度
        # 深度图：depth_frame , RGB图：color_frame
        # depth_frame = self.device.capture_depth().data()
        depth_frame1 = self.depth_image
        depth_height, depth_width = 1405 - 50, 1877 - 80  # 深度图像的宽度和高度
        depth_frame = depth_frame1[50:1405, 80:1877]
        # 获取点云数据
        point_xyz_map = self.device.capture_point_xyz()
        point_xyz_data = point_xyz_map.data()
        # 计算斜率
        # 获取上、下、左、右距离
        v_top_list, v_left_list, v_bot_list, v_rig_list = [], [], [], []
        for i in range(60):
            for j in range(60):
                x, y, v_t = point_xyz_data[int(40 + i)][int(1024 + j)]
                if v_t > 50:
                    v_top_list.append(v_t)
                x, y, v_l = point_xyz_data[int(768 + i)][int(210 + j)]
                if v_l > 50:
                    v_left_list.append(v_l)
                x, y, v_d = point_xyz_data[int(1302 - i)][int(1024 - j)]
                if v_d > 50:
                    v_bot_list.append(v_d)
                x, y, v_r = point_xyz_data[int(768 - i)][int(1870 - j)]
                if v_r > 50:
                    v_rig_list.append(v_r)

        if len(v_top_list) < 1:
            print("上一条")
            for i in range(1, 2048):
                x, y, v_t = point_xyz_data[int(40)][int(i)]
                if v_t > 50:
                    v_top_list.append(v_t)
        if len(v_left_list) < 1:
            print("左一条")
            for i in range(1, 1536):
                x, y, v_l = point_xyz_data[int(i)][int(210)]
                if v_l > 50:
                    v_left_list.append(v_l)
        if len(v_bot_list) < 1:
            print("下一条")
            for i in range(1, 2048):
                x, y, v_t = point_xyz_data[int(1302)][int(i)]
                if v_t > 50:
                    v_top_list.append(v_t)
        if len(v_rig_list) < 1:
            print("右一条")
            for i in range(1, 1536):
                x, y, v_r = point_xyz_data[int(i)][int(1870)]
                if v_r > 50:
                    v_rig_list.append(v_r)

        v_top = np.mean(v_top_list)
        v_left = np.mean(v_left_list)
        v_bot = np.mean(v_bot_list)
        v_rig = np.mean(v_rig_list)
        print("上:", v_top, "左:", v_left, "下:", v_bot, "右:", v_rig)
        if v_top > v_bot:
            v_tb = True
            v_1 = v_top - v_bot
        else:
            v_tb = False
            v_1 = v_bot - v_top
        if v_left > v_rig:
            v_lr = True
            v_0 = v_left - v_rig
        else:
            v_lr = False
            v_0 = v_rig - v_left
        v_1mean = v_1 / 1292
        v_0mean = v_0 / 1690

        fov_width_pixels = 0  # 视野范围，不用动
        fov_height_pixels = 0

        max_vertical_distance = {}
        horizontal_difference = 0
        # 在帧上绘制检测结果
        # 获取要分析的图像内容
        self.label_stats = {}
        self.detected_labels = []
        res = results.xyxy[0]
        if self.selected_model in ["铝板", "补丁铝板"]:
            pass
            # 如果这个模型的检测结果小于10个，则和另一个模型进行合并检测

            if len(res) < 10:
                # 调用深色方吕版的模型
                b_path = "D:\\PythonCode\\yolov5\\runs\\train\\all\\shenselvban.pt"
                models1 = torch.hub.load('D:\PythonCode\yolov5', 'custom', path=b_path, source='local')
                results1 = models1(color_frame1)
                # print("浅色模型的结果", results.xyxy[0])
                # print("深色模型的结果", results1.xyxy[0])
                # 两个结果整合，使用torch.cat把两个检测结果的tens0r拼起来
                res = torch.cat((res, results1.xyxy[0]), dim=0)
                # NMS
                res = nms(res, 0.1)
                print("两个模型整合、去重后:\n", res)
                # 整合、去重操作后，检测框会丰富起来，再根据自身需求进行下一步操作
        for *xyxy, conf, cls in res:
            mids = []
            vertical_distances = []
            # rotated_image, rotated_boxes = rotate_image_and_boxes(color_frame, map(int, xyxy))
            x1, y1, x2, y2 = map(int, xyxy)

            if y2 - y1 <= 1 or x2 - x1 <= 1:
                # color_frame[y1:y2, x1:x2] = [255, 255, 255]
                # if y2 - y1 <= 1 or x2 - x1 <= 1:

                # print("检测到水平物体，跳过当前数据组")
                continue
            x_center = (x1 + x2) / 2  # 计算病害中心点的x坐标
            column_index = int(x_center / (image_width / 10))  # 计算列索引，这里将图像宽度等分为10份
            box_w, box_h = x2 - x1, y2 - y1
            if y1 < 50:
                y1 = 50
            if y2 < 50:
                y2 = 50
            # 更新标签统计字典
            label_name = results.names[int(cls)]
            if label_name not in self.label_stats:
                self.label_stats[label_name] = 1
            else:
                self.label_stats[label_name] += 1
            # 计算另外两个角的坐标
            x3, y3 = x1, y2  # 左下角
            x4, y4 = x2, y1  # 右上角
            # 调用映射坐标方法
            try:
                if self.selected_model == "勾搭铝板":
                    print("loaded_ModelName:", self.selected_model)
                    counts = gd_mapping2depth(x1, y1, x2, y2)
                else:
                    counts = mapping2depth(x1, y1, x2, y2)
            except:
                print("坐标框转换计算位置出错，跳过当前病害位置")
                continue

            for count in counts:
                try:
                    depth_x1, depth_y1, depth_x2, depth_y2, mean, shape = count
                    # cv2.rectangle(depth_frame, (depth_x1, depth_y1),
                    #               (depth_x2, depth_y2),
                    #               (0, 0, 255), 2)
                    # cv2.imshow("depth_frame", depth_frame)

                    # 调用映射坐标方法

                    # print(
                    #     f"Rectangular Coordinates: ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3}), ({x4}, {y4})")  # x1左上，x2右下，x3左下，x4右上
                    # 打印彩色图框位置信息
                    # print(f"Color Image Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
                    # 打印深度图框位置信息
                    # print(f"Depth Image Coordinates: ({depth_x1}, {depth_y1}) to ({depth_x2}, {depth_y2})")
                    # 判断框的位置信息是不是在图像内
                    if 0 <= depth_x1 < depth_width and 0 <= depth_y1 < depth_height and \
                            0 <= depth_x2 < depth_width and 0 <= depth_y2 < depth_height:

                        # if self.height != 1500 or self.width != 2000:
                        #     x, y, depth_value_at_x1_y1 = point_xyz_data[int(depth_y1 + 59)][int(depth_x1 + 96)]
                        #     x, y, depth_value_at_x2_y2 = point_xyz_data[int(depth_y2 + 59)][int(depth_x2 + 96)]
                        # else:
                        if shape == 1:
                            depth_y = (depth_y1 + depth_y2) / 2
                            x, y, depth_value_at_x1_y1 = point_xyz_data[int(depth_y + 50)][int(depth_x1 + 80)]
                            x, y, depth_value_at_x2_y2 = point_xyz_data[int(depth_y + 50)][int(depth_x2 + 80)]
                            if v_lr:
                                depth_value_at_x1_y1 = depth_value_at_x1_y1 - ((1660 - depth_x1) * v_0mean)
                                depth_value_at_x2_y2 = depth_value_at_x2_y2 - ((1660 - depth_x2) * v_0mean)
                            else:
                                depth_value_at_x1_y1 = depth_value_at_x1_y1 - (depth_x1 * v_0mean)
                                depth_value_at_x2_y2 = depth_value_at_x2_y2 - (depth_x2 * v_0mean)
                            # cv2.rectangle(depth_frame1, (int(depth_x1 + 78), int(depth_y + 48)),
                            #               (int(depth_x1 + 82), int(depth_y + 52)),
                            #               (0, 0, 255), 2)
                            # cv2.rectangle(depth_frame1, (int(depth_x2 + 78), int(depth_y + 48)),
                            #               (int(depth_x2 + 82), int(depth_y + 52)),
                            #               (0, 0, 255), 2)
                            # cv2.imshow("depth_frame1", depth_frame1)
                        elif shape == 0:
                            depth_x = (depth_x1 + depth_x2) / 2
                            x, y, depth_value_at_x1_y1 = point_xyz_data[int(depth_y1 + 50)][int(depth_x + 80)]
                            x, y, depth_value_at_x2_y2 = point_xyz_data[int(depth_y2 + 50)][int(depth_x + 80)]
                            if v_tb:
                                depth_value_at_x1_y1 = depth_value_at_x1_y1 - ((1262 - depth_y1) * v_1mean)
                                depth_value_at_x2_y2 = depth_value_at_x2_y2 - ((1262 - depth_y2) * v_1mean)
                            else:
                                depth_value_at_x1_y1 = depth_value_at_x1_y1 - (depth_y1 * v_1mean)
                                depth_value_at_x2_y2 = depth_value_at_x2_y2 - (depth_y2 * v_1mean)
                            # cv2.rectangle(depth_frame1, (int(depth_x + 78), int(depth_y1 + 48)),
                            #               (int(depth_x + 82), int(depth_y1 + 52)),
                            #               (0, 0, 255), 2)
                            # cv2.rectangle(depth_frame1, (int(depth_x + 78), int(depth_y2 + 48)),
                            #               (int(depth_x + 82), int(depth_y2 + 52)),
                            #               (0, 0, 255), 2)
                            # cv2.imshow("depth_frame1", depth_frame1)
                        if depth_value_at_x1_y1 < 50 or depth_value_at_x2_y2 < 50:
                            # print("至少一个点的深度值为0，跳过当前数据组")
                            continue
                        distance_to_object = round(depth_value_at_x1_y1)

                        vertical_distance = self.calculate_vertical_distance(depth_value_at_x1_y1,
                                                                             depth_value_at_x2_y2)
                        vertical_distances.append(vertical_distance)
                        # print(depth_value_at_x1_y1)
                        # print(depth_value_at_x2_y2)
                        image_width = 2000
                        image_height = 1500
                        rounded_distance = round(distance_to_object, -2)  # 四舍五入到最接近的百位数
                        if rounded_distance in self.distance_to_object_data:
                            fov_width_pixels, fov_height_pixels = self.distance_to_object_data[rounded_distance]
                            # 其他处理...
                        else:
                            print(f"无法找到 {rounded_distance} 对应的键")

                        row_index = int(y1 / image_height * 10)
                        col_index = int(x1 / image_width * 10)
                        if (row_index, col_index) not in max_vertical_distance:
                            max_vertical_distance[(row_index, col_index)] = (
                                x1, y1, x2, y2, x3, y3, x4, y4, vertical_distance)
                        else:
                            existing_vertical_distance = max_vertical_distance[(row_index, col_index)][-1]
                            if vertical_distance > existing_vertical_distance:
                                max_vertical_distance[(row_index, col_index)] = (
                                    x1, y1, x2, y2, x3, y3, x4, y4, vertical_distance)
                        # if vertical_distance > 3:



                    else:  # 框的位置信息不在图像内
                        # print("深度图像坐标超出范围，跳过当前数据组")
                        continue
                    try:

                        depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
                    except Exception as e:
                        print(f"转换深度图失败")
                        # print(f"转换深度图失败, 错误信息: {e}")
                    finally:
                        check_img = depth_frame[depth_y1:depth_y2, depth_x1:depth_x2]

                        w, h = check_img.shape[1], check_img.shape[0]
                        # 不用计算灰度阈值了，深度图的灰度阈值很清晰
                        # min = int(np.min(
                        #     check_img[int((w // 2) - 5):int((w // 2) + 5), int((h // 2) - 5):int((h // 2) + 5)]))
                        # 二值化，大于阈值得到1
                        ret, thresh = cv2.threshold(check_img, 100, 1, cv2.THRESH_BINARY)
                        # 矩阵之和，白一些的像素有多少
                        summ = np.sum(thresh)
                        # # 总像素减白像素除以宽/高 ，得到一行/一列黑像素有多少

                        # vertical_difference = ((w * h) - summ) // mean
                        # # print(vertical_difference)
                        if shape == 1 and summ != None:
                            row_sums = np.sum(thresh, axis=1)
                            # 总像素减白像素除以宽/高 ，得到一行/一列黑像素有多少
                            midd = ((w * h) - summ) // h
                            mids.append(midd)
                            # print("行和:", row_sums)
                        elif shape == 0 and summ != None:
                            # 计算每列的总和
                            row_sums = np.sum(thresh, axis=0)
                            # 总像素减白像素除以宽/高 ，得到一行/一列黑像素有多少
                            midd = ((w * h) - summ) // w
                            mids.append(midd)
                            # print("列和:", row_sums)
                        # vertical_list = [mean - i for i in row_sums]
                        # maxx, minn = max(vertical_list), min(vertical_list)
                except Exception as e:
                    print(f"错误信息: {e}")
                    pass
            if len(mids) == 0:
                continue
            midd = max(mids)
            vertical_distance = max(vertical_distances)
            # vertical_difference = abs(y2 - y1)  # 左上角对应左下角的差值
            # vertical_distance1 = fov_height_pixels * maxx / image_height
            # vertical_distance2 = fov_height_pixels * minn / image_height
            vertical_distance3 = fov_height_pixels * midd / image_height
            detected_label = {
                "column_index": column_index,  # 添加列索引
                "LC": vertical_distance,
                "LF": vertical_distance3,
            }
            self.detected_labels.append(detected_label)
            if vertical_distance3 >= 10 or vertical_distance >= 3.0:
                global_lc_lf_conditions.append(detected_label)
            # 对检测到的标签进行排序，确保按列索引顺序处理
            global_lc_lf_conditions.sort(key=lambda x: x['column_index'])

            # 更新检测标签列表后，计算最大的LC值
            if self.detected_labels:
                max_lc = max(label['LC'] for label in self.detected_labels if 'LC' in label)
                self.lcMaxUpdated.emit(f"最大LC: {max_lc:.2f}")  # 发射信号，包含最大LC值
            else:
                self.lcMaxUpdated.emit("未检测到任何对象")  # 如果没有标签，发射此消息
            label = f'LC{vertical_distance:.2f}\nLF:{vertical_distance3:.2f}\n'
            if vertical_distance3 >= 3 or vertical_distance >= 3.0:
                self.alarmStatusChanged.emit(True)
                self.tankuang = True
                if vertical_distance3 >= 10:
                    print("触发报警！水平距离大于等于十公分。")
                elif vertical_distance >= 3.0:
                    print("触发报警！垂直距离大于等于三毫米。")

                    pygame.mixer.music.load("D:/PythonCode/sql/sp.mp3")  # 替换为你的音频文件路径
                    pygame.mixer.music.play()

                # if vertical_distance3 >= 10 or vertical_distance >= 3.0:
                #     if vertical_distance3 >= 10:
                #         print("触发报警！水平距离大于等于十公分。")
                #     elif vertical_distance >= 3.0:
                #         print("触发报警！垂直距离大于等于三毫米。")
                #
                #         pygame.mixer.music.load("D:/PycharmProjects/pythonProject/mp3/sp.mp3")  # 替换为你的音频文件路径
                #         pygame.mixer.music.play()
                print(f"Rectangular Coordinates: ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3}), ({x4}, {y4})")


            # cv2.putText(color_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                self.alarmStatusChanged.emit(False)
            lines = label.split('\n')

            # 转换边界框坐标
            rotated_boxes = []
            (h, w) = color_frame1.shape[:2]
            rox1, roy1 = (w - x2, h - y2)
            rox2, roy2 = (w - x1, h - y1)
            rotated_boxes.append((rox1, roy1, rox2, roy2))
            cv2.rectangle(color_frame, (rox1, roy1), (rox2, roy2), (0, 0, 255), 2)

            y = roy1 + 30
            for label in lines:
                cv2.putText(color_frame, label, (rox1 + (box_w // 2), y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 0), 2)
                y += 30
            cv2.imshow("color_frame", color_frame)

            # cv2.putText(color_frame, label, (x1 + (box_w // 2), y1 + (box_h // 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            #             (0, 0, 255), 2)
        detection_img = QImage(color_frame.data, image_width, image_height, QImage.Format_RGB888)
        return detection_img, self.tankuang

    def process_grille(self, color_frame):
        self.tankuang = False
        global global_lc_cf_conditions  # 引用全局变量
        global_lc_cf_conditions = []  # 每次处理前清空全局变量列表
        self.line_threshold = 67  # 基于您的图像尺寸和视野，您可能需要调整这个值
        self.column_threshold = 60
        # QTimer.singleShot(500, lambda: self.detectionCompleted.emit(detection_img))
        self.distance_to_object_data = {
            1200: (1200, 1200),  # 相机距离拍摄物1.2米，视野宽度1.2米，视野高度1米
            1300: (1300, 1070),  # 相机距离拍摄物1.3米，视野宽度1.3米，视野高度1.07米
            1400: (1400, 1130),  # 相机距离拍摄物1.4米，视野宽度1.4米，视野高度1.13米
            1500: (1500, 1200),  # 相机距离拍摄物1.5米，视野宽度1.5米，视野高度1.2米
            1600: (1600, 1280),  # 相机距离拍摄物1.6米，视野宽度1.6米，视野高度1.28米
            1700: (1700, 1360),  # 相机距离拍摄物1.7米，视野宽度1.7米，视野高度1.36米
            1800: (1800, 1440),  # 相机距离拍摄物1.8米，视野宽度1.8米，视野高度1.44米
            1900: (1900, 1520),  # 相机距离拍摄物1.9米，视野宽度1.9米，视野高度1.52米
            2000: (2000, 1600),  # 相机距离拍摄物2.0米，视野宽度2.0米，视野高度1.60米
            2100: (2100, 1680),  # 相机距离拍摄物2.1米，视野宽度2.1米，视野高度1.68米
            2200: (2200, 1760),  # 相机距离拍摄物2.2米，视野宽度2.2米，视野高度1.76米
            2300: (2300, 1840),  # 相机距离拍摄物2.3米，视野宽度2.3米，视野高度1.84米
            2400: (2400, 1920),  # 相机距离拍摄物2.4米，视野宽度2.4米，视野高度1.92米
            2500: (2500, 2000),  # 相机距离拍摄物2.5米，视野宽度2.5米，视野高度2.0米
            2600: (2600, 2080),  # 相机距离拍摄物2.6米，视野宽度2.6米，视野高度2.08米
            2700: (2700, 2160),  # 相机距离拍摄物2.7米，视野宽度2.7米，视野高度2.16米
            2800: (2800, 2240),  # 相机距离拍摄物2.8米，视野宽度2.8米，视野高度2.24米
            2900: (2900, 2320),  # 相机距离拍摄物2.9米，视野宽度2.9米，视野高度2.32米
            3000: (3000, 2400)  # 相机距离拍摄物3.0米，视野宽度3.0米，视野高度2.40米
        }
        if self.selected_model not in self.model_cache:
            print(f"模型 {self.selected_model} 尚未加载")
            return
        model = self.model_cache[self.selected_model]
        model.conf = 0.2  # 根据需要调整置信度阈值
        model.iou = 0.0  # 根据需要调整NMS IOU阈值
        image_width = 2000
        image_height = 1500
        # 进行检测
        results = model(color_frame)
        # 获取彩色图像宽高、深度图宽高
        image_width, image_height = color_frame.shape[1], color_frame.shape[0]  # 彩色图像的宽度和高度
        # 深度图：depth_frame , RGB图：color_frame
        # depth_frame = self.device.capture_depth().data()
        depth_frame = self.depth_image
        # depth_height, depth_width = 1405 - 50, 1877 - 80  # 深度图像的宽度和高度
        # depth_frame = depth_frame[50:1405, 80:1877]
        depth_height, depth_width = 1536 - 50, 1910 - 80  # 深度图像的宽度和高度
        depth_frame = depth_frame[50:, 80:1910]
        # 获取点云数据
        point_xyz_map = self.device.capture_point_xyz()
        point_xyz_data = point_xyz_map.data()
        fov_width_pixels = 0  # 视野范围，不用动
        fov_height_pixels = 0

        max_vertical_distance = {}
        horizontal_difference = 0
        # 在帧上绘制检测结果
        # 获取要分析的图像内容
        self.label_stats = {}
        self.detected_labels = []
        for *xyxy, conf, cls in results.xyxy[0]:
            if cls == 1 or cls == "cuofengg":
                x1, y1, x2, y2 = map(int, xyxy)
                if y2 - y1 <= 1 or x2 - x1 <= 1 or x2 < 200 or y2 > 1400 or y1 < 85:
                    print("检测到水平物体，跳过当前数据组")
                    continue
                x_center = (x1 + x2) / 2
                column_index = int(x_center / (image_width / 10))  # 将图像宽度等分为10份以计算列索引
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                box_w, box_h = x2 - x1, y2 - y1
                if y1 < 50:
                    y1 = 50
                if y2 < 50:
                    y2 = 50
                # 更新标签统计字典
                label_name = results.names[int(cls)]
                if label_name not in self.label_stats:
                    self.label_stats[label_name] = 1
                else:
                    self.label_stats[label_name] += 1
                # 计算另外两个角的坐标
                x3, y3 = x1, y2  # 左下角
                x4, y4 = x2, y1  # 右上角
                # 映射
                b_depth_x1 = int(x1 * depth_width / image_width)
                b_depth_y1 = int(y1 * depth_height / image_height)
                b_depth_x2 = int(x2 * depth_width / image_width)
                b_depth_y2 = int(y2 * depth_height / image_height)

                is_horizontal = abs(x2 - x1) > abs(y2 - y1)
                if is_horizontal:
                    c_depth_x1, c_depth_y1, c_depth_x2, c_depth_y2 = b_depth_x1, b_depth_y1 + (
                            box_h // 4), b_depth_x2, b_depth_y2 - (box_h // 4)
                    mean = c_depth_y2 - c_depth_y1
                    shape = 0
                    # 两个点计算直线度
                    z1y = (c_depth_y1 + c_depth_y2) // 2
                    zh = int((y2 - y1) * 0.7)
                    z1 = c_depth_x1, z1y
                    z2 = c_depth_x2, z1y
                    # # 计算获取深度z值的点竖框x轴的中心点
                    # d_depth_x1, d_depth_y1, d_depth_x2, d_depth_y2 = c_depth_x1, (c_depth_y1+c_depth_y2) // 2, c_depth_x2, (c_depth_y1+c_depth_y2) // 2
                else:
                    c_depth_x1, c_depth_y1, c_depth_x2, c_depth_y2 = b_depth_x1 + (
                            box_w // 4), b_depth_y1, b_depth_x2 - (box_w // 4), b_depth_y2
                    mean = c_depth_x2 - c_depth_x1
                    shape = 1
                    # 两个点计算直线度
                    z1x = (c_depth_x1 + c_depth_x2) // 2
                    zw = int((x2 - x1) * 0.7)
                    z1 = z1x, c_depth_y1, z1x + int((c_depth_x2 - c_depth_x1) * 0.7), c_depth_y1
                    z2 = z1x, c_depth_y2, z1x + int((c_depth_x2 - c_depth_x1) * 0.7), c_depth_y2
                    # # 计算获取深度z值的点竖框y轴的中心点
                    # d_depth_x1, d_depth_y1, d_depth_x2, d_depth_y2 = (c_depth_x1 + c_depth_x2) // 2, c_depth_y1, (c_depth_x1 + c_depth_x2) // 2, c_depth_y2
                # 调整框在深度图上的位置
                if c_depth_y2 < 1250:
                    if c_depth_y2 <= 857:
                        # print("c_x1+14,c_x2+14")
                        c_depth_x1, c_depth_x2 = c_depth_x1 + 14, c_depth_x2 + 14
                    elif 875 < c_depth_y2 and b_depth_x2 < 358:
                        # print("c_y1 - 17, c_y2 - 17")
                        c_depth_y1, c_depth_y2 = c_depth_y1 - 17, c_depth_y2 - 17
                        if c_depth_y2 > 1100:
                            # print("c_x1, c_x2 = c_x1 - 7, c_x2 - 7")
                            c_depth_x1, c_depth_x2 = c_depth_x1 - 7, c_depth_x2 - 7
                    elif c_depth_x2 > 1000 and c_depth_y2 > 931:
                        # print("c_y1 + 14, c_y2 + 14")
                        c_depth_y1, c_depth_y2 = c_depth_y1 + 14, c_depth_y2 + 14
                    if is_horizontal:
                        # 计算获取深度z值的点竖框x轴的中心点
                        d_depth_x1, d_depth_y1, d_depth_x2, d_depth_y2 = c_depth_x1, (
                                                                                             c_depth_y1 + c_depth_y2) // 2, c_depth_x2, (
                                                                                             c_depth_y1 + c_depth_y2) // 2
                    else:
                        # 计算获取深度z值的点竖框y轴的中心点
                        d_depth_x1, d_depth_y1, d_depth_x2, d_depth_y2 = (
                                                                                 c_depth_x1 + c_depth_x2) // 2, c_depth_y1, (
                                                                                 c_depth_x1 + c_depth_x2) // 2, c_depth_y2
                    print(
                        f"Rectangular Coordinates: ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3}), ({x4}, {y4})")  # x1左上，x2右下，x3左下，x4右上
                    # 打印彩色图框位置信息
                    print(f"Color Image Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
                    # 打印深度图框位置信息
                    print(f"Depth Image Coordinates: ({c_depth_x1}, {c_depth_y1}) to ({c_depth_x2}, {c_depth_y2})")
                    # 判断框的位置信息是不是在图像内
                    if 0 <= c_depth_x1 < depth_width and 0 <= c_depth_y1 < depth_height and \
                            0 <= c_depth_x2 < depth_width and 0 <= c_depth_y2 < depth_height:
                        if self.height != 1500 or self.width != 2000:
                            x, y, depth_value_at_x1_y1 = point_xyz_data[int(d_depth_y1 + 59)][int(d_depth_x1 + 96)]
                            x, y, depth_value_at_x2_y2 = point_xyz_data[int(d_depth_y2 + 59)][int(d_depth_x2 + 96)]
                        else:
                            # 获取深度值，此处进行计算
                            x, y, depth_value_at_x1_y1 = point_xyz_data[d_depth_y1 + 50, d_depth_x1 + 80]
                            x, y, depth_value_at_x2_y2 = point_xyz_data[d_depth_y2 + 50, d_depth_x2 + 80]
                        distance_to_object = round(depth_value_at_x1_y1)

                        vertical_distance_int = self.calculate_vertical_distance(depth_value_at_x1_y1,
                                                                                 depth_value_at_x2_y2)
                        vertical_distance = f"LC{vertical_distance_int: .2f}"
                        if depth_value_at_x1_y1 < 1 or depth_value_at_x2_y2 < 1:
                            vertical_distance = "loss"
                        print(depth_value_at_x1_y1)
                        print(depth_value_at_x2_y2)
                        image_width = 2000
                        image_height = 1500
                        rounded_distance = round(distance_to_object, -2)  # 四舍五入到最接近的百位数
                        if rounded_distance in self.distance_to_object_data:
                            fov_width_pixels, fov_height_pixels = self.distance_to_object_data[rounded_distance]
                            # 其他处理...
                        else:
                            print(f"无法找到 {rounded_distance} 对应的键")

                        row_index = int(y1 / image_height * 10)
                        col_index = int(x1 / image_width * 10)
                        if (row_index, col_index) not in max_vertical_distance:
                            max_vertical_distance[(row_index, col_index)] = (
                                x1, y1, x2, y2, x3, y3, x4, y4, vertical_distance)
                        else:
                            existing_vertical_distance = max_vertical_distance[(row_index, col_index)][-1]
                            if vertical_distance > existing_vertical_distance:
                                max_vertical_distance[(row_index, col_index)] = (
                                    x1, y1, x2, y2, x3, y3, x4, y4, vertical_distance)

                        cv2.rectangle(color_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)


                    else:  # 框的位置信息不在图像内
                        print("深度图像坐标超出范围，跳过当前数据组")
                        continue
                    try:
                        # 转换灰度图 直接调用深度图，深度图就是灰度图
                        # check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2GRAY)
                        # 计算区域内最小值(最黑)

                        depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
                    except Exception as e:
                        print(f"转换深度图失败, 错误信息: {e}")
                    finally:
                        check_img = depth_frame[c_depth_y1:c_depth_y2, c_depth_x1:c_depth_x2]

                        w, h = check_img.shape[1], check_img.shape[0]
                        # 不用计算灰度阈值了，深度图的灰度阈值很清晰
                        # min = int(np.min(
                        #     check_img[int((w // 2) - 5):int((w // 2) + 5), int((h // 2) - 5):int((h // 2) + 5)]))
                        # 二值化，大于阈值得到1
                        ret, thresh = cv2.threshold(check_img, 100, 1, cv2.THRESH_BINARY)
                        # 矩阵之和，白一些的像素有多少
                        summ = np.sum(thresh)
                        # # 总像素减白像素除以宽/高 ，得到一行/一列黑像素有多少

                        # vertical_difference = ((w * h) - summ) // mean
                        # # print(vertical_difference)
                        if shape == 1 and summ != None:
                            row_sums = np.sum(thresh, axis=1)
                            # 总像素减白像素除以宽/高 ，得到一行/一列黑像素有多少
                            midd = ((w * h) - summ) // w
                            z = 0
                            zz = [point_xyz_data[z1[1], z1[0]], point_xyz_data[z2[1], z2[0]]]
                            for i in range(zw):
                                z1x, z1y, z1z = point_xyz_data[int(z1[1]) + 50, int(z1[0] + i) + 80]
                                z2x, z2y, z2z = point_xyz_data[int(z2[1]) + 50, int(z2[0] + i) + 80]
                                # 假设zz, z1z, 和 z2z已经定义且是NumPy数组
                                condition1 = np.abs(zz[0] - z1z) > 500
                                condition2 = np.abs(zz[1] - z2z) > 500
                                c1 = zz[0] + 10
                                print("z1", z1z, "z2", z2z)

                                # 检查是否至少有一个元素满足条件
                                if condition1.any() or condition2.any():
                                    zc = abs(z1z - z2z)
                                    if zc > 150:
                                        if (zc < c1).any():
                                            # print(f"{zc}")
                                            z += 1
                            # print("行和:", row_sums)
                        elif shape == 0 and summ != None:
                            # 计算每列的总和
                            row_sums = np.sum(thresh, axis=0)
                            # 总像素减白像素除以宽/高 ，得到一行/一列黑像素有多少
                            midd = ((w * h) - summ) // h
                            z = 0
                            zz = [point_xyz_data[z1[1], z1[0]], point_xyz_data[z2[1], z2[0]]]
                            for i in range(zh):
                                z1x, z1y, z1z = point_xyz_data[int(z1[1] + i) + 50, int(z1[0]) + 80]
                                z2x, z2y, z2z = point_xyz_data[int(z2[1] + i) + 50, int(z2[0]) + 80]
                                # 假设zz, z1z, 和 z2z已经定义且是NumPy数组
                                condition1 = np.abs(zz[0] - z1z) > 500
                                condition2 = np.abs(zz[1] - z2z) > 500
                                c1 = zz[0] - 10
                                # print(f"z1\t,{z1x}\t, {z1y}\t, {z1z}\t, z2\t, {z2x}\t, {z2y}\t, {z2z}\t")
                                # print(f"z1,{z1z}\t,z2z,{z2z}")

                                # 检查是否至少有一个元素满足条件
                                if condition1.any() or condition2.any():
                                    zc = abs(z1z - z2z)
                                    if zc > 150:
                                        if (zc < c1).any():
                                            # print(f"{zc}")
                                            z += 1
                            # print("列和:", row_sums)
                        else:
                            print("框的位置出错")
                            midd = 0
                            z = 0
                        # vertical_list = [mean - i for i in row_sums]
                        # maxx, minn = max(vertical_list), min(vertical_list)
                        # cv2.rectangle(depth_frame, (int(c_depth_x1 - 3), int(c_depth_y1 - 3)),
                        #               (int(c_depth_x2 + 3), int(c_depth_y2 + 3)),
                        #               (0, 0, 255), 2)

                        # vertical_difference = abs(y2 - y1)  # 左上角对应左下角的差值
                    # vertical_distance1 = fov_height_pixels * maxx / image_height
                    # vertical_distance2 = fov_height_pixels * minn / image_height
                    vertical_distance3 = fov_height_pixels * midd / image_height
                    vertical_distance4 = fov_height_pixels * z / image_height
                    detected_label = {
                        "column_index": column_index,  # 添加列索引
                        "LC": vertical_distance,
                        # "LF": vertical_distance3,
                        "CF": vertical_distance4,
                    }
                    self.detected_labels.append(detected_label)
                    if vertical_distance4 >= 10 or vertical_distance_int >= 3.0:
                        global_lc_cf_conditions.append(detected_label)
                        # 对检测到的标签进行排序，确保按列索引顺序处理
                    global_lc_lf_conditions.sort(key=lambda x: x['column_index'])
                    if self.detected_labels:
                        max_lc = max(label['LC'] for label in self.detected_labels if 'LC' in label)
                        max_lc1 = f"最大LC: {max_lc}"
                        self.lcMaxUpdated.emit(max_lc1)  # 发射信号，包含最大LC值
                    else:
                        self.lcMaxUpdated.emit("未检测到任何对象")  # 如果没有标签，发射此消息
                    label = f'{vertical_distance} \n LF:{vertical_distance3:.2f} \n CF:{vertical_distance4:.2f}\n'
                    if vertical_distance3 >= 10 or vertical_distance_int >= 3.0:
                        self.alarmStatusChanged.emit(True)
                        self.tankuang = True
                        if vertical_distance3 >= 10:
                            print("触发报警！水平距离大于等于十公分。")
                        elif vertical_distance_int >= 3.0:
                            print("触发报警！垂直距离大于等于三毫米。")

                            pygame.mixer.music.load("D:/PythonCode/sql/sp.mp3")  # 替换为你的音频文件路径
                            pygame.mixer.music.play()

                        print(f"Rectangular Coordinates: ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3}), ({x4}, {y4})")
                        # 文本位置在框中间
                        # cv2.putText(color_frame, label, (x1 + (box_w // 2), y1 + (box_h // 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        #             1.0, (0, 0, 255), 2)

                    else:
                        self.alarmStatusChanged.emit(False)
                    lines = label.split('\n')
                    y = y1 + 30
                    for label in lines:
                        cv2.putText(color_frame, label, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (0, 255, 0), 2)
                        y += 30
                    # cv2.putText(color_frame, label, (x1 + (box_w // 2), y1 + (box_h // 2)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    #             (0, 0, 255), 2)
                    # cv2.putText(color_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        # 处理检测结果
        # self.process_detections(results, color_frame)
        # 更新UI显示
        # self.display_image(color_frame)
        detection_img = QImage(color_frame.data, image_width, image_height, QImage.Format_RGB888)
        return detection_img, self.tankuang

    def process_aluminum_square_tube(self, color_frame):
        self.tankuang = False
        global global_lc_cf_conditions  # 引用全局变量
        global_lc_cf_conditions = []  # 每次处理前清空全局变量列表
        self.line_threshold = 67  # 基于您的图像尺寸和视野，您可能需要调整这个值
        self.column_threshold = 60
        # QTimer.singleShot(500, lambda: self.detectionCompleted.emit(detection_img))
        self.distance_to_object_data = {
            1200: (1200, 1200),  # 相机距离拍摄物1.2米，视野宽度1.2米，视野高度1米
            1300: (1300, 1070),  # 相机距离拍摄物1.3米，视野宽度1.3米，视野高度1.07米
            1400: (1400, 1130),  # 相机距离拍摄物1.4米，视野宽度1.4米，视野高度1.13米
            1500: (1500, 1200),  # 相机距离拍摄物1.5米，视野宽度1.5米，视野高度1.2米
            1600: (1600, 1280),  # 相机距离拍摄物1.6米，视野宽度1.6米，视野高度1.28米
            1700: (1700, 1360),  # 相机距离拍摄物1.7米，视野宽度1.7米，视野高度1.36米
            1800: (1800, 1440),  # 相机距离拍摄物1.8米，视野宽度1.8米，视野高度1.44米
            1900: (1900, 1520),  # 相机距离拍摄物1.9米，视野宽度1.9米，视野高度1.52米
            2000: (2000, 1600),  # 相机距离拍摄物2.0米，视野宽度2.0米，视野高度1.60米
            2100: (2100, 1680),  # 相机距离拍摄物2.1米，视野宽度2.1米，视野高度1.68米
            2200: (2200, 1760),  # 相机距离拍摄物2.2米，视野宽度2.2米，视野高度1.76米
            2300: (2300, 1840),  # 相机距离拍摄物2.3米，视野宽度2.3米，视野高度1.84米
            2400: (2400, 1920),  # 相机距离拍摄物2.4米，视野宽度2.4米，视野高度1.92米
            2500: (2500, 2000),  # 相机距离拍摄物2.5米，视野宽度2.5米，视野高度2.0米
            2600: (2600, 2080),  # 相机距离拍摄物2.6米，视野宽度2.6米，视野高度2.08米
            2700: (2700, 2160),  # 相机距离拍摄物2.7米，视野宽度2.7米，视野高度2.16米
            2800: (2800, 2240),  # 相机距离拍摄物2.8米，视野宽度2.8米，视野高度2.24米
            2900: (2900, 2320),  # 相机距离拍摄物2.9米，视野宽度2.9米，视野高度2.32米
            3000: (3000, 2400)  # 相机距离拍摄物3.0米，视野宽度3.0米，视野高度2.40米
        }
        if self.selected_model not in self.model_cache:
            print(f"模型 {self.selected_model} 尚未加载")
            return
        model = self.model_cache[self.selected_model]
        model.conf = 0.2  # 根据需要调整置信度阈值
        model.iou = 0.0  # 根据需要调整NMS IOU阈值
        image_width = 2000
        image_height = 1500
        # 进行检测
        results = model(color_frame)
        # 处理检测结果
        self.process_detections(results, color_frame)
        # 更新UI显示
        # self.display_image(color_frame)
        detection_img = QImage(color_frame.data, image_width, image_height, QImage.Format_RGB888)
        return detection_img,self.tankuang

    def draw_boxes(self, image, boxes, distances, y_diffs=None, depth_diffs=None, color=(0, 255, 0), thickness=2,
                   font_scale=0.5, font_thickness=2):
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box[:4]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            info_texts = []
            if i < len(distances):
                info_texts.append(f"LF: {distances[i]:.2f}\n")
            if y_diffs and i < len(y_diffs):
                info_texts.append(f"CF: {y_diffs[i]:.2f}\n")
            if depth_diffs and i < len(depth_diffs):
                info_texts.append(f"LC: {depth_diffs[i]:.2f}\n")

            text = "".join(info_texts)
            lines = text.split('\n')
            y = int(y1) - 10
            for label in lines:
                cv2.putText(image, label, (int(x2) + 100, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                            font_thickness)
                y += 30
            # cv2.putText(image, text, (int(x1) + 10, int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
            #             font_thickness)

    def draw_boxes_for_type(self, boxes, color_frame, color=(0, 255, 0), thickness=2):
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            cv2.rectangle(color_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    def process_detections(self, results, color_frame):
        '''
        :param results: 框的信息
        :param color_frame: 彩色图片
        :return:
        '''
        global vertical_distance, vertical_list
        image_width_px = 2000  # 图像宽度，单位：像素
        image_height_px = 1500  # 图像高度，单位：像素

        detections = results.xyxy[0].numpy()  # 转换结果为numpy数组

        cuofeng_class_id = 0  # 假设的类别ID，需要用实际的替换
        cuofengg_class_id = 1  # 假设的另一个类别ID

        cuofengg_boxes = detections[detections[:, 5] == cuofengg_class_id]  # 筛选cuofengg类型的边界框
        boxes_type_1 = detections[detections[:, 5] == cuofeng_class_id]  # 筛选cuofeng类型的边界框
        # depth_frame1 = self.device.capture_depth().data()
        depth_frame1 = self.depth_image
        # 读取深度图
        depth_frame = depth_frame1[50:1405, 80:1877]
        # 深度图的分辨率
        depth_height, depth_width = 1405 - 50, 1877 - 80
        # 获取点云数据
        point_xyz_map = self.device.capture_point_xyz()
        point_xyz_data = point_xyz_map.data()
        self.fov_width_pixels = 0
        self.fov_height_pixels = 0
        self.label_stats = {}

        for *xyxy, conf, cls in results.xyxy[0]:
            # x1, y1, x2, y2 = map(int, xyxy)
            # print("yuanshi", results.xyxy[0])

            # 更新标签统计字典
            label_name = results.names[int(cls)]
            if label_name not in self.label_stats:
                self.label_stats[label_name] = 1
            else:
                self.label_stats[label_name] += 1

        # 打印每个边界框的四个顶点坐标
        for detection in detections:
            x1, y1, x2, y2, conf, class_id = detection[:6]
            center_x = (x1 + x2) / 2
            self.column_index = int(center_x / (image_width_px / 10))
            print(f"边界框类别ID {class_id}:")
            print(f"    左上角 (x1, y1): ({x1}, {y1})")
            print(f"    右上角 (x2, y1): ({x2}, {y1})")
            print(f"    左下角 (x1, y2): ({x1}, {y2})")
            print(f"    右下角 (x2, y2): ({x2}, {y2})")
            # depth_height, depth_width = 1405 - 50, 1877 - 80
            depth_x1 = int(x1 * depth_width / image_width_px)
            depth_y1 = int(y1 * depth_height / image_height_px)
            depth_x2 = int(x2 * depth_width / image_width_px)
            depth_y2 = int(y2 * depth_height / image_height_px)
            depth_value_at_x1_y1 = depth_frame[depth_y1, depth_x1]
            # depth_value_at_x2_y2 = depth_frame[depth_y2, depth_x2]
            distance_to_object = round(depth_value_at_x1_y1)
            rounded_distance = round(distance_to_object, -2)  # 四舍五入到最接近的百位数

            if rounded_distance in self.distance_to_object_data:
                self.fov_width_pixels, self.fov_height_pixels = self.distance_to_object_data[rounded_distance]
            else:
                print(f"无法找到 {rounded_distance} 对应的键")
            self.pixel_size_width_m = self.fov_width_pixels / image_width_px  # 每像素代表的宽度，单位：米
            self.pixel_size_height_m = self.fov_height_pixels / image_height_px  # 每像素代表的高度，单位：米

        def calculate_pixel_difference(point1, point2):
            return abs(point1[0] - point2[0]), abs(point1[1] - point2[1])

        # 定义一个函数来计算框的中心点
        def calculate_box_center(box):
            return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), box[2] - box[0]

        for cuofengg_box in cuofengg_boxes:
            # depth_frame = self.device.capture_depth().data()

            # depth_height, depth_width = 1405 - 50, 1877 - 80
            inside_cuofeng_boxes = detections[(detections[:, 5] == cuofeng_class_id) &
                                              (detections[:, 0] >= cuofengg_box[0]) &
                                              (detections[:, 1] >= cuofengg_box[1]) &
                                              (detections[:, 2] <= cuofengg_box[2]) &
                                              (detections[:, 3] <= cuofengg_box[3])]
            inside_cuofeng_boxes = sorted(inside_cuofeng_boxes, key=lambda x: x[0])  # 按x坐标排序
            print(f"k:{inside_cuofeng_boxes}")
            for i in range(len(inside_cuofeng_boxes) - 1):
                # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n",
                #       f"kuang1:{inside_cuofeng_boxes[i]}\tkuang2:{inside_cuofeng_boxes[i + 1]}")

                center1, box_h = calculate_box_center(inside_cuofeng_boxes[i])
                center2, box2_h = calculate_box_center(inside_cuofeng_boxes[i + 1])
                # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n",
                #       f"kuang1:{center1, box_h}\tkuang2:{center2, box2_h}")

                x1, y1 = center1
                x2, y2 = center2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                is_horizontal = abs(x2 - x1) > abs(y2 - y1)

                if is_horizontal:
                    x1, y1, x2, y2 = x1, y1 - (box_h // 4), x2, y2 + (box2_h // 4)
                else:
                    x1, y1, x2, y2 = x1 - (box_h // 4), y1, x2 + (box2_h // 4), y2

                # # 深度图的分辨率
                # depth_height, depth_width = 1405 - 50, 1877 - 80
                # RGB图的分辨率
                # image_width, image_height = 2000, 1500
                # 映射到深度图
                counts = lv_mapping2deep(x1, y1, x2, y2)
                if len(counts) == 6 :
                    c_depth_x1, c_depth_y1, c_depth_x2, c_depth_y2, mean, shape = counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]
                else:
                    continue

                # d_height, d_width = 1405 - 50, 1910 - 80
                # weight, height = 2000, 1500
                # d_x1 = abs(x1 + (x1 * (0.02 * (x1 / weight))))
                # # d_x1 = x1
                # d_y1 = abs(y1 - (y1 * (0.05 * (y1 / height))))
                # d_x2 = abs(x2 + (x2 * (0.012 * (x2 / weight))))
                # # d_x2 = x2
                # d_y2 = abs(y2 - (y2 * (0.05 * (y2 / height))))
                # # print("\tRGB", x1, "Dep", d_x1, "\tRGB", y1, "Dep", d_y1, "\tRGB", x2, "Dep", d_x2, "\tRGB", y2, "Dep", d_y2)
                # c_depth_x1 = int(d_x1 * d_width / weight)
                # c_depth_y1 = int(d_y1 * d_height / height)
                # c_depth_x2 = int(d_x2 * d_width / weight)
                # c_depth_y2 = int(d_y2 * d_height / height)

                # c_depth_x1 = int(x1 * depth_width / image_width_px)
                # c_depth_y1 = int(y1 * depth_height / image_height_px)
                # c_depth_x2 = int(x2 * depth_width / image_width_px)
                # c_depth_y2 = int(y2 * depth_height / image_height_px)

                # if c_depth_y2 < 1250:
                #     if c_depth_y2 <= 857:
                #         # print("c_depth_x1+14,depth_x2+14")
                #         c_depth_x1, c_depth_x2 = c_depth_x1 + 18, c_depth_x2 + 18
                #     if c_depth_x1 > 1000 and 857 < c_depth_y1:
                #         c_depth_x1, c_depth_x2 = c_depth_x1 + 18, c_depth_x2 + 18
                #     if 875 < c_depth_y1 and c_depth_x1 < 358:
                #         # print("depth_y1 - 17, c_depth_y2 - 17")
                #         c_depth_y1, c_depth_y2 = c_depth_y1 - 17, c_depth_y2 - 17
                #         if c_depth_y1 > 1100:
                #             # print("c_depth_x1, c_depth_x2 = c_depth_x1 - 7, c_depth_x2 - 7")
                #             c_depth_x1, c_depth_x2 = c_depth_x1 - 7, c_depth_x2 - 7
                #     if c_depth_x1 > 1000 and c_depth_y1 > 931:
                #         # print("depth_y1 + 14, c_depth_y2 + 14")
                #         c_depth_y1, c_depth_y2 = c_depth_y1 + 14, c_depth_y2 + 14
                # cv2.rectangle(depth_frame, (c_depth_x1, c_depth_y1),
                #               (c_depth_x2, c_depth_y2),
                #               (0, 0, 255), 2)
                # 截取要计算的位置
                try:
                    # 转换灰度图 直接调用深度图，深度图就是灰度图
                    # check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2GRAY)
                    # 计算区域内最小值(最黑)

                    depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
                except Exception as e:
                    print(f"转换深度图失败, 错误信息: {e}")
                # depth_fram = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
                check_img = depth_frame[c_depth_y1:c_depth_y2, c_depth_x1:c_depth_x2]
                w, h = check_img.shape[1], check_img.shape[0]
                # 二值化，大于阈值得到1
                ret, thresh = cv2.threshold(check_img, 100, 1, cv2.THRESH_BINARY)
                # 矩阵之和，白一些的像素有多少
                # print(thresh)
                summ = np.sum(thresh)
                # is_horizontal = abs(c_depth_x2 - c_depth_x1) > abs(c_depth_y2 - c_depth_y1)
                if is_horizontal:
                    # 横向框
                    right_top_point = [inside_cuofeng_boxes[i][2], inside_cuofeng_boxes[i][1]]  # 左边框的右上角顶点
                    left_top_point = [inside_cuofeng_boxes[i + 1][0], inside_cuofeng_boxes[i + 1][1]]  # 右边框的左上角顶点
                    x_diff, y_diff = calculate_pixel_difference(right_top_point, left_top_point)
                    # # 计算每列的总和
                    # row_sums = np.sum(thresh, axis=1)
                    # print("列和:", row_sums)
                    # 总像素减白像素除以宽/高 ，得到一行/一列黑像素有多少
                    # 平均值
                    if summ:
                        midd = ((w * h) - summ) // h
                        print("狂赌像素", midd)
                        # 计算每排裂缝像素
                        # vertical_list = [c_depth_y2 - c_depth_y1 - i for i in row_sums]
                        vertical_list = int(midd * self.pixel_size_width_m)
                    else:
                        print(1)
                        vertical_list = 0
                    if self.height != 1500 or self.width != 2000:
                        vertical_list = vertical_list / 0.875
                    x_diff_mm = int(x_diff * self.pixel_size_width_m)
                    y_diff_mm = int(y_diff * self.pixel_size_height_m)
                    print(
                        f"横向 - 框 {i} 和框 {i + 1} 之间的像素差值: x差 = {x_diff_mm}, y差 = {y_diff_mm}, 裂缝宽度={vertical_list}")

                    # 获取深度值并计算差值
                    try:
                        depth_y = (c_depth_y2 + c_depth_y1) // 2
                        # depth_value_right_top = depth_frame1[int(c_depth_x1 + 80), int(depth_y + 50)]
                        # depth_value_left_top = depth_frame1[int(c_depth_x2 + 80), int(depth_y + 50)]
                        # if self.height != 1500 or self.width != 2000:
                        #     try:
                        #         x, y, depth_value_right_top = point_xyz_data[int(depth_y -9)][int(c_depth_x1 -13)]
                        #     except:
                        #         depth_value_right_top = 0
                        #     try:
                        #         x, y, depth_value_left_top = point_xyz_data[int(depth_y -9)][int(c_depth_x2 -13)]
                        #     except:
                        #         depth_value_left_top = 0
                        # else:
                        list1, list2 = [], []
                        for ii in range(1, 5):
                            x, y, depth_value_right_top1 = point_xyz_data[int(depth_y + 50 + ii)][
                                int(c_depth_x1 + 80)]
                            if depth_value_right_top1 > 50:
                                list1.append(depth_value_right_top1)
                            x, y, depth_value_right_top2 = point_xyz_data[int(depth_y + 50 - ii)][
                                int(c_depth_x1 + 80)]
                            if depth_value_right_top2 > 50:
                                list1.append(depth_value_right_top2)
                            x, y, depth_value_left_top1 = point_xyz_data[int(depth_y + 50 + ii)][int(c_depth_x2 + 80)]
                            if depth_value_left_top1 > 50:
                                list2.append(depth_value_left_top1)
                            x, y, depth_value_left_top2 = point_xyz_data[int(depth_y + 50 - ii)][int(c_depth_x2 + 80)]
                            if depth_value_left_top2 > 50:
                                list2.append(depth_value_left_top2)
                        # print("x加80，y+50, 去软件里看", c_depth_x1 + 80, depth_y + 50, c_depth_x2 + 80, depth_y + 50)
                        depth_value_right_top = np.min(list1)
                        depth_value_left_top = np.min(list2)
                        vertical_distance = self.calculate_vertical_distance(depth_value_right_top,
                                                                             depth_value_left_top)
                        # depth_frame2 = self.device.capture_depth().data()
                        # cv2.rectangle(depth_frame1, (int(c_depth_x1 + 77), int(c_depth_y1 + 47)),
                        #               (int(c_depth_x1 + 87), int(c_depth_y1 + 47)),
                        #               (0, 0, 255), 2)
                        # cv2.rectangle(depth_frame1, (int(c_depth_x2 + 77), int(c_depth_y2 + 47)),
                        #               (int(c_depth_x2 - 83), int(c_depth_y2 + 53)),
                        #               (0, 0, 255), 2)
                        # # color_image = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
                        #
                        # cv2.imshow("depth_frame", depth_frame1)
                        # cv2.imshow("depth_frame1", depth_frame1)
                    except Exception as e:
                        print(f"超出边界，错误信息{e}")
                        vertical_distance = 0

                    # print(f"depthshape{depth_frame.shape}k:{depth_value_right_top},{depth_value_left_top}")
                    print(f"横向 - 框 {i} 和框 {i + 1} 之间的深度差值: {vertical_distance}mm")
                    if y_diff_mm > 10:
                        pygame.mixer.music.load("D:/PythonCode/sql/sp.mp3")  # 替换为你的音频文件路径
                        pygame.mixer.music.play()

                    self.draw_boxes(color_frame, [inside_cuofeng_boxes[i], inside_cuofeng_boxes[i + 1]],
                                    [vertical_list], [y_diff_mm], [vertical_distance])
                    detected_label = {
                        "column_index": self.column_index,
                        "LC": vertical_distance,
                        "CF": y_diff_mm
                    }
                    global_lc_cf_conditions.append(detected_label)
                    global_lc_lf_conditions.sort(key=lambda x: x['column_index'])
                else:
                    # 纵向框
                    upper_box_right_bottom = (inside_cuofeng_boxes[i][2], inside_cuofeng_boxes[i][3])  # 上边框的右下角坐标（x,y）
                    lower_box_right_top = (
                        inside_cuofeng_boxes[i + 1][2], inside_cuofeng_boxes[i + 1][1])  # 下边框的右上角坐标（x,y）

                    # 计算x方向和y方向的差值

                    x_diff = abs(upper_box_right_bottom[0] - lower_box_right_top[0])  # 计算两个框在x方向上的差异

                    y_diff = abs(upper_box_right_bottom[1] - lower_box_right_top[1])  # 计算两个框在y方向上的差异
                    # row_sums = np.sum(thresh, axis=0)
                    # print("行和:", row_sums)
                    # 总像素减白像素除以宽/高 ，得到一行/一列黑像素有多少
                    # 平均值
                    if summ:
                        midd = ((w * h) - summ) // w
                        # 计算每排裂缝像素
                        # vertical_list = [c_depth_y2 - c_depth_y1 - i for i in row_sums]
                        vertical_list = int(midd * self.pixel_size_width_m)
                    else:
                        vertical_list = 0
                    if self.height != 1500 or self.width != 2000:
                        vertical_list = vertical_list / 0.850
                    x_diff_mm = int(x_diff * self.pixel_size_width_m)

                    y_diff_mm = int(y_diff * self.pixel_size_height_m)

                    # 转换坐标以适应深度图的分辨率
                    depth_upper_box_right_bottom = (int(upper_box_right_bottom[0] * depth_width / image_width_px),
                                                    int(upper_box_right_bottom[1] * depth_height / image_height_px))
                    depth_lower_box_right_top = (int(lower_box_right_top[0] * depth_width / image_width_px),
                                                 int(lower_box_right_top[1] * depth_height / image_height_px))
                    # 获取深度值并计算差值
                    # 获取深度值并计算差值
                    try:
                        depth_x = (c_depth_x2 + c_depth_x1) // 2
                        # depth_value_right_top = depth_frame1[int(depth_x + 80), int(c_depth_y1 + 50)]
                        # depth_value_left_top = depth_frame1[int(depth_x + 80), int(c_depth_y2 + 50)]
                        list1, list2 = [], []
                        for ii in range(1, 5):
                            x, y, depth_value_right_top1 = point_xyz_data[int(c_depth_y1 + 50)][
                                int(depth_x + 80 + ii)]
                            if depth_value_right_top1 > 50:
                                list1.append(depth_value_right_top1)
                            x, y, depth_value_right_top2 = point_xyz_data[int(c_depth_y1 + 50)][
                                int(depth_x + 80 - ii)]
                            if depth_value_right_top2 > 50:
                                list1.append(depth_value_right_top2)
                            x, y, depth_value_left_top1 = point_xyz_data[int(c_depth_y2 + 50)][int(depth_x + 80 + ii)]
                            if depth_value_left_top1 > 50:
                                list2.append(depth_value_left_top1)
                            x, y, depth_value_left_top2 = point_xyz_data[int(c_depth_y2 + 50)][int(depth_x + 80 - ii)]
                            if depth_value_left_top2 > 50:
                                list2.append(depth_value_left_top2)
                        depth_value_right_top = np.min(list1)
                        depth_value_left_top = np.min(list2)

                        # print("x加80，y+50, 去软件里看", depth_x + 80, c_depth_y1 + 50, depth_x + 80, c_depth_y2 + 50)
                        # depth_difference = abs(depth_value_right_top - depth_value_left_top)
                        vertical_distance = self.calculate_vertical_distance(depth_value_right_top,
                                                                            depth_value_left_top)
                        # depth_frame2 = self.device.capture_depth().data()

                        # cv2.rectangle(depth_frame1, (int(c_depth_x1 + 77), int(c_depth_y1 + 47)),
                        #               (int(c_depth_x1 + 83), int(c_depth_y1 + 53)),
                        #               (0, 0, 255), 2)
                        # cv2.rectangle(depth_frame1, (int(c_depth_x2 + 77), int(c_depth_y2 + 47)),
                        #               (int(c_depth_x2 + 83), int(c_depth_y2 + 53)),
                        #               (0, 0, 255), 2)
                        # cv2.imshow("depth_frame", depth_frame1)

                        print(f"depthshape{depth_frame.shape}k:{depth_value_right_top},{depth_value_left_top}")
                        # print(f"横向 - 框 {i} 和框 {i + 1} 之间的深度差值: {vertical_distance}mm")
                        print(f"纵向 - 框 {i} 和框 {i + 1} 之间的深度差值: {vertical_distance}mm")
                    except Exception as e:
                        print(f"超出边界，错误信息{e}")
                        vertical_distance = 0

                    print(
                        f"纵向 - 框 {i} 和框 {i + 1} 之间的像素差值: x差 = {x_diff_mm}mm, y差 = {y_diff_mm}mm,, 裂缝宽度={vertical_list}")

                if vertical_list >= 10 or vertical_distance >= 3.0:
                    self.alarmStatusChanged.emit(True)
                    self.tankuang = True
                    print("触发报警！垂直距离大于等于三毫米或水平距离大于等于十公分")
                    # if inside_cuofeng_boxes >= 10:
                    #     print("触发报警！水平距离大于等于十公分。")
                    # elif depth_difference >= 3.0:
                    #     print("触发报警！垂直距离大于等于三毫米。")

                    pygame.mixer.music.load("D:/PythonCode/sql/cz.mp3")  # 替换为你的音频文件路径
                    pygame.mixer.music.play()
                    detected_label = {
                        "column_index": self.column_index,
                        "LC": vertical_distance,
                        "CF": y_diff_mm
                    }
                    global_lc_cf_conditions.append(detected_label)
                    global_lc_lf_conditions.sort(key=lambda x: x['column_index'])
                    if self.detected_labels:
                        max_lc = max(label['LC'] for label in self.detected_labels if 'LC' in label)
                        self.lcMaxUpdated.emit(f"最大LC: {max_lc}")  # 发射信号，包含最大LC值
                    else:
                        self.lcMaxUpdated.emit("未检测到任何对象")  # 如果没有标签，发射此消息
                    # 打印或处理大于3毫米的深度差的标签
                else:
                    self.alarmStatusChanged.emit(False)
                for label in self.detected_labels:
                    print(f"深度差大于3毫米的标签: {label}")
                # distance_text = f"lc:{depth_difference},lf:{vertical_list2},cf:{x_diff_mm}"
                # cv2.putText(color_frame, distance_text, (center1[0],center1[1]-10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                self.draw_boxes(color_frame, [inside_cuofeng_boxes[i], inside_cuofeng_boxes[i + 1]],
                                [vertical_list], [y_diff_mm], [vertical_distance])

        self.draw_boxes_for_type(boxes_type_1, color_frame, color=(0, 255, 0))
        self.draw_boxes_for_type(cuofengg_boxes, color_frame, color=(255, 0, 0))

    def process_image(self, color_frame):
        # 进行实际的图像处理，这里只是一个占位符
        # 实际中你可能会用到一些图像处理技术，比如边缘检测、特征提取等
        detection_img = color_frame  # 占位符，实际应处理图像
        return detection_img

    def check_alarm_condition(self, detection_img):
        # 判断图像处理结果是否满足报警条件
        # 这里需要根据实际需求进行实现，比如图像中是否存在某些异常特征
        # 这只是一个占位符
        alarm_condition_met = False
        if self.some_condition_based_on(detection_img):
            alarm_condition_met = True
        return alarm_condition_met

    def some_condition_based_on(self, detection_img):
        # 根据图像内容判断是否满足报警条件的占位符
        # 实际实现应根据你的需求来
        return False

    def trigger_alarm(self):
        # 触发报警的操作，可以是发出警告声音、显示警告信息等
        print("触发报警！水平距离大于等于十公分。")
        pygame.mixer.music.load("D:/PythonCode/sql/sp.mp3")  # 替换为你的音频文件路径
        pygame.mixer.music.play()

    def trigger_alarm1(self):
        # 触发报警的操作，可以是发出警告声音、显示警告信息等
        print("触发报警！垂直距离大于等于三毫米。")
        pygame.mixer.music.load("D:/PythonCode/sql/cz.mp3")  # 替换为你的音频文件路径
        pygame.mixer.music.play()

    def analyse_data(self):
        # 在这里执行分析
        # 发射一个信号，通知分析完成
        self.analysisCompleted.emit()

    def get_depth(self):
        return self.depth_image

    def get_current_images(self):
        return self.current_color_image, self.current_depth_image

    def get_label_stats(self):
        return self.label_stats

    def show_zoom_window(self):
        # 显示 zoomwindow 窗口的逻辑
        # 这里假设你有一个 ZoomWindow 类，实例化并显示它
        zoom_window = ZoomWindow(self.parent_window)
        zoom_window.show()

    # 假设这是触发ZoomWindow显示的方法
    def on_images_captured(self, color_img, depth_img):
        # 创建ZoomWindow实例
        zoom_window = ZoomWindow(color_img, self, self.parent_window)
        # 设置detected_labels（如果已有）
        zoom_window.set_detected_labels(self.detected_labels)
        print(f'测试:', self.detected_labels)
        # 新增：传递label_stats
        zoom_window.set_label_stats(self.get_label_stats())  # 确保你有这样的方法获取label_stats
        # 新增传递模型名称
        zoom_window.set_model_name(self.selected_model)
        print('ce:', self.get_label_stats())
        zoom_window.exec()


class SerialReaderThread(QThread):
    def __init__(self, serial_port, data_handler):
        super().__init__()
        self.serial_port = serial_port
        self.data_handler = data_handler

    def run(self):
        while True:
            try:
                data = self.serial_port.read(2)  # 读取期望的数据长度
                if data:
                    self.data_handler(data)
            except Exception as e:
                # 处理读取串口数据时的异常
                print(f"读取串口数据时出错：{str(e)}")


class ZoomableImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)
        self.is_zoomed = False
        self.zoom_factor = 1.0
        self.setMinimumSize(50, 50)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.last_mouse_pos = None

    def mouseDoubleClickEvent(self, event):
        if self.is_zoomed:
            self.setGeometry(self.original_geometry)
            self.setScaledContents(False)
            self.is_zoomed = False
            self.zoom_factor = 1.0
        else:
            self.original_geometry = self.geometry()
            self.setGeometry(self.parentWidget().geometry())
            self.setScaledContents(True)
            self.is_zoomed = True
            self.zoom_factor = 1.0
        self.update_pixmap()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor /= 1.1
        self.update_pixmap()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            self.move(self.pos() + delta)
            self.last_mouse_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = None

    def update_pixmap(self):
        if self.pixmap():
            size = self.size() * self.zoom_factor
            self.setPixmap(self.pixmap().scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))


class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event):
        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self.scale(factor, factor)


class ZoomWindow(QDialog):
    def __init__(self, image, capture_thread, parent_window):
        super().__init__()
        self.setWindowTitle("放大图像")
        self.setGeometry(100, 100, 1400, 800)
        self.image = image
        self.capture_thread = capture_thread  # 捕获线程的引用
        self.parent_window = parent_window  # MyWindow 窗口的引用
        self.detected_labels = []
        self.label_stats = {}
        self.alarm_conditions = []
        self.init_ui()
        self.capture_thread.alarmStatusChanged.connect(self.update_alarm_status)
        # self.capture_thread.detectedAlarmConditions.connect(self.update_alarm_conditions)

    def init_ui(self):
        self.button_style = """
                          QPushButton {
                              background-color: #89CFF0;
                              border-style: solid;
                              border-radius: 15px;
                              border-width: 2px;
                              border-color: #0057B7;
                              min-width: 90px;
                              max-width: 90px;
                              min-height: 90px;
                              max-height: 90px;
                          }
                          QPushButton:pressed {
                              background-color: #0057B7;
                              border-color: #002855;
                          }
                          QPushButton:hover {
                              border-width: 3px;
                          }
                      """
        self.palette = QPalette()  # 创建调色板
        self.setPalette(self.palette)  # 将调色板应用于窗口

        self.graphics_view = ZoomableGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(self.image))
        self.scene.addItem(self.pixmap_item)
        self.graphics_view.setScene(self.scene)

        # 添加保存和取消按钮
        confirm_btn = QPushButton("停止")
        confirm_btn.setStyleSheet(self.button_style)
        gather_btn = QPushButton("继续采集")
        gather_btn.setStyleSheet(self.button_style)
        save_button = QPushButton("保存")
        save_button.setStyleSheet(self.button_style)
        cancel_button = QPushButton("取消")
        cancel_button.setStyleSheet(self.button_style)

        # self.location = QLineEdit("检测位置")
        # self.people = QLineEdit("巡检人员")
        # self.damageDesc = QLineEdit("病害位置描述")
        self.location_lxt = QLineEdit("病害位置")
        # self.description_des = QLineEdit("病害描述")
        # self.disposal_dis = QLineEdit("是否处置")
        self.disposal_1 = QLineEdit("处理措施")
        # 将 "是否处置" 改为 QComboBox
        self.disposal_dis = QComboBox()
        self.disposal_dis.addItems(["否", "是"])
        self.disposal_dis.currentIndexChanged.connect(self.update_disposal)
        self.disposal_1 = QLineEdit("处理措施")
        line_edit_style = """
            QLineEdit {
                font-size: 14px;  /* 设置字体大小 */
                height: 50px;     /* 设置高度 */
            }
        """
        # self.location.setStyleSheet(line_edit_style)
        # self.people.setStyleSheet(line_edit_style)
        # self.damageDesc.setStyleSheet(line_edit_style)
        self.location_lxt.setStyleSheet(line_edit_style)
        # self.description_des.setStyleSheet(line_edit_style)
        self.disposal_dis.setStyleSheet(line_edit_style)
        self.disposal_1.setStyleSheet(line_edit_style)

        # 保存按钮点击事件
        save_button.clicked.connect(self.save_image)
        # 绑定按钮点击事件
        confirm_btn.clicked.connect(self.stop_capture)
        gather_btn.clicked.connect(self.resume_capture)
        # 取消按钮点击事件
        cancel_button.clicked.connect(self.reject)

        # 创建按钮布局
        button_layout = QHBoxLayout()
        button_layout.addWidget(confirm_btn)
        button_layout.addWidget(gather_btn)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        label_layout = QHBoxLayout()
        # label_layout.addWidget(self.location)

        # label_layout.addWidget(self.people)
        label_layout.addWidget(self.location_lxt)
        # label_layout.addWidget(self.description_des)
        label_layout.addWidget(self.disposal_dis)
        label_layout.addWidget(self.disposal_1)

        # 创建主布局
        layout = QVBoxLayout()
        layout.addWidget(self.graphics_view)
        layout.addLayout(button_layout)
        layout.addLayout(label_layout)

        self.setLayout(layout)

    # def update_alarm_conditions(self, alarm_conditions):
    #     self.alarm_conditions = alarm_conditions
    def update_disposal(self):
        if self.disposal_dis.currentText() == "是":
            self.disposal_1.setText("加固")
        else:
            self.disposal_1.clear()  # 当选择为 "否" 时清空处理措施

    def update_alarm_status(self, is_alarm):
        if is_alarm:
            self.palette.setColor(QPalette.Window, QColor(Qt.red))
        else:
            self.palette.setColor(QPalette.Window, QColor(Qt.white))
        self.setPalette(self.palette)
        self.repaint()  # 立即请求重绘

    def stop_capture(self):
        self.capture_thread.stopCapture.emit()  # 发射停止采集信号

    def resume_capture(self):
        self.capture_thread.resumeCapture.emit()  # 发射恢复采集信号

    def set_detected_labels(self, labels):
        self.detected_labels = labels

    def set_label_stats(self, label_stats):
        self.label_stats = label_stats

    def set_model_name(self, model_name):
        self.model_name = model_name

    def save_image(self):
        global global_lc_lf_conditions  # 引用全局变量
        global global_lc_cf_conditions  # 引用全局变量
        current_date = QDateTime.currentDateTime().toString("yyyyMMddHHmmss")
        current_datd = datetime.now().strftime('%Y%m%d')

        people_text = self.parent_window.get_last_inspector()
        location_text = self.location_lxt.text()
        # description_text = self.description_des.text()
        disposal_text = 0 if self.disposal_dis.currentText() == "是" else 1
        disposal1_text = self.disposal_1.text()
        create_time = current_date

        selected_options = self.parent_window.get_selected_options()
        line_name = selected_options["line_name"]
        station_name = selected_options["station_name"]
        folder_path = f"{line_name}-{station_name}-{current_datd}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        current_time = QDateTime.currentDateTime().toString("yyyy-MMdd-hhmmss")
        image_file_name = f"{current_time}.png"
        image_save_path = os.path.join(folder_path, image_file_name)
        self.image.save(image_save_path)

        ceiling_type_mapping = {
            '铝板': 1,
            '栅格': 2,
            '冲孔板': 3,
            '拉网': 4,
            '铝条板': 5,
            '栅格加网': 6,
            '铝方通': 7,
            '矿棉板': 8,
            '勾搭铝板': 9,
        }
        ceiling_type_number = ceiling_type_mapping.get(self.model_name, 0)

        damage_type = []
        damage_data = []

        # 初始化病害描述
        description_parts = []

        # 处理 global_lc_lf_conditions
        for condition in global_lc_lf_conditions:
            lc_value = float(condition['LC']) if isinstance(condition['LC'], (np.float32, str)) else condition['LC']
            lf_value = float(condition['LF']) if isinstance(condition['LF'], (np.float32, str)) else condition['LF']

            # 如果 LF 大于 10 厘米，则保存为 QS
            if lf_value > 10:
                if "QS" not in damage_type:
                    damage_type.append("QS")
                damage_data.append(f"QS:{lf_value:.2f}")  # 保留两位小数

            # 如果 LC 大于 3 且 LF 不为 0.0 且 LF 小于等于 10，则保存 LC 和 LF
            elif lc_value > 3 and lf_value > 3:
                if "LC" not in damage_type:
                    damage_type.append("LC")
                if "LF" not in damage_type:
                    damage_type.append("LF")
                damage_data.append(f"LC:{lc_value:.2f}, LF:{lf_value:.2f}")  # 保留两位小数
                description_parts.append(f"大于三毫米的LC: {lc_value:.2f}, 大于三毫米的LF: {lf_value:.2f}")
            # 如果 LC 大于 3 且 LF 为 0.0，则只保存 LC
            elif lc_value > 3 and lf_value < 3:
                if "LC" not in damage_type:
                    damage_type.append("LC")
                damage_data.append(f"LC:{lc_value:.2f}")  # 保留两位小数
                description_parts.append(f"大于三毫米的LC: {lc_value:.2f}")
            # 如果 LC 为 0.0 且 LF 大于 3 且 LF 小于等于 10，则只保存 LF
            elif lc_value < 3 and lf_value > 3 and lf_value <= 10:
                if "LF" not in damage_type:
                    damage_type.append("LF")
                damage_data.append(f"LF:{lf_value:.2f}")  # 保留两位小数
                description_parts.append(f"大于三毫米的LF: {lf_value:.2f}")
        # 处理 global_lc_cf_conditions
        for condition in global_lc_cf_conditions:
            lc_value = float(condition['LC'].split()[1]) if isinstance(condition['LC'], str) else float(condition['LC'])
            cf_value = float(condition['CF'].split()[1]) if isinstance(condition['CF'], str) else float(condition['CF'])

            # 如果 LC 大于 3 且 CF 不为 0.0，则保存 LC 和 CF
            if lc_value > 3 and cf_value > 3:
                if "LC" not in damage_type:
                    damage_type.append("LC")
                if "CF" not in damage_type:
                    damage_type.append("CF")
                damage_data.append(f"LC:{lc_value:.2f}, CF:{cf_value:.2f}")  # 保留两位小数
                description_parts.append(f"大于三毫米的LC: {lc_value:.2f}, 大于三毫米的CF: {cf_value:.2f}")
            # 如果 LC 大于 3 且 CF 为 0.0，则只保存 LC
            elif lc_value > 3 and cf_value < 3:
                if "LC" not in damage_type:
                    damage_type.append("LC")
                damage_data.append(f"LC:{lc_value:.2f}")  # 保留两位小数
                description_parts.append(f"大于三毫米的LC: {lc_value:.2f}")
            # 如果 LC 为 0.0 且 CF 大于 3，则只保存 CF
            elif lc_value < 3 and cf_value > 3:
                if "CF" not in damage_type:
                    damage_type.append("CF")
                damage_data.append(f"CF:{cf_value:.2f}")  # 保留两位小数
                description_parts.append(f"大于三毫米的CF: {cf_value:.2f}")
        # 只有在 damageType 和 damageData 不为空时才保存数据
        if damage_type and damage_data:
            damage_type_str = ", ".join(damage_type)  # 使用描述而非编号
            damage_data_str = ", ".join(damage_data)  # 将所有符合条件的数据合并为一个字符串

            description_str = ", ".join(description_parts)  # 将列表转换为逗号分隔的字符串

            new_data_entry = {
                "inspector": people_text,
                "line_id": selected_options["line_id"],
                "line_name": line_name,
                "station_id": selected_options["station_id"],
                "station_name": station_name,
                "celling_id": selected_options["celling_id"],
                "celling_name": selected_options["celling_name"],
                "depart_id": selected_options["depart_id"],
                "depart_name": selected_options["depart_name"],
                "inspection_time": create_time,
                "damageType": damage_type_str,  # 保存类别LC, LF, CF, QS
                "damageDetail": description_str,  # 这里将description_parts转换为字符串
                "damageLocation": location_text,
                "picName": image_file_name,
                "ceilingType": ceiling_type_number,
                "isDisposed": disposal_text,
                "measures": disposal1_text,
                "damageData": damage_data_str  # 保存符合条件的数值数据
            }

            self.parent_window.db_cursor.execute('''
                INSERT INTO inspections (inspector, line_id, line_name, station_id, station_name, celling_id, celling_name, 
                depart_id, depart_name, inspection_time, damageType, damageDetail, damageLocation,picName, ceilingType, isDisposed, measures, damageData)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                new_data_entry["inspector"],
                new_data_entry["line_id"],
                new_data_entry["line_name"],
                new_data_entry["station_id"],
                new_data_entry["station_name"],
                new_data_entry["celling_id"],
                new_data_entry["celling_name"],
                new_data_entry["depart_id"],
                new_data_entry["depart_name"],
                new_data_entry["inspection_time"],
                new_data_entry["damageType"],
                new_data_entry["damageDetail"],  # 插入转换后的字符串
                new_data_entry["damageLocation"],
                new_data_entry["picName"],
                new_data_entry["ceilingType"],
                new_data_entry["isDisposed"],
                new_data_entry["measures"],
                new_data_entry["damageData"]
            ))
            self.parent_window.db_connection.commit()

        global_lc_lf_conditions = []  # 清空全局变量列表
        global_lc_cf_conditions = []  # 清空全局变量列表
        self.accept()


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 初始化选择项
        self.selected_line = ""
        self.selected_station = ""
        self.selected_area = ""
        self.selected_center = ""
        button_font_size = "25px"
        button_width = 200
        button_height = 80
        # 获取 Dracula 颜色
        dracula = DraculaColors()
        self.initial_measurements_to_discard = 5  # 需要丢弃的初始测量次数
        self.current_measurement_count = 0  # 当前丢弃的测量次数
        self.device = Device()
        self.setWindowTitle("SDK")

        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)

        self.setGeometry(100, 100, 800, 600)
        self.data = CaptureAllData()
        self.save_images_flag = False
        self.t_name = ""
        self.num_collect = 0
        self.is_measurement_started = False  # 初始化is_measurement_started
        self.is_measurement_paused = False  # 初始化is_measurement_paused
        self.model_combo = CustomComboBox()
        self.setStyleSheet(f"""

                   QWidget {{
                       background-color: {dracula.background};
                       color: {dracula.foreground};
                   }}
                    QComboBox {{
                        border: 2px solid #333;  /* 边框颜色和厚度 */
                        border-radius: 15px;     /* 边框圆角的半径 */
                        padding: 5px 15px;       /* 内填充，格式为：上下 左右 */
                        background-color: #222;  /* 背景颜色 */
                        color: #fff;             /* 文字颜色 */
                        min-width: 6em;          /* 最小宽度 */
                        height: 40px;            /* 高度 */
                    }}

                    QComboBox::drop-down {{
                        subcontrol-origin: padding;
                        subcontrol-position: center right;
                        width: 40px;             /* 下拉箭头的宽度 */
                        border-left-width: 1px;
                        border-left-color: #555;
                        border-left-style: solid; /* 左边界的样式 */
                        border-top-right-radius: 13px;  /* 右上角的圆角 */
                        border-bottom-right-radius: 13px;  /* 右下角的圆角 */
                    }}

                    QComboBox::down-arrow {{

                        width: 20px;
                        height: 20px;
                    }}

                    QComboBox::hover {{
                        border-color: #555;      /* 鼠标悬停时边框颜色 */
                    }}
                   QPushButton {{
                       background-color: {dracula.yellow};
                       color: black;;
                       border: none;
                       padding: 20px;
                       border-radius: 20px;
                       font-size: {button_font_size};
                       button_width:{button_width};
                       button_height:{button_height};

                   }}
                   QPushButton:hover {{
                       background-color: {dracula.pink};
                   }}
                   QLabel {{
                        background-color: #222; /* 背景颜色 */
                        color: #fff;           /* 文字颜色 */
                        border: 2px solid #333;/* 边框颜色和厚度 */
                        border-radius: 15px;   /* 边框圆角的半径 */
                        padding: 5px 10px;     /* 内填充，格式为：上下 左右 */
                        font-size: 16px;       /* 字体大小 */
                        }}
                   QLabel#dataLabel {{
                       background-color: #333; /* 深色背景 */
                        color: #fff;           /* 白色文字 */
                        border: 2px solid #555;/* 边框颜色和厚度 */
                        border-radius: 15px;   /* 边框圆角的半径 */
                        padding: 5px 10px;     /* 内填充，格式为：上下 左右 */
                        font-size: 16px;       /* 字体大小 */
                        text-align: center;    /* 文字居中 */
                   }}
               """)
        layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        label_layout = QHBoxLayout()
        tag_layout = QHBoxLayout()

        sta_layout = QHBoxLayout()
        unit_layout = QHBoxLayout()
        work_layout = QHBoxLayout()
        image_layout = QHBoxLayout()
        alarm_layout = QHBoxLayout()
        layout.addLayout(tag_layout, 0)
        layout.addLayout(unit_layout)
        layout.addLayout(work_layout)
        layout.addLayout(alarm_layout)
        layout.addLayout(btn_layout)
        layout.addLayout(sta_layout)
        layout.addLayout(label_layout)
        # 初始化字体设置
        font = QFont("Arial", 15)  # 设置字体为Arial，大小为20

        self.depth_differences = []
        self.line_label = QLabel("选择线路")
        self.line_label.setAlignment(Qt.AlignCenter)
        self.line_label.setFrameStyle(QLabel.Panel | QLabel.Sunken)
        self.line_label.setFixedSize(130, 30)
        self.line_label.mousePressEvent = self.update_lines

        self.line_combo = QComboBox()
        self.line_combo.setFont(font)  # 设置下拉框的字体

        self.station_label = QLabel("选择车站")
        self.station_label.setAlignment(Qt.AlignCenter)
        self.station_label.setFrameStyle(QLabel.Panel | QLabel.Sunken)
        self.station_label.setFixedSize(130, 30)
        self.station_label.mousePressEvent = self.update_stations
        self.station_combo = QComboBox()
        self.station_combo.setFont(font)  # 设置下拉框的字体
        self.area_label = QLabel("选择区域")
        self.area_label.setAlignment(Qt.AlignCenter)
        self.area_label.setFrameStyle(QLabel.Panel | QLabel.Sunken)
        self.area_label.setFixedSize(130, 30)
        self.area_label.mousePressEvent = self.update_areas
        self.area_combo = QComboBox()
        self.area_combo.setFont(font)  # 设置下拉框的字体
        # 添加选项：站台、站厅、通道

        self.line_combo.setFixedSize(160, 40)  # 设置固定大小为宽度 160，高度 40
        self.station_combo.setFixedSize(160, 40)
        self.area_combo.setFixedSize(160, 40)
        tag_layout.addWidget(self.line_label, 0, Qt.AlignTop)
        tag_layout.addWidget(self.line_combo, 0, Qt.AlignTop)
        tag_layout.addWidget(self.station_label, 0, Qt.AlignTop)
        tag_layout.addWidget(self.station_combo, 0, Qt.AlignTop)
        tag_layout.addWidget(self.area_label, 0, Qt.AlignTop)
        tag_layout.addWidget(self.area_combo, 0, Qt.AlignTop)

        self.centre_label = QLabel("中心")
        self.centre_label.setAlignment(Qt.AlignCenter)
        self.centre_label.setFrameStyle(QLabel.Panel | QLabel.Sunken)
        self.centre_label.setFixedSize(130, 30)
        self.centre_label.mousePressEvent = self.update_centers

        self.center_combo = QComboBox()
        self.center_combo.setFont(font)  # 设置下拉框的字体

        unit_layout.addWidget(self.centre_label)
        unit_layout.addWidget(self.center_combo)

        self.centre_label1 = QLabel("工区")
        self.centre_label1.setAlignment(Qt.AlignCenter)
        self.centre_label1.setFrameStyle(QLabel.Panel | QLabel.Sunken)
        self.centre_label1.setFixedSize(130, 30)

        self.area_combo2 = QComboBox()
        # 添加选项：站台、站厅、通道
        area_options = ["工区1", "工区2", "工区3"]
        self.area_combo2.setFont(font)  # 设置下拉框的字体
        self.area_combo2.addItems(area_options)

        unit_layout.addWidget(self.centre_label1)
        unit_layout.addWidget(self.area_combo2)

        self.centre_label2 = QLabel("吊顶样式")
        self.centre_label2.setAlignment(Qt.AlignCenter)
        self.centre_label2.setFrameStyle(QLabel.Panel | QLabel.Sunken)
        self.centre_label2.setFixedSize(130, 30)
        self.centre_label2.mousePressEvent = self.update_ceiling_types

        self.ceiling_type_combo = QComboBox()
        self.ceiling_type_combo.setFont(font)  # 设置下拉框的字体
        self.area_combo2.setFixedSize(160, 40)  # 设置固定大小为宽度 160，高度 40
        self.center_combo.setFixedSize(160, 40)
        self.ceiling_type_combo.setFixedSize(160, 40)
        unit_layout.addWidget(self.centre_label2)
        unit_layout.addWidget(self.ceiling_type_combo)

        self.model_label = QLabel("吊顶类型")
        self.model_label.setAlignment(Qt.AlignCenter)
        self.model_label.setFrameStyle(QLabel.Panel | QLabel.Sunken)
        self.model_label.setFixedSize(130, 30)

        self.model_combo = QComboBox()
        # 向下拉框添加模型选项
        model_options = ["补丁铝板", "铝板", "栅格", "反光冲孔板", "冲孔板", "勾搭铝板", "拉网", "铝条板", "栅格加网", "铝方通", "矿棉板", "垂片", "高铝板"]
        self.model_combo.setFixedSize(160, 40)
        self.model_combo.setFont(font)  # 设置下拉框的字体
        self.model_combo.addItems(model_options)
        self.model_combo.currentIndexChanged.connect(self.update_selected_model)
        work_layout.addWidget(self.model_label)
        work_layout.addWidget(self.model_combo)
        # 创建一个水平布局来包含选项标签和图片标签
        item_layout = QHBoxLayout()
        image_layout.addLayout(item_layout)

        # 将图片布局添加到工作布局中
        work_layout.addLayout(image_layout)
        self.capture_interval_label = QLabel("采集间隔（毫秒）")
        self.capture_interval_label.setAlignment(Qt.AlignCenter)
        self.capture_interval_label.setFrameStyle(QLabel.Panel | QLabel.Sunken)
        self.capture_interval_label.setFixedSize(130, 30)

        self.capture_interval_combo = QComboBox()
        interval_options = ["10000", "12000", "15000", "20000", "25000", "35000"]  # 采集时间间隔选项，以毫秒为单位
        self.capture_interval_combo.setFont(font)  # 设置下拉框的字体
        self.capture_interval_combo.setFixedSize(160, 40)
        self.capture_interval_combo.addItems(interval_options)
        self.capture_interval_combo.setCurrentIndex(0)  # 默认选择第一个选项，即10秒（10000毫秒）
        self.capture_interval_combo.currentIndexChanged.connect(self.update_capture_interval)

        work_layout.addWidget(self.capture_interval_label)
        work_layout.addWidget(self.capture_interval_combo)

        self.capture_interval_label1 = QLabel("曝光度")
        self.capture_interval_label1.setAlignment(Qt.AlignCenter)
        self.capture_interval_label1.setFrameStyle(QLabel.Panel | QLabel.Sunken)
        self.capture_interval_label1.setFixedSize(130, 30)
        self.exposure_combo = QComboBox()
        self.exposure_combo.setFont(font)  # 设置下拉框的字体
        exposure_options = ["40", "60", "80", "100", "120", "140", "160"]  # 曝光度选项
        self.exposure_combo.setFixedSize(160, 40)
        self.exposure_combo.addItems(exposure_options)
        self.exposure_combo.currentIndexChanged.connect(self.capture_images)
        work_layout.addWidget(self.capture_interval_label1)
        work_layout.addWidget(self.exposure_combo)

        self.alarm_label1 = QLabel("报警类型及信息")
        self.alarm_label1.setAlignment(Qt.AlignCenter)
        self.alarm_label1.setFrameStyle(QLabel.Panel | QLabel.Sunken)
        self.alarm_label1.setFixedSize(130, 30)
        alarm_layout.addWidget(self.alarm_label1)
        self.data_display2 = QLineEdit()
        self.data_display2.setFixedSize(160, 40)
        alarm_layout.addWidget(self.data_display2)

        self.db_connection = sqlite3.connect("D:/PythonCode/sql/example.db")
        self.db_cursor = self.db_connection.cursor()
        self.image_label = ZoomableImageLabel()
        self.image_label.setFixedSize(400, 350)
        self.image_label.mouseDoubleClickEvent = lambda event: self.show_zoom_window(self.image_label)
        label_layout.addWidget(self.image_label)

        self.depth_label = ZoomableImageLabel()
        self.depth_label.setFixedSize(400, 350)
        self.depth_label.mouseDoubleClickEvent = lambda event: self.show_zoom_window(self.depth_label)
        label_layout.addWidget(self.depth_label)

        self.image_label2 = ZoomableImageLabel()
        self.image_label2.setFixedSize(400, 350)
        self.image_label2.mouseDoubleClickEvent = lambda event: self.show_zoom_window(self.image_label2)
        label_layout.addWidget(self.image_label2)

        self.data_label = QLabel("采集数量: 0")
        sta_layout.addWidget(self.data_label)
        self.data_label.setObjectName("dataLabel")

        self.line = QPushButton("启动")
        # self.line.setFixedHeight(40)  # 设置按钮的高度为 40 像素
        # self.line.setStyleSheet(self.button_style)
        self.gather = QPushButton("采集")
        # self.gather.setFixedHeight(40)
        # self.gather.setStyleSheet(self.button_style)
        self.stop_1 = QPushButton("停止")
        # self.stop_1.setFixedHeight(40)
        # self.stop_1.setStyleSheet(self.button_style)
        # self.storage = QPushButton("存储")
        # self.storage.setFixedHeight(40)
        # self.storage.setStyleSheet(self.button_style)
        # self.analyse = QPushButton("分析")
        # self.analyse.setFixedHeight(40)
        # self.analyse.setStyleSheet(self.button_style)
        # self.pack = QPushButton("打包")
        # self.update_btn = QPushButton("更新")
        self.export_btn = QPushButton("导出")
        # self.pack.setFixedHeight(40)
        # self.pack.setStyleSheet(self.button_style)

        btn_layout.addWidget(self.line)
        btn_layout.addWidget(self.gather)
        btn_layout.addWidget(self.stop_1)
        # btn_layout.addWidget(self.storage)
        # btn_layout.addWidget(self.anal
        # btn_layout.addWidget(self.update_btn)
        btn_layout.addWidget(self.export_btn)

        self.line.clicked.connect(self.line_1)
        self.gather.clicked.connect(self.start_timer)
        self.stop_1.clicked.connect(self.stop_timer)
        # self.update_btn.clicked.connect(self.update_data)
        self.export_btn.clicked.connect(self.export_data)
        # self.update_btn.connect()
        # self.storage.clicked.connect(self.save_images)
        # self.analyse.clicked.connect(self.analysis_completed)


        layout.addLayout(btn_layout)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_measurement_data)
        self.image_label.installEventFilter(self)
        self.depth_label.installEventFilter(self)

        self.result_images = []  # 存储分析结果图像
        self.current_result_index = 0
        self.model_combo.setCurrentIndex(0)  # 设置吊顶类型下拉框为铝板
        self.update_selected_model(0)  # 更新选中的模型，这会触发显示对应图片的功能
        self.connected = False
        self.current_station_dir = ""
        self.your_class_instance = YourClass()
        self.already_warned = False
        self.capture_thread = ImageCaptureThread(self.device, self)
        main_widget = QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        # self.received_data = []
        self.alarm_label4 = QLabel("测距仪数据")
        self.alarm_label4.setAlignment(Qt.AlignCenter)
        self.alarm_label4.setFrameStyle(QLabel.Panel | QLabel.Sunken)
        self.alarm_label4.setFixedSize(130, 30)
        alarm_layout.addWidget(self.alarm_label4)
        self.data_display = QTextEdit()
        self.data_display.setPlainText("这里显示数据")
        self.data_display.setFixedSize(160, 40)
        alarm_layout.addWidget(self.data_display)
        # self.button_2 = QPushButton("HEX")
        # self.button_2.setStyleSheet(self.button_style)
        # self.button_2.clicked.connect(self.on_read_measurement)
        # btn_layout.addWidget(self.button_2)
        self.button_3 = QPushButton("调整相机高度")
        self.button_4 = QPushButton("完成采集")
        # self.button_3.setStyleSheet(self.button_style)
        self.button_3.clicked.connect(self.on_start_measurement)
        self.button_4.clicked.connect(self.on_start_measurement1)
        # self.button_4 = QPushButton("打开测距激光")
        # # self.button_4.setStyleSheet(self.button_style)
        # self.button_4.clicked.connect(self.on_laser_on)
        # btn_layout.addWidget(self.button_4)
        # btn_layout.addWidget(self.button_3)
        # self.button_5 = QPushButton("关闭测距激光")
        # self.button_5.setStyleSheet(self.button_style)
        # self.button_5.clicked.connect(self.on_laser_off)
        # btn_layout.addWidget(self.button_5)
        # self.button_1 = QPushButton("连接测距仪")
        # self.button_1.setStyleSheet(self.button_style)
        # self.button_1.clicked.connect(self.connect_or_disconnect)
        btn_layout.addWidget(self.button_3)
        btn_layout.addWidget(self.line)
        btn_layout.addWidget(self.gather)
        btn_layout.addWidget(self.stop_1)
        # btn_layout.addWidget(self.storage)
        # btn_layout.addWidget(self.analyse)
        # btn_layout.addWidget(self.pack)
        # btn_layout.addWidget(self.update_btn)
        btn_layout.addWidget(self.export_btn)
        # btn_layout.addWidget(self.button_1)
        btn_layout.addWidget(self.button_4)
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addSpacerItem(spacer)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_measurement_data)
        self.timer.start(1000)

        self.alarm_label3 = QLabel("巡检人员")
        self.alarm_label3.setAlignment(Qt.AlignCenter)
        self.alarm_label3.setFrameStyle(QLabel.Panel | QLabel.Sunken)
        self.alarm_label3.setFixedSize(130, 30)
        alarm_layout.addWidget(self.alarm_label3)

        self.inspector_input = QLineEdit()  # 创建一个新的输入框用于输入巡检人员
        self.inspector_input.setFixedSize(160, 30)  # 设置输入框的大小
        self.inspector_input.textChanged.connect(self.update_inspector_info)  # 绑定信号
        alarm_layout.addWidget(self.inspector_input)  # 将巡检人员输入框添加到布局中


        # 绑定下拉框变化事件
        self.line_combo.currentIndexChanged.connect(self.update_selected_options)
        self.station_combo.currentIndexChanged.connect(self.update_selected_options)
        self.area_combo.currentIndexChanged.connect(self.update_selected_options)
        self.center_combo.currentIndexChanged.connect(self.update_selected_options)
        tag_layout.setSpacing(1)  # 减小间隔

        self.fixed_port_name = 'COM3'  # 请将此处替换为你的实际串口名称，如 'COM3' 或 '/dev/ttyUSB0'
        self.baudrate = 19200
        self.data_bits = serial.EIGHTBITS
        self.stop_bits = serial.STOPBITS_ONE
        self.parity = serial.PARITY_NONE
        self.flow_control = False

        self.connected = False
        self.serial_port = None

        # 自动连接到固定串口
        self.auto_connect_measurement_device()

    def update_inspector_info(self, text):
        """保存巡检人员输入信息"""
        self.last_inspector = text

    def get_last_inspector(self):
        """获取上次输入的巡检人员信息"""
        return self.last_inspector

    def on_start_measurement1(self):

        self.your_class_instance.relay3_state_changed(True)
        time.sleep(0.2)
        self.your_class_instance.relay4_state_changed(True)
        time.sleep(37)
        self.your_class_instance.relay1_state_changed(False)
        time.sleep(0.2)
        self.your_class_instance.relay2_state_changed(False)
        time.sleep(0.2)

        self.your_class_instance.relay3_state_changed(False)
        time.sleep(0.2)

        self.your_class_instance.relay4_state_changed(False)


    def update_selected_options(self):
        self.selected_line = self.line_combo.currentText()
        self.selected_station = self.station_combo.currentText()
        self.selected_area = self.area_combo.currentText()
        self.selected_center = self.center_combo.currentText()


    def get_selected_options(self):
        # 在选择完线路、车站、区域和中心后，获取这些选择项的详细信息
        line_id, line_name = self.get_line_details(self.selected_line)
        station_id, station_name = self.get_station_details(self.selected_station)
        celling_id, celling_name = self.get_area_details(self.selected_area)
        depart_id, depart_name = self.get_center_details(self.selected_center)

        return {
            "line_id": line_id,
            "line_name": line_name,
            "station_id": station_id,
            "station_name": station_name,
            "celling_id": celling_id,
            "celling_name": celling_name,
            "depart_id": depart_id,
            "depart_name": depart_name
        }


    def get_line_details(self, line_name):
        # 根据 line_name 从数据库或其他来源获取 line_id 和 line_name
        self.db_cursor.execute("SELECT line_id, line_name FROM facilities WHERE line_name=?", (line_name,))
        result = self.db_cursor.fetchone()
        return (result[0], line_name) if result else (None, line_name)

    def get_station_details(self, station_name):
        self.db_cursor.execute("SELECT station_id, station_name FROM facilities WHERE station_name=?", (station_name,))
        result = self.db_cursor.fetchone()
        return (result[0], station_name) if result else (None, station_name)

    def get_area_details(self, area_name):
        selected_line = self.line_combo.currentText()
        selected_station = self.station_combo.currentText()
        self.db_cursor.execute(
            "SELECT celling_id FROM facilities WHERE line_name=? AND station_name=? AND celling_name=?",
            (selected_line, selected_station, area_name))
        result = self.db_cursor.fetchone()
        return (result[0], area_name) if result else (None, area_name)

    def get_center_details(self, center_name):
        # 根据 center_name 从数据库或其他来源获取 center_id 和 center_name
        self.db_cursor.execute("SELECT depart_id FROM facilities WHERE depart_name=?", (center_name,))
        result = self.db_cursor.fetchone()
        return result[0], center_name if result else (None, center_name)


    def load_serial_ports(self):
        ports = list(serial.tools.list_ports.comports())
        port_names = [port.device for port in ports]
        self.serial_port_combo.clear()
        self.serial_port_combo.addItems(port_names)


    def auto_connect_measurement_device(self):
        try:
            self.serial_port = serial.Serial(
                port=self.fixed_port_name,
                baudrate=self.baudrate,
                bytesize=self.data_bits,
                stopbits=self.stop_bits,
                parity=self.parity,
                rtscts=self.flow_control,
                xonxoff=self.flow_control
            )
            if self.serial_port.is_open:
                QMessageBox.information(self, "连接成功", f"已成功连接到串口 {self.fixed_port_name}")
                self.connected = True
            else:
                raise serial.SerialException("串口打开失败")
        except serial.SerialException as e:
            print(f"无法连接到串口 {self.fixed_port_name}：{str(e)}")
            QMessageBox.warning(self, "连接失败", f"无法连接到串口 {self.fixed_port_name}")


    def connect_or_disconnect(self):
        if not self.connected:
            self.auto_connect_measurement_device()
        else:
            self.disconnect_serial()


    def disconnect_serial(self):
        if self.connected and self.serial_port:
            try:
                self.serial_port.close()
                QMessageBox.information(self, "断开连接", "已成功断开串口连接")
                self.connected = False
            except Exception as e:
                QMessageBox.warning(self, "断开连接失败", f"断开串口连接时出错：{str(e)}")


    def send_command_1(self, command):
        try:
            if self.connected and self.serial_port.is_open:
                self.serial_port.write(bytes.fromhex(command))
                time.sleep(0.1)
                response = self.serial_port.read_all()
                return response
            else:
                QMessageBox.warning(self, "错误", "串口未打开，请先连接串口")
                return b''
        except Exception as e:
            QMessageBox.warning(self, "发送命令失败", f"发送命令时出错：{str(e)}")
            return b''


    def laser_on(self):
        try:
            command = "AA 00 01 BE 00 01 00 01 C1"
            self.response = self.send_command_1(command)
            print("激光已开启")
        except Exception as e:
            print("错误：", e)


    def laser_off(self):
        try:
            command = "AA 00 01 BE 00 01 00 00 C0"
            self.response = self.send_command_1(command)
            print("激光已关闭")
        except Exception as e:
            print("错误：", e)


    def start_continuous_measurement(self):
        command = "AA 00 00 20 00 01 00 04 25"
        self.response = self.send_command_1(command)


    def read_measurement_result(self):
        command = "AA 80 00 22 A2"
        response = self.send_command_1(command)
        return response


    def on_laser_on(self):
        self.laser_on()


    def on_laser_off(self):
        self.laser_off()


    def on_start_measurement(self):
        # if not self.connected:
        #     self.auto_connect_measurement_device()
        # else:
        #     self.disconnect_serial()
        self.start_continuous_measurement()
        self.is_measurement_started = True
        self.is_measurement_paused = False
        self.current_measurement_count = 0  # 重置测量计数器


    def parse_measurement_data(self, data):
        print("测量数据字节串:", data.hex())
        distance_data = data[6:10]
        distance_int = int.from_bytes(distance_data, byteorder='big', signed=True)
        scaling_factor = 0.001
        distance_meters = distance_int * scaling_factor
        print("解析出的距离 (米):", distance_meters)
        return distance_meters


    def on_read_measurement(self):
        measurement_data = self.read_measurement_result()
        distance_meters = self.parse_measurement_data(measurement_data)
        print("测量距离 (米):", distance_meters)
        self.data_display.append(f"测量距离 (米): {distance_meters:.2f} 米")


    def update_measurement_data(self):
        try:
            if self.is_measurement_started and not self.is_measurement_paused:
                measurement_data = self.read_measurement_result()
                self.distance_meters = self.parse_measurement_data(measurement_data)

                # 检查是否需要丢弃初始测量数据
                if self.current_measurement_count < self.initial_measurements_to_discard:
                    self.current_measurement_count += 1
                    print(f"丢弃初始测量数据 {self.current_measurement_count}/{self.initial_measurements_to_discard}")
                    return

                self.data_display.append(f"测量距离 (米): {self.distance_meters:.2f} 米")

                # 使用一个小范围来比较浮点数
                tolerance = 0.05
                if self.distance_meters > 1.5 + tolerance:
                    self.your_class_instance.relay1_state_changed(True)
                    time.sleep(0.1)
                    self.your_class_instance.relay2_state_changed(True)
                    time.sleep(0.1)
                    self.your_class_instance.relay3_state_changed(False)
                    time.sleep(0.1)
                    self.your_class_instance.relay4_state_changed(False)
                elif self.distance_meters < 1.5 - tolerance:
                    self.your_class_instance.relay1_state_changed(False)
                    time.sleep(0.1)
                    self.your_class_instance.relay2_state_changed(False)
                    time.sleep(0.1)
                    self.your_class_instance.relay3_state_changed(True)
                    time.sleep(0.1)
                    self.your_class_instance.relay4_state_changed(True)
                else:
                    # 当距离在 1.5米 ± 容差 范围内时
                    self.turn_off_all_relays()
                    QMessageBox.information(self, "测量停止", "测距仪已测得1.5米，测量已停止。")
                    self.is_measurement_paused = True
                time.sleep(0.1)
        except Exception as e:
            print("更新测量数据时发生错误:", str(e))


    def turn_off_all_relays(self):
        self.your_class_instance.relay1_state_changed(False)
        time.sleep(0.1)
        self.your_class_instance.relay2_state_changed(False)
        time.sleep(0.1)
        self.your_class_instance.relay3_state_changed(False)
        time.sleep(0.1)
        self.your_class_instance.relay4_state_changed(False)


    def on_stop_measurement(self):
        self.is_measurement_started = False
        self.is_measurement_paused = False
        self.timer.stop()
        print("测量已停止。")

    def update_capture_interval(self):
        self.selected_interval = int(self.capture_interval_combo.currentText())  # 获取用户选择的采集时间间隔（毫秒）
        print("采集间隔更新为:", self.selected_interval)

    # def pack_1(self):
    #     # 弹出文件夹选择对话框，允许选择多个文件夹
    #     folders = QFileDialog.getExistingDirectory(self, "选择文件夹", "")
    #
    #     if folders:
    #         # 将选定的文件夹路径转换为列表，以便处理多个文件夹
    #         folders = [folders]
    #
    #         for folder in folders:
    #             if os.path.isdir(folder):
    #                 # 提取文件夹的基本名称作为压缩文件的名称
    #                 folder_name = os.path.basename(folder)
    #
    #                 # 设置压缩文件的输出路径和名称
    #                 zip_path = f"{folder_name}.zip"
    #
    #                 # 创建一个ZIP文件
    #                 with ZipFile(zip_path, 'w') as zipf:
    #                     # 遍历文件夹中的所有文件，并将它们添加到ZIP文件中
    #                     for root, dirs, files in os.walk(folder):
    #                         for file in files:
    #                             file_path = os.path.join(root, file)
    #                             # 将文件夹名称附加到arcname的前面
    #                             arcname = os.path.join(folder_name, os.path.relpath(file_path, start=folder))
    #                             zipf.write(file_path, arcname=arcname)
    #
    #         QMessageBox.information(self, "成功", "ZIP文件创建成功!")
    #     else:
    #         QMessageBox.warning(self, "警告", "未选择文件夹，打包操作已取消。")

    def loadWorkAreas(self):
        try:
            self.db_cursor.execute("SELECT AreaName FROM WorkAreas ORDER BY AreaName ASC")
            work_areas = [row[0] for row in self.db_cursor.fetchall()]
            self.area_combo2.clear()  # 清除旧的选项
            self.area_combo2.addItems(work_areas)  # 加载新的选项
        except Exception as e:
            print("Error loading work areas: ", e)

    def loadInspectors(self):
        try:
            self.db_cursor.execute("SELECT Name FROM Inspectors ORDER BY Name ASC")
            inspectors = [row[0] for row in self.db_cursor.fetchall()]
            self.area_combo3.clear()  # 清除旧的选项
            self.area_combo3.addItems(inspectors)  # 加载新的选项
        except Exception as e:
            print("Error loading inspectors: ", e)

    def update_detection_result(self, detection_img):
        # 在 image_label2 中显示检测结果
        self.image_label2.setPixmap(QPixmap.fromImage(detection_img))
        self.image_label2.setAlignment(Qt.AlignCenter)

    def update_depth_label(self, depth_text):
        self.depth_label2.setText(depth_text)
        # self.calculate_and_display_difference()

    def update_depth_label1(self, depth_text1):
        self.depth_label3.setText(depth_text1)
        # self.calculate_and_display_difference()

    # def update_data(self):
    #     selected_line = self.line_combo.currentText()
    #     selected_station = self.station_combo.currentText()
    #     selected_area = self.area_combo.currentText()
    #
    #     if not selected_line or not selected_station or not selected_area:
    #         QMessageBox.warning(self, "选择错误", "请选择线路、车站和区域")
    #         return
    #
    #     try:
    #         # 从 file_load_log 表中获取最新的 log_id 和文件路径
    #         self.db_cursor.execute('''
    #            SELECT log_id, file_path FROM file_load_log
    #            WHERE status = 'Success' AND file_name LIKE '%.json'
    #            ORDER BY load_start_time DESC LIMIT 1
    #            ''')
    #         log_entry = self.db_cursor.fetchone()
    #
    #         if log_entry:
    #             log_id, json_file_path = log_entry
    #         else:
    #             QMessageBox.warning(self, "日志错误", "未找到成功加载的日志记录")
    #             return
    #
    #         # 读取最新的 JSON 文件
    #         if os.path.exists(json_file_path):
    #             with open(json_file_path, 'r', encoding='utf-8') as file:
    #                 json_data = json.load(file)
    #
    #             updated = False
    #             if "data" in json_data:
    #                 for entry in json_data["data"]:
    #                     if self.check_and_update_facility(entry, log_id):
    #                         updated = True
    #
    #             if updated:
    #                 QMessageBox.information(self, "更新成功", "数据已更新")
    #             else:
    #                 QMessageBox.warning(self, "未更新", "没有找到匹配的数据进行更新")
    #         else:
    #             QMessageBox.warning(self, "文件不存在", f"JSON文件 {json_file_path} 不存在")
    #
    #     except Exception as e:
    #         QMessageBox.warning(self, "更新错误", f"更新数据时发生错误: {str(e)}")

    def export_data(self):
        # 连接到SQLite数据库
        db_path = os.path.join("sql", "example.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 获取当前日期，格式为yyyyMMdd
        current_date = datetime.now().strftime('%Y%m%d')

        # 查询当前日期的数据
        query = f"""
               SELECT * FROM inspections
               WHERE substr(inspection_time, 1, 8) = '{current_date}'
           """
        cursor.execute(query)
        latest_data = cursor.fetchall()

        # 检查是否有数据返回
        if not latest_data:
            print("没有数据可导出")
            return

        # 使用字典存储按线路和车站分组的数据
        data_dict = defaultdict(list)

        for row in latest_data:
            line_name = row[3]
            station_name = row[5]
            key = f"{line_name}-{station_name}-{current_date}"

            # 处理 damageType 和 damageData 字段
            damage_types = [dt.strip() for dt in row[12].split(",")] if row[12] else []

            # 解析数据库中的 damageData 字符串
            raw_damage_data = row[19]
            damage_data = defaultdict(list)

            if raw_damage_data:
                for item in raw_damage_data.split(", "):
                    if ":" in item:  # 确保项中包含冒号
                        damage_type, value = item.split(":")
                        damage_type = damage_type.strip()  # 去除前后空格
                        value = value.strip().replace("}", "")  # 清理数据
                        try:
                            damage_data[damage_type].append(float(value))
                        except ValueError:
                            print(f"无法转换为浮点数的值：'{value}'，已跳过此项")

            # 重新组织 damageData 为需要的格式
            formatted_damage_data = {damage_type: damage_data.get(damage_type, []) for damage_type in damage_types}

            data = {
                "id": row[0],
                "inspector": row[1],
                "line_id": int(row[2]),  # 保持为整数
                "line_name": row[3],
                "station_id": int(row[4]),  # 保持为整数
                "station_name": row[5],
                "ceiling_id": int(row[6]),  # 保持为整数
                "ceiling_name": row[7],
                "depart_id": int(row[8]),
                "depart_name": row[9],
                "inspection_time": row[10],
                "damageType": row[12],
                "damageData": formatted_damage_data,
                "damageDetail": row[13],
                "damageLocation": row[14],
                "picName": row[15],
                "ceilingType": row[16],
                "isDisposed": row[17],
                "measures": row[18]
            }
            data_dict[key].append(data)

        # 创建并写入JSON文件
        for key, data_list in data_dict.items():
            folder_path = key  # 直接使用线路和车站信息作为文件夹名

            # 创建文件夹（如果不存在）
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # 将数据写入JSON文件
            file_path = os.path.join(folder_path, 'latest_data.json')
            with open(file_path, 'w', encoding='utf-8') as json_file:
                json.dump({"data": data_list}, json_file, ensure_ascii=False, indent=4)
            print(f"数据已保存到 {file_path}")

        # 压缩整个文件夹
        zip_file_path = f"{folder_path}.zip"
        shutil.make_archive(folder_path, 'zip', folder_path)

        print(f"压缩文件已创建：{zip_file_path}")

        # 关闭游标和连接
        cursor.close()
        conn.close()

    def check_and_update_facility(self, entry, log_id):
        """
        检查 JSON 文件中的数据是否与数据库中的数据一致，如果不一致则更新数据库。
        """
        line_id = int(entry['line_id'])
        station_id = int(entry['station_id'])
        celling_id = int(entry['celling_id'])
        depart_id = int(entry['depart_id'])

        # 查询数据库中的记录
        self.db_cursor.execute('''
           SELECT load_log_id, line_id, line_name, station_id, station_name, celling_id, celling_name, depart_id, depart_name, ceiling_texture
           FROM facilities
           WHERE line_id=? AND station_id=? AND celling_id=? AND depart_id=?
           ''', (line_id, station_id, celling_id, depart_id))
        db_record = self.db_cursor.fetchone()

        if db_record:
            db_load_log_id, db_line_id, db_line_name, db_station_id, db_station_name, db_celling_id, db_celling_name, db_depart_id, db_depart_name, db_ceiling_texture = db_record
            if (db_line_id != line_id or
                    db_station_id != station_id or
                    db_celling_id != celling_id or
                    db_depart_id != depart_id or
                    db_line_name != entry['line_name'] or
                    db_station_name != entry['station_name'] or
                    db_celling_name != entry['celling_name'] or
                    db_depart_name != entry['depart_name'] or
                    db_ceiling_texture != entry['ceilingType'] or
                    db_load_log_id != log_id):
                # 更新数据库记录
                self.db_cursor.execute('''
                   UPDATE facilities
                   SET load_log_id=?, line_name=?, station_name=?, celling_name=?, depart_name=?, ceiling_texture=?
                   WHERE line_id=? AND station_id=? AND celling_id=? AND depart_id=?
                   ''', (log_id, entry['line_name'], entry['station_name'], entry['celling_name'], entry['depart_name'],
                         entry['ceilingType'],
                         line_id, station_id, celling_id, depart_id))
                self.db_connection.commit()
                return True
        return False

    def __del__(self):
        # 关闭数据库连接
        self.db_connection.close()

    def capture_images(self):
        # 获取用户选择的曝光时间
        exposure_time = float(self.exposure_combo.currentText())

        # 设置曝光时间
        Common.show_error(self.device.set_scan_2d_exposure_time(exposure_time))
        # 捕获彩色图像和深度图像
        self.color_image = self.device.capture_color().data()
        self.depth_image = self.device.capture_depth().data()

    def show_zoom_window(self, label):
        pixmap = label.pixmap()
        if pixmap:
            zoom_window = ZoomWindow(pixmap)
            zoom_window.exec()

    def show_zoom_window_method(self):
        zoom_window = ZoomWindow(self)
        zoom_window.exec_()

    def update_lines(self, event=None):
        self.db_cursor.execute("SELECT DISTINCT line_name FROM facilities")
        lines = [row[0] for row in self.db_cursor.fetchall()]

        self.line_combo.clear()
        self.line_combo.addItems(lines)
        self.line_combo.show()

        self.save_dir = os.getcwd() + "/data/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 自动更新车站和相关的中心、吊顶样式
        self.update_stations()
        self.update_centers()
        self.update_ceiling_types()

    def update_stations(self, event=None):
        selected_line = self.line_combo.currentText()

        if selected_line:
            self.db_cursor.execute("SELECT DISTINCT station_name FROM facilities WHERE line_name=?", (selected_line,))
            stations = [row[0] for row in self.db_cursor.fetchall()]

            self.station_combo.clear()
            self.station_combo.addItems(stations)
            self.station_combo.show()

        # 自动更新区域和相关的中心、吊顶样式
        self.update_areas()
        self.update_centers()
        self.update_ceiling_types()

    def update_areas(self, event=None):
        selected_line = self.line_combo.currentText()
        selected_station = self.station_combo.currentText()
        if selected_line and selected_station:
            self.db_cursor.execute(
                "SELECT DISTINCT celling_name FROM facilities WHERE line_name=? AND station_name=?",
                (selected_line, selected_station))
            areas = [row[0] for row in self.db_cursor.fetchall()]
            self.area_combo.clear()
            self.area_combo.addItems(areas)
            self.area_combo.show()

        # 自动更新相关的中心、吊顶样式
        self.update_centers()
        self.update_ceiling_types()

    def update_centers(self, event=None):
        selected_line = self.line_combo.currentText()
        selected_station = self.station_combo.currentText()
        selected_area = self.area_combo.currentText()
        if selected_line and selected_station and selected_area:
            self.db_cursor.execute(
                "SELECT DISTINCT depart_name FROM facilities WHERE line_name=? AND station_name=? AND celling_name=?",
                (selected_line, selected_station, selected_area))
            centers = [row[0] for row in self.db_cursor.fetchall()]
            self.center_combo.clear()
            self.center_combo.addItems(centers)
            self.center_combo.show()

    def update_ceiling_types(self, event=None):
        selected_line = self.line_combo.currentText()
        selected_station = self.station_combo.currentText()
        selected_area = self.area_combo.currentText()
        if selected_line and selected_station and selected_area:
            self.db_cursor.execute(
                "SELECT DISTINCT ceiling_texture FROM facilities WHERE line_name=? AND station_name=? AND celling_name=?",
                (selected_line, selected_station, selected_area))
            ceiling_types = [row[0] for row in self.db_cursor.fetchall()]
            self.ceiling_type_combo.clear()
            self.ceiling_type_combo.addItems(ceiling_types)
            self.ceiling_type_combo.show()

    def start_timer(self):
        QMessageBox.information(self, "开始采集。", "正在采集，请稍后")
        self.capture_thread = ImageCaptureThread(self.device, self)

        # self.capture_thread = ImageCaptureThread(self.device)  # 重新创建线程实例
        selected_model = self.model_combo.currentText()
        self.capture_thread.set_selected_model(selected_model)
        self.capture_thread.imagesCaptured.connect(self.update_labels)
        self.capture_thread.imagesCapturedAndSaved.connect(self.save_images)
        self.capture_thread.detectionCompleted.connect(self.update_detection_result)
        self.capture_thread.depthvalue.connect(self.update_depth_label)
        self.capture_thread.depthvalue1.connect(self.update_depth_label1)
        # self.capture_thread.set_capture_interval(self.selected_interval)  # 将采集时间间隔设置到图像采集线程中
        self.capture_thread.set_capture_interval(self.selected_interval if hasattr(self,
                                                                                   'selected_interval') else 10000)  # 将采集时间间隔设置到图像采集线程中，默认值为10000毫秒（10秒）
        self.capture_thread.lcMaxUpdated.connect(self.update_alarm_label)
        self.capture_thread.start()  # 启动线程

    def update_alarm_label(self, text):
        self.data_display2.setText(text)  # 更新UI显示最大LC值

    def update_selected_model(self, model_index):
        # 获取当前选择的吊顶类型
        self.capture_thread = ImageCaptureThread(self.device, self)
        selected_model = self.model_combo.itemText(model_index)
        self.capture_thread.set_selected_model(selected_model)

    def update_depth_image(self):
        depth_min = 725  # 期望的深度范围最小值
        depth_max = 856  # 期望的深度范围最大值
        slider_value = self.depth_slider.value()
        processed_depth_image = self.capture_thread.get_depth()  # 自定义的深度图像处理函数

        if processed_depth_image is not None and isinstance(processed_depth_image, np.ndarray):
            # 将slider_value映射到725到856的范围
            depth_value = int(np.interp(slider_value, [self.depth_slider.minimum(), self.depth_slider.maximum()],
                                        [depth_min, depth_max]))

            # 映射深度图像范围到725到856
            processed_depth_image = np.clip(processed_depth_image, depth_min, depth_max)

            # 将深度图像转换为颜色图像
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(processed_depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 更新self.depth_label的Pixmap，显示更新后的深度图像
            depth_img = QImage(depth_colormap.data, depth_colormap.shape[1], depth_colormap.shape[0],
                               QImage.Format_RGB888)
            self.depth_label.setPixmap(QPixmap.fromImage(depth_img))
        else:
            print("Error: Invalid processed_depth_image")

    def update_labels(self, color_img, depth_img):
        self.image_label.setPixmap(QPixmap.fromImage(color_img))
        self.depth_label.setPixmap(QPixmap.fromImage(depth_img))

    def line_1(self):
        selected_line = self.line_combo.currentText()
        selected_station = self.station_combo.currentText()
        selected_area = self.area_combo.currentText()
        if selected_line and selected_station and selected_area:
            self.current_station_dir = os.path.join(self.save_dir, selected_line, selected_station, selected_area)
            if not os.path.exists(self.current_station_dir):
                os.makedirs(self.current_station_dir)

            print(f"已创建目录：{self.current_station_dir}")
        else:
            print("请选择线路、车站和区域以创建文件夹。")

        Common.find_camera_list(self)

        if Common.choose_camera_and_connect(self):
            self.connected = True
            print("Connected to the Mech-Eye device successfully.")
            QMessageBox.information(self, "连接成功", f"已成功连接到相机 ")
        else:
            print("Failed to connect to the Mech-Eye device.")
            QMessageBox.warning(self, "连接失败", f"连接到相机失败 ")

    def capture_color_map1(self):
        color_map = self.device.capture_color()
        color_file = os.path.join(self.current_station_dir, self.t_name + "ColorMap.png")
        cv2.imencode('.png', color_map.data())[1].tofile(color_file)
        print("Capture and save color image : {}".format(color_file))

    def capture_depth_map1(self):
        depth_map = self.device.capture_depth()
        depth_file = os.path.join(self.current_station_dir, self.t_name + "DepthMap.tiff")
        cv2.imencode('.tiff', depth_map.data())[1].tofile(depth_file)
        print("Capture and save depth image : {}".format(depth_file))

    def capture_point_cloud(self):
        points_xyz = self.device.capture_point_xyz()
        points_xyz_data = points_xyz.data()
        points_xyz_o3d = o3d.geometry.PointCloud()
        points_xyz_o3d.points = o3d.utility.Vector3dVector(points_xyz_data.reshape(-1, 3) * 0.001)
        o3d.io.write_point_cloud(os.path.join(self.current_station_dir, self.t_name + "PointCloudXYZ.ply"),
                                 points_xyz_o3d)
        print("Point cloud saved to path" + self.current_station_dir + self.t_name + "PointCloudXYZ.ply")

    def capture_color_point_cloud(self):
        points_xyz_bgr = self.device.capture_point_xyz_bgr().data()
        points_reshape = points_xyz_bgr.reshape(-1, 6)
        points_xyz_rgb_points = points_reshape[:, :3] * 0.001
        point_xyz_rgb_colors = points_reshape[:, 3:6][:, ::-1] / 255

        points_xyz_rgb_o3d = o3d.geometry.PointCloud()
        points_xyz_rgb_o3d.points = o3d.utility.Vector3dVector(points_xyz_rgb_points.astype(np.float64))
        points_xyz_rgb_o3d.colors = o3d.utility.Vector3dVector(point_xyz_rgb_colors.astype(np.float64))
        o3d.io.write_point_cloud(os.path.join(self.current_station_dir, self.t_name + "PointCloudXYZRGB.ply"),
                                 points_xyz_rgb_o3d)
        print("Point cloud saved to path" + self.current_station_dir + self.t_name + "PointCloudXYZ.ply")

    def save_images(self):
        if not self.connected:
            print("请先点击启动按钮连接到 Mech-Eye 设备")
            return

        self.num_collect = self.num_collect + 1
        self.t_name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        self.data_label.setText(f"采集数量: {self.num_collect}")
        print("采集一次")
        print("第%d次数据采集" % self.num_collect)



        # 使用传递的图像保存
        self.save_color_image(self.capture_thread.color_img)
        self.save_depth_image(self.capture_thread.depth_img)

    def save_color_image(self, color_img):
        color_file = os.path.join(self.current_station_dir, self.t_name + "ColorImage.png")
        color_img.save(color_file)
        print("Capture and save color image:", color_file)

    def save_depth_image(self, depth_img):
        depth_file = os.path.join(self.current_station_dir, self.t_name + "DepthImage.png")

        depth_img.save(depth_file)
        print("Capture and save depth image:", depth_file)

    def save_ply_files(self):
        if not self.connected:
            print("请先点击启动按钮连接到 Mech-Eye 设备")
            return

        # 保存点云PLY文件
        points_xyz_ply = o3d.io.read_point_cloud(
            os.path.join(self.current_station_dir, self.t_name + "PointCloudXYZ.ply"))
        points_xyz_ply_path = os.path.join(self.current_station_dir, self.t_name + "PointCloudXYZ.ply")
        o3d.io.write_point_cloud(points_xyz_ply_path, points_xyz_ply)

        # 保存带颜色的点云PLY文件
        points_xyz_rgb_ply = o3d.io.read_point_cloud(
            os.path.join(self.current_station_dir, self.t_name + "PointCloudXYZRGB.ply"))
        points_xyz_rgb_ply_path = os.path.join(self.current_station_dir, self.t_name + "PointCloudXYZRGB.ply")
        o3d.io.write_point_cloud(points_xyz_rgb_ply_path, points_xyz_rgb_ply)

    def stop_timer(self):
        if self.capture_thread.isRunning():
            self.capture_thread.is_running = False  # 通知线程停止运行
            self.capture_thread.terminate()  # 强制终止线程
            self.capture_thread.wait()  # 等待线程完全停止
            self.timer.stop()  # 停止Qt计时器


def main():
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
