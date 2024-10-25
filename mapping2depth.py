import numpy as np
import torch

def lv_mapping2deep(x1, y1, x2, y2):
    '''
    铝方通专用裂缝检测框的坐标映射到深度图的坐标
    :param x1: 裂缝检测框的左上x坐标
    :param y1:裂缝检测框的左上y坐标
    :param x2:裂缝检测框的右下x坐标
    :param y2:裂缝检测框的右下y坐标
    :return:
    '''
    global depth_x2, depth_x1, depth_y1, depth_y2
    c_box = []
    d_height, d_width =  1405 - 50, 1910 - 80
    weight, height = 2000, 1500
    d_x1 = abs(x1 + (x1 * ( 0.02*(x1 / weight))))
    # d_x1 = x1
    d_y1 = abs(y1 - (y1 * ( 0.05*(y1 / height))))
    d_x2 = abs(x2 + (x2 * ( 0.02*(x2 / weight))))
    # d_x2 = x2
    d_y2 = abs(y2 - (y2 * ( 0.05*(y2 / height))))
    # print("\tRGB", x1, "Dep", d_x1, "\tRGB", y1, "Dep", d_y1, "\tRGB", x2, "Dep", d_x2, "\tRGB", y2, "Dep", d_y2)
    b_depth_x1 = int(d_x1 * d_width / weight)
    b_depth_y1 = int(d_y1 * d_height / height)
    b_depth_x2 = int(d_x2 * d_width / weight)
    b_depth_y2 = int(d_y2 * d_height / height)
    # 判断是否为横向标签框 根据标签框的形状计算不同的差值
    is_horizontal = abs(b_depth_x2 - b_depth_x1) > abs(b_depth_y2 - b_depth_y1)
    b_w, b_h = b_depth_x2 - b_depth_x1, b_depth_y2 - b_depth_y1  # 获取宽高, 用来以不同方式摘取待分析图像内容

    # 横的比竖的数值大，is_horizontal为Ture，横向标签框
    if is_horizontal:
        shape = 0
        if b_depth_y2 < 395 or b_depth_y1 < 395:

            b_depth_y1 = b_depth_y1 - int((395 - b_depth_y1) * 0.04)
            b_depth_y2 = b_depth_y2 - int((395 - b_depth_y2) * 0.04)
        else:
            b_depth_y1 = b_depth_y1 + int((b_depth_y1 - 395) * 0.08)
            b_depth_y2 = b_depth_y2 + int((b_depth_y2 - 395) * 0.08)


        mean = b_depth_y2 - b_depth_y1
        counts = [b_depth_x1, b_depth_y1, b_depth_x2, b_depth_y2, mean, shape]

    # 竖的比横的数值大，is_horizontal为False，竖向标签框
    else:
        # 纵向标签框
        shape = 1
        if b_depth_y2 < 395 or b_depth_y1 < 395:

            b_depth_y1 = b_depth_y1 - int((395 - b_depth_y1) * 0.04)
            b_depth_y2 = b_depth_y2 - int((395 - b_depth_y2) * 0.04)
        else:
            b_depth_y1 = b_depth_y1 + int((b_depth_y1 - 395) * 0.08)
            b_depth_y2 = b_depth_y2 + int((b_depth_y2 - 395) * 0.08)

        mean = b_depth_x2 - b_depth_x1
        counts = [b_depth_x1, b_depth_y1, b_depth_x2, b_depth_y2, mean, shape]

    # for cut in counts:
    c_x1, c_y1, c_x2, c_y2, c_mean, c_shape = counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]
    if 80 < int(c_y2) < 1230:
        # if 875 < c_y1 and c_x1 < 358:
        #     if c_y1 > 500:
        #         c_x1, c_x2 = c_x1 - 7, c_x2 - 7
        return counts
    return []


def gd_mapping2depth(x1, y1, x2, y2):
    '''
    将RGB图的坐标映射到深度图的坐标，需要先进行检测，得到检测框的位置坐标才能进行映射，在遍历获取xyxy坐标循环内调用
    接深度图的z轴计算、裂缝宽度X/Y轴计算
    :param x1: RGB图的左上x坐标
    :param y1:RGB图的左上y坐标
    :param x2:RGB图的右下x坐标
    :param y2:RGB图的右下y坐标
    :return:counts : 列表，里面有[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]
    depth_x1, depth_y1, depth_x2, depth_y2,
    mean(计算裂缝像素需要除的:竖除h、横除w)：
    shape：形状，0:横框 1:竖框
    只用于计算用的位置坐标，不能用来画框，请与用来画框的位置坐标做区分
    '''
    c_box = []
    d_height, d_width = 1405 - 50, 1910 - 80
    weight, height = 2000, 1500
    d_x1 = abs(x1 + (x1 * (0.02 * (x1 / weight))))
    # d_x1 = x1
    d_y1 = abs(y1 - (y1 * (0.05 * (y1 / height))))
    d_x2 = abs(x2 + (x2 * (0.012 * (x2 / weight))))
    # d_x2 = x2
    d_y2 = abs(y2 - (y2 * (0.05 * (y2 / height))))
    # print("\tRGB", x1, "Dep", d_x1, "\tRGB", y1, "Dep", d_y1, "\tRGB", x2, "Dep", d_x2, "\tRGB", y2, "Dep", d_y2)
    b_depth_x1 = int(d_x1 * d_width / weight)
    b_depth_y1 = int(d_y1 * d_height / height)
    b_depth_x2 = int(d_x2 * d_width / weight)
    b_depth_y2 = int(d_y2 * d_height / height)
    # print("\tRGB", x1, "Dep", b_depth_x1, "\tRGB", y1, "Dep", b_depth_y1, "\tRGB", x2, "Dep", b_depth_x2, "\tRGB", y2, "Dep", b_depth_y2)
    # # 映射彩色图像坐标到深度图像坐标
    # # 深度图比彩色图像大，框按照彩色图进行目标检测，按照比例去计算框在深度图的位置
    # # 深度图的分辨率
    # depth_height, depth_width = 1405 - 50, 1877 - 80
    # # RGB图的分辨率
    # image_width, image_height = 2000 , 1500
    # b_depth_x1 = int(x1 * depth_width / image_width)
    # b_depth_y1 = int(y1 * depth_height / image_height)
    # b_depth_x2 = int(x2 * depth_width / image_width)
    # b_depth_y2 = int(y2 * depth_height / image_height)
    # depth_x3 = int(x1 * depth_width / image_width)
    # depth_y3 = int(y1 * depth_height / image_height)
    # depth_x4 = int(x2 * depth_width / image_width)
    # depth_y4 = int(y2 * depth_height / image_height)

    # 判断是否为横向标签框 根据标签框的形状计算不同的差值
    is_horizontal = abs(b_depth_x2 - b_depth_x1) > abs(b_depth_y2 - b_depth_y1)
    b_w, b_h = b_depth_x2 - b_depth_x1, b_depth_y2 - b_depth_y1  # 获取宽高, 用来以不同方式摘取待分析图像内容
    print(b_depth_x1, b_depth_y1, b_depth_x2, b_depth_y2)

    # 横的比竖的数值大，is_horizontal为Ture，横向标签框
    if is_horizontal:
        ratio = b_h / b_w
        shape = 0
        if b_depth_y2 < 395 or b_depth_y1 < 395:
            b_depth_y1 = b_depth_y1 - int((395 - b_depth_y1) * 0.04)
            b_depth_y2 = b_depth_y2 - int((395 - b_depth_y2) * 0.04)
        else:
            b_depth_y1 = b_depth_y1 + int((b_depth_y1 - 395) * 0.08)
            b_depth_y2 = b_depth_y2 + int((b_depth_y2 - 395) * 0.08)

        c_w = b_w - (b_w * ratio)  # 横的长，多截一点，少留一点
        c_h = b_h * 0.5  # 竖的太短，直接不截掉，全部保留竖着的像素
        depth_x1, depth_y1 = b_depth_x1 + int(c_w / 1.5), b_depth_y1 - int(c_h / 1.5)
        depth_x2, depth_y2 = b_depth_x2 - int(c_w / 1.5), b_depth_y2 + int(c_h / 1.5)
        # 整理坐标点
        depth_x1, depth_x2 = min(depth_x1, depth_x2), max(depth_x1, depth_x2)
        depth_y1, depth_y2 = min(depth_y1, depth_y2), max(depth_y1, depth_y2)


        mean = depth_y2 - depth_y1 - 10
        counts = [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
        # counts.append([b_depth_x1, b_depth_y1, b_depth_x2, b_depth_y2, mean, shape])
        count = b_w // (mean // 2)
        for i in range(1, count):
            up_depth_x1, up_depth_y1 = depth_x1 - (mean * i), depth_y1
            up_depth_x2, up_depth_y2 = depth_x2 - (mean * i), depth_y2

            dw_depth_x1, dw_depth_y1 = depth_x1 + (mean * i), depth_y1
            dw_depth_x2, dw_depth_y2 = depth_x2 + (mean * i), depth_y2

            # 整理坐标点
            up_depth_x1, up_depth_x2 = min(up_depth_x1, up_depth_x2), max(up_depth_x1, up_depth_x2)
            up_depth_y1, up_depth_y2 = min(up_depth_y1, up_depth_y2), max(up_depth_y1, up_depth_y2)
            dw_depth_x1, dw_depth_x2 = min(dw_depth_x1, dw_depth_x2), max(dw_depth_x1, dw_depth_x2)
            dw_depth_y1, dw_depth_y2 = min(dw_depth_y1, dw_depth_y2), max(dw_depth_y1, dw_depth_y2)
            if up_depth_x1 > b_depth_x1 + 10 :
                    counts.append([up_depth_x1, up_depth_y1, up_depth_x2, up_depth_y2, mean, shape])
            if dw_depth_x2 < b_depth_x2 - 10:
                    counts.append([dw_depth_x1, dw_depth_y1, dw_depth_x2, dw_depth_y2, mean, shape])
    # 竖的比横的数值大，is_horizontal为False，竖向标签框
    else:
        # 纵向标签框
        shape = 1
        ratio = b_w / b_h
        if b_depth_y2 < 395 or b_depth_y1 < 395:

            b_depth_y1 = b_depth_y1 - int((395 - b_depth_y1) * 0.04)
            b_depth_y2 = b_depth_y2 - int((395 - b_depth_y2) * 0.04)
        else:
            b_depth_y1 = b_depth_y1 + int((b_depth_y1 - 395) * 0.08)
            b_depth_y2 = b_depth_y2 + int((b_depth_y2 - 395) * 0.08)

        c_w = b_w * 0.5  # 横的太短，直接不截掉，全部保留竖着的像素
        c_h = b_h - (b_h * ratio)  # 竖的长，多截一点，少留一点
        depth_x1, depth_y1 = b_depth_x1 - int(c_w / 1.5), b_depth_y1 + int(c_h / 1.5)
        depth_x2, depth_y2 = b_depth_x2 + int(c_w / 1.5), b_depth_y2 - int(c_h / 1.5)
        # 整理坐标点
        depth_x1, depth_x2 = min(depth_x1, depth_x2), max(depth_x1, depth_x2)
        depth_y1, depth_y2 = min(depth_y1, depth_y2), max(depth_y1, depth_y2)

        mean = depth_x2 - depth_x1 - 10
        counts = [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
        # counts.append([b_depth_x1, b_depth_y1, b_depth_x2, b_depth_y2, mean, shape])
        count = b_h // (mean // 2)
        for i in range(1, count):
            up_depth_x1, up_depth_y1 = depth_x1, depth_y1 - (mean * i)
            up_depth_x2, up_depth_y2 = depth_x2, depth_y2 - (mean * i)

            dw_depth_x1, dw_depth_y1 = depth_x1, depth_y1 + (mean * i)
            dw_depth_x2, dw_depth_y2 = depth_x2, depth_y2 + (mean * i)

            # 整理坐标点
            up_depth_x1, up_depth_x2 = min(up_depth_x1, up_depth_x2), max(up_depth_x1, up_depth_x2)
            up_depth_y1, up_depth_y2 = min(up_depth_y1, up_depth_y2), max(up_depth_y1, up_depth_y2)
            dw_depth_x1, dw_depth_x2 = min(dw_depth_x1, dw_depth_x2), max(dw_depth_x1, dw_depth_x2)
            dw_depth_y1, dw_depth_y2 = min(dw_depth_y1, dw_depth_y2), max(dw_depth_y1, dw_depth_y2)
            if up_depth_y1 > b_depth_y1 + 10:
                # print(up_depth_y1,"----------------------", b_depth_y1+20)
                counts.append([up_depth_x1, up_depth_y1, up_depth_x2, up_depth_y2, mean, shape])
            if dw_depth_y2 < b_depth_y2 - 10:
                # print(dw_depth_y2,"----------------------",b_depth_y2-10)
                counts.append([dw_depth_x1, dw_depth_y1, dw_depth_x2, dw_depth_y2, mean, shape])
    # print("counts：：：：：：\n",counts)
    for cut in counts:
        c_x1, c_y1, c_x2, c_y2, c_mean, c_shape = cut
        if 80 < c_y1 and c_y2 < 1230 and c_x1 > 150:
            if 875 < c_y1 and c_x1 < 358:
                if c_y1 > 500:
                    c_x1, c_x2 = c_x1 - 7, c_x2 - 7
            c_box.append([c_x1, c_y1, c_x2, c_y2, c_mean, c_shape])
    return c_box


def mapping2depth(x1, y1, x2, y2):
    '''
    将RGB图的坐标映射到深度图的坐标，需要先进行检测，得到检测框的位置坐标才能进行映射，在遍历获取xyxy坐标循环内调用
    接深度图的z轴计算、裂缝宽度X/Y轴计算
    :param x1: RGB图的左上x坐标
    :param y1:RGB图的左上y坐标
    :param x2:RGB图的右下x坐标
    :param y2:RGB图的右下y坐标
    :return:counts : 列表，里面有[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]
    depth_x1, depth_y1, depth_x2, depth_y2,
    mean(计算裂缝像素需要除的:竖除h、横除w)：
    shape：形状，0:横框 1:竖框
    只用于计算用的位置坐标，不能用来画框，请与用来画框的位置坐标做区分
    '''
    c_box = []
    d_height, d_width =  1405 - 50, 1910 - 80
    weight, height = 2000, 1500
    d_x1 = abs(x1 + (x1 * ( 0.02*(x1 / weight))))
    # d_x1 = x1
    d_y1 = abs(y1 - (y1 * ( 0.05*(y1 / height))))
    d_x2 = abs(x2 + (x2 * ( 0.012*(x2 / weight))))
    # d_x2 = x2
    d_y2 = abs(y2 - (y2 * ( 0.05*(y2 / height))))
    # print("\tRGB", x1, "Dep", d_x1, "\tRGB", y1, "Dep", d_y1, "\tRGB", x2, "Dep", d_x2, "\tRGB", y2, "Dep", d_y2)
    b_depth_x1 = int(d_x1 * d_width / weight)
    b_depth_y1 = int(d_y1 * d_height / height)
    b_depth_x2 = int(d_x2 * d_width / weight)
    b_depth_y2 = int(d_y2 * d_height / height)
    # # 映射彩色图像坐标到深度图像坐标
    # # 深度图比彩色图像大，框按照彩色图进行目标检测，按照比例去计算框在深度图的位置
    # # 深度图的分辨率
    # depth_height, depth_width = 1405 - 50, 1877 - 80
    # # RGB图的分辨率
    # image_width, image_height = 2000 , 1500
    # b_depth_x1 = int(x1 * depth_width / image_width)
    # b_depth_y1 = int(y1 * depth_height / image_height)
    # b_depth_x2 = int(x2 * depth_width / image_width)
    # b_depth_y2 = int(y2 * depth_height / image_height)
    # depth_x3 = int(x1 * depth_width / image_width)
    # depth_y3 = int(y1 * depth_height / image_height)
    # depth_x4 = int(x2 * depth_width / image_width)
    # depth_y4 = int(y2 * depth_height / image_height)

    # 判断是否为横向标签框 根据标签框的形状计算不同的差值
    is_horizontal = abs(b_depth_x2 - b_depth_x1) > abs(b_depth_y2 - b_depth_y1)
    b_w, b_h = b_depth_x2 - b_depth_x1, b_depth_y2 - b_depth_y1  # 获取宽高, 用来以不同方式摘取待分析图像内容


    # 横的比竖的数值大，is_horizontal为Ture，横向标签框
    if is_horizontal:
        ratio = b_h / b_w
        shape = 0
        if b_w > 100 or b_h > 100:  # 判断框大小，如果框太小了，就不再截了
            if ratio >= 0.6:  # 如果框太方了就得多处理一下
                c_w = b_w - (b_w * (1 - ratio))
                c_h = b_h - (b_h * (1 - ratio))
                depth_x1, depth_y1 = b_depth_x1 + int(c_w / 2), b_depth_y1 + int(c_h / 2)
                depth_x2, depth_y2 = b_depth_x2 - int(c_w / 2), b_depth_y2 - int(c_h / 2)
                mean = depth_x2 - depth_x1
                if depth_y2 < 1450 and depth_y1 > 120 and depth_x2 < 2030:
                    return [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
                else:
                    return [[]]

            elif 0.3 < ratio < 0.5:
                c_w = b_w - (b_w * ratio)  # 横的长，多截一点，少留一点
                c_h = b_h - (b_h * (1 - ratio))  # 竖的短，少截掉一点，多留一些
                depth_x1, depth_y1 = b_depth_x1 + int(c_w / 2), b_depth_y1 + int(c_h / 2)
                depth_x2, depth_y2 = b_depth_x2 - int(c_w / 2), b_depth_y2 - int(c_h / 2)
                mean = depth_x2 - depth_x1
                if depth_y2 < 1450 and depth_y1 > 120 and depth_x2 < 2030:
                    return [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
                else:
                    return [[]]
            elif 0.1 < ratio <= 0.3:
                c_w = b_w - (b_w * ratio)  # 横的长，多截一点，少留一点
                c_h = b_h * 2  # 竖的太短，直接不截掉，全部保留竖着的像素
                depth_x1, depth_y1 = b_depth_x1 + int(c_w / 2), b_depth_y1 - int(c_h / 2)
                depth_x2, depth_y2 = b_depth_x2 - int(c_w / 2), b_depth_y2 + int(c_h / 2)
            elif ratio <= 0.1:
                c_w = b_w - (b_w * ratio)  # 横的长，多截一点，少留一点
                c_h = b_h * 1.5  # 竖的太短，直接不截掉，全部保留竖着的像素
                depth_x1, depth_y1 = b_depth_x1 + int(c_w / 2), b_depth_y1 - int(c_h / 2)
                depth_x2, depth_y2 = b_depth_x2 - int(c_w / 2), b_depth_y2 + int(c_h / 2)
        else:  # 框太小，就不截掉了

            c_w = b_w - (b_w * ratio)  # 横的长，多截一点，少留一点
            c_h = b_h * 2   # 竖的太短，直接不截掉，全部保留竖着的像素
            depth_x1, depth_y1 = b_depth_x1 + int(c_w / 2), b_depth_y1 - int(c_h / 2)
            depth_x2, depth_y2 = b_depth_x2 - int(c_w / 2), b_depth_y2 + int(c_h / 2)
            mean = depth_x2 - depth_x1
            if depth_y2 < 1450 and depth_y1 > 120 and depth_x2 < 2030:
                return [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
            else:
                return [[]]

        mean = depth_y2 - depth_y1
        counts = [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
        count = b_w // (mean//2)
        for i in range(1, count):
            up_depth_x1, up_depth_y1 = depth_x1 - (mean * i), depth_y1
            up_depth_x2, up_depth_y2 = depth_x2 - (mean * i), depth_y2

            dw_depth_x1, dw_depth_y1 = depth_x1 + (mean * i), depth_y1
            dw_depth_x2, dw_depth_y2 = depth_x2 + (mean * i), depth_y2
            if up_depth_x1 > b_depth_x1 or dw_depth_y2 < b_depth_y2:
                counts.append([up_depth_x1, up_depth_y1, up_depth_x2, up_depth_y2, mean, shape])
                counts.append([dw_depth_x1, dw_depth_y1, dw_depth_x2, dw_depth_y2,mean, shape])
    # 竖的比横的数值大，is_horizontal为False，竖向标签框
    else:
        # 纵向标签框
        shape = 1
        ratio = b_w / b_h
        if b_w > 100 or b_h > 100:  # 判断框大小，如果框太小了，就不再截了
            if ratio >= 0.6:  # 如果框太方了就得多处理一下
                c_w = b_w - (b_w * (1 - ratio))
                c_h = b_h - (b_h * (1 - ratio))
                depth_x1, depth_y1 = b_depth_x1 + int(c_w / 2), b_depth_y1 + int(c_h / 2)
                depth_x2, depth_y2 = b_depth_x2 - int(c_w / 2), b_depth_y2 - int(c_h / 2)
                mean = depth_x2 - depth_x1
                if depth_y2 < 1450 and depth_y1 > 120 and depth_x2 < 2030:
                    return [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
                else:
                    return [[]]
            elif 0.3 < ratio < 0.5:
                c_w = b_w - (b_w * (1 - ratio))  # 横的长，多截一点，少留一点
                c_h = b_h - (b_h * ratio)  # 竖的短，少截掉一点，多留一些
                depth_x1, depth_y1 = b_depth_x1 + int(c_w / 2), b_depth_y1 + int(c_h / 2)
                depth_x2, depth_y2 = b_depth_x2 - int(c_w / 2), b_depth_y2 - int(c_h / 2)
                mean = depth_x2 - depth_x1
                if depth_y2 < 1450 and depth_y1 > 120 and depth_x2 < 2030:
                    return [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
                else:
                    return [[]]
            elif 0.1 < ratio <= 0.3:
                c_w = b_w * 2  # 横的太短，直接不截掉，全部保留竖着的像素
                c_h = b_h - (b_h * ratio)  # 竖的长，多截一点，少留一点
                depth_x1, depth_y1 = b_depth_x1 - int(c_w / 2), b_depth_y1 + int(c_h / 2)
                depth_x2, depth_y2 = b_depth_x2 + int(c_w / 2), b_depth_y2 - int(c_h / 2)
            elif ratio < 0.1:
                c_w = b_w * 1.5  # 横的太短，直接不截掉，全部保留竖着的像素
                c_h = b_h - (b_h * ratio)  # 竖的长，多截一点，少留一点
                depth_x1, depth_y1 = b_depth_x1 - int(c_w / 2), b_depth_y1 + int(c_h / 2)
                depth_x2, depth_y2 = b_depth_x2 + int(c_w / 2), b_depth_y2 - int(c_h / 2)
        else:  # 框太小，就不截掉了
            c_w = b_w * 2  # 横的太短，直接不截掉，全部保留竖着的像素
            c_h = b_h - (b_h * ratio)  # 竖的长，多截一点，少留一点
            depth_x1, depth_y1 = b_depth_x1 - int(c_w / 2), b_depth_y1 + int(c_h / 2)
            depth_x2, depth_y2 = b_depth_x2 + int(c_w / 2), b_depth_y2 - int(c_h / 2)
            mean = depth_x2 - depth_x1
            if depth_y2 < 1450 and depth_y1 > 120 and depth_x2 < 2030:
                return [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
            else:
                return [[]]

        mean = depth_x2 - depth_x1
        counts = [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
        count = b_h // (mean//2)
        for i in range(1, count):
            up_depth_x1, up_depth_y1 = depth_x1, depth_y1 - (mean * i)
            up_depth_x2, up_depth_y2 = depth_x2, depth_y2 - (mean * i)

            dw_depth_x1, dw_depth_y1 = depth_x1, depth_y1 + (mean * i)
            dw_depth_x2, dw_depth_y2 = depth_x2, depth_y2 + (mean * i)
            if up_depth_x1 > b_depth_x1 or dw_depth_y2 < b_depth_y2:
                counts.append([up_depth_x1, up_depth_y1, up_depth_x2, up_depth_y2, mean, shape])
                counts.append([dw_depth_x1, dw_depth_y1, dw_depth_x2, dw_depth_y2,mean, shape])
    # print("counts：：：：：：\n",counts)
    for cut in counts:
        c_x1, c_y1, c_x2, c_y2, c_mean, c_shape = cut
        if 80 < c_y2 < 1230 and c_x1 > 150:
            # if c_y2 <= 857 :
            #     # print("c_x1+14,c_x2+14")
            #     c_x1,c_x2 = c_x1+14,c_x2+14
            if 875 < c_y1 and c_x1 < 358:
                #     # print("c_y1 - 17, c_y2 - 17")
                #     c_y1, c_y2 = c_y1 - 17, c_y2 - 17
                if c_y1 > 500:
                    # print("c_x1, c_x2 = c_x1 - 7, c_x2 - 7")
                    c_x1, c_x2 = c_x1 - 7, c_x2 - 7
            # elif c_x1 > 1000 and c_y1 > 931:
            #     # print("c_y1 + 14, c_y2 + 14")
            #     c_y1, c_y2 = c_y1 + 14, c_y2 + 14
            c_box.append([c_x1, c_y1, c_x2, c_y2, c_mean, c_shape])
    return c_box


def S_mapping2depth(x1, y1,x2,y2):
    # 获取图像的高度和宽度
    global depth_x2, depth_x1, depth_y1, depth_y2
    c_box = []

    d_height, d_width = 1536, 2048
    weight, height = 2000, 1500
    d_x1 = abs(x1 + (x1 * ( 0.015*(x1 / weight))))
    # d_x1 = x1
    d_y1 = abs(y1 + (y1 * ( 0.05*(y1 / height))))
    d_x2 = abs(x2 + (x2 * ( 0.02*(x2 / weight))))
    # d_x2 = x2
    d_y2 = abs(y2 + (y2 * ( 0.05*(y2 / height))))
    # print("\tRGB", x1, "Dep", d_x1, "\tRGB", y1, "Dep", d_y1, "\tRGB", x2, "Dep", d_x2, "\tRGB", y2, "Dep", d_y2)
    b_depth_x1 = int(d_x1 * d_width / weight)
    b_depth_y1 = int(d_y1 * d_height / height)
    b_depth_x2 = int(d_x2 * d_width / weight)
    b_depth_y2 = int(d_y2 * d_height / height)
    # print("\tRGB", x1, "Dep", b_depth_x1, "\tRGB", y1, "Dep", b_depth_y1, "\tRGB", x2, "Dep", b_depth_x2, "\tRGB", y2, "Dep", b_depth_y2)



    # 判断是否为横向标签框 根据标签框的形状计算不同的差值
    is_horizontal = abs(b_depth_x2 - b_depth_x1) > abs(b_depth_y2 - b_depth_y1)
    b_w, b_h = b_depth_x2 - b_depth_x1, b_depth_y2 - b_depth_y1  # 获取宽高, 用来以不同方式摘取待分析图像内容


    # 横的比竖的数值大，is_horizontal为Ture，横向标签框
    if is_horizontal:
        ratio = b_h / b_w
        shape = 0
        if b_w > 100 or b_h > 100:  # 判断框大小，如果框太小了，就不再截了
            if ratio >= 0.6:  # 如果框太方了就得多处理一下
                c_w = b_w - (b_w * (1 - ratio))
                c_h = b_h - (b_h * (1 - ratio))
                depth_x1, depth_y1 = b_depth_x1 + int(c_w / 2), b_depth_y1 + int(c_h / 2)
                depth_x2, depth_y2 = b_depth_x2 - int(c_w / 2), b_depth_y2 - int(c_h / 2)
                mean = depth_x2 - depth_x1
                if depth_y2 < 1450 and depth_y1 > 120 and depth_x2 < 2030:
                    return [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
                else:
                    return [[]]
            elif 0.3 < ratio < 0.5:
                c_w = b_w - (b_w * ratio)  # 横的长，多截一点，少留一点
                c_h = b_h - (b_h * (1 - ratio))  # 竖的短，少截掉一点，多留一些
                depth_x1, depth_y1 = b_depth_x1 + int(c_w / 2), b_depth_y1 + int(c_h / 2)
                depth_x2, depth_y2 = b_depth_x2 - int(c_w / 2), b_depth_y2 - int(c_h / 2)
                mean = depth_x2 - depth_x1
                if depth_y2 < 1450 and depth_y1 > 120 and depth_x2 < 2030:
                    return [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
                else:
                    return [[]]
            elif 0.1 < ratio <= 0.3:
                c_w = b_w - (b_w * ratio)  # 横的长，多截一点，少留一点
                c_h = b_h - (b_h * (1 - ratio))  # 竖的短，少截掉一点，多留一些
                depth_x1, depth_y1 = b_depth_x1 + int(c_w / 2), b_depth_y1 - int(c_h / 2)
                depth_x2, depth_y2 = b_depth_x2 - int(c_w / 2), b_depth_y2 + int(c_h / 2)
            elif ratio <= 0.1:
                c_w = b_w - (b_w * ratio)  # 横的长，多截一点，少留一点
                c_h = b_h * 1.2  # 竖的太短，直接不截掉，全部保留竖着的像素
                depth_x1, depth_y1 = b_depth_x1 + int(c_w / 2), b_depth_y1 - int(c_h / 2)
                depth_x2, depth_y2 = b_depth_x2 - int(c_w / 2), b_depth_y2 + int(c_h / 2)
        else:  # 框太小，就不截掉了

            c_w = b_w - (b_w * ratio)  # 横的长，多截一点，少留一点
            c_h = b_h * 2   # 竖的太短，直接不截掉，全部保留竖着的像素
            depth_x1, depth_y1 = b_depth_x1 + int(c_w / 2), b_depth_y1 - int(c_h / 2)
            depth_x2, depth_y2 = b_depth_x2 - int(c_w / 2), b_depth_y2 + int(c_h / 2)
            mean = depth_x2 - depth_x1
            if depth_y2 < 1450 and depth_y1 > 120 and depth_x2 < 2030:
                return [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
            else:
                return [[]]

        mean = depth_y2 - depth_y1
        counts = [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
        print(b_w,mean)
        if mean < 2:
            mean = 2
        count = b_w // (mean//2)
        for i in range(1, count):
            up_depth_x1, up_depth_y1 = depth_x1 - (mean * i), depth_y1
            up_depth_x2, up_depth_y2 = depth_x2 - (mean * i), depth_y2

            dw_depth_x1, dw_depth_y1 = depth_x1 + (mean * i), depth_y1
            dw_depth_x2, dw_depth_y2 = depth_x2 + (mean * i), depth_y2
            if up_depth_x1 > b_depth_x1 or dw_depth_y2 < b_depth_y2:
                counts.append([up_depth_x1, up_depth_y1, up_depth_x2, up_depth_y2, mean, shape])
                counts.append([dw_depth_x1, dw_depth_y1, dw_depth_x2, dw_depth_y2,mean, shape])
    # 竖的比横的数值大，is_horizontal为False，竖向标签框
    else:
        # 纵向标签框
        shape = 1
        ratio = b_w / b_h
        if b_w > 100 or b_h > 100:  # 判断框大小，如果框太小了，就不再截了
            if ratio >= 0.6:  # 如果框太方了就得多处理一下
                c_w = b_w - (b_w * (1 - ratio))
                c_h = b_h - (b_h * (1 - ratio))
                depth_x1, depth_y1 = b_depth_x1 + int(c_w / 2), b_depth_y1 + int(c_h / 2)
                depth_x2, depth_y2 = b_depth_x2 - int(c_w / 2), b_depth_y2 - int(c_h / 2)
                mean = depth_x2 - depth_x1
                if depth_y2 < 1450 and depth_y1 > 120 and depth_x2 < 2030:
                    return [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
                else:
                    return [[]]
            elif 0.3 < ratio < 0.5:
                c_w = b_w - (b_w * (1 - ratio))  # 横的长，多截一点，少留一点
                c_h = b_h - (b_h * ratio)  # 竖的短，少截掉一点，多留一些
                depth_x1, depth_y1 = b_depth_x1 + int(c_w / 2), b_depth_y1 + int(c_h / 2)
                depth_x2, depth_y2 = b_depth_x2 - int(c_w / 2), b_depth_y2 - int(c_h / 2)
                mean = depth_x2 - depth_x1
                if depth_y2 < 1450 and depth_y1 > 120 and depth_x2 < 2030:
                    return [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
                else:
                    return [[]]
            elif 0.1 < ratio <= 0.3:
                c_w = b_w - (b_w * (1 - ratio))  # 横的长，多截一点，少留一点
                c_h = b_h - (b_h * ratio)  # 竖的短，少截掉一点，多留一些
                depth_x1, depth_y1 = b_depth_x1 - int(c_w / 2), b_depth_y1 + int(c_h / 2)
                depth_x2, depth_y2 = b_depth_x2 + int(c_w / 2), b_depth_y2 - int(c_h / 2)
            elif ratio < 0.1:
                c_w = b_w * 1.2  # 横的太短，直接不截掉，全部保留竖着的像素
                c_h = b_h - (b_h * ratio)  # 竖的长，多截一点，少留一点
                depth_x1, depth_y1 = b_depth_x1 - int(c_w / 2), b_depth_y1 + int(c_h / 2)
                depth_x2, depth_y2 = b_depth_x2 + int(c_w / 2), b_depth_y2 - int(c_h / 2)
        else:  # 框太小，就不截掉了
            c_w = b_w * 2  # 横的太短，直接不截掉，全部保留竖着的像素
            c_h = b_h - (b_h * ratio)  # 竖的长，多截一点，少留一点
            depth_x1, depth_y1 = b_depth_x1 - int(c_w / 2), b_depth_y1 + int(c_h / 2)
            depth_x2, depth_y2 = b_depth_x2 + int(c_w / 2), b_depth_y2 - int(c_h / 2)
            mean = depth_x2 - depth_x1
            if depth_y2 < 1450 and depth_y1 > 120 and depth_x2 < 2030:
                return [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
            else:
                return [[]]

        mean = depth_x2 - depth_x1
        counts = [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
        count = b_h // (mean//2)
        for i in range(1, count):
            up_depth_x1, up_depth_y1 = depth_x1, depth_y1 - (mean * i)
            up_depth_x2, up_depth_y2 = depth_x2, depth_y2 - (mean * i)

            dw_depth_x1, dw_depth_y1 = depth_x1, depth_y1 + (mean * i)
            dw_depth_x2, dw_depth_y2 = depth_x2, depth_y2 + (mean * i)
            if up_depth_x1 > b_depth_x1 or dw_depth_y2 < b_depth_y2:
                counts.append([up_depth_x1, up_depth_y1, up_depth_x2, up_depth_y2, mean, shape])
                counts.append([dw_depth_x1, dw_depth_y1, dw_depth_x2, dw_depth_y2,mean, shape])
    for cut in counts:
        c_x1, c_y1, c_x2, c_y2, c_mean, c_shape = cut
        if c_y2 < 1450 and c_y1 > 120 and c_x2 < 2030:
            c_box.append([c_x1, c_y1, c_x2, c_y2, c_mean, c_shape])
    return c_box


def S_gd_mapping2depth(x1, y1,x2,y2):
    # 获取图像的高度和宽度
    c_box = []

    d_height, d_width = 1536, 2048
    weight, height = 2008, 1518
    d_x1 = abs(x1 + (x1 * ( 0.015*(x1 / weight))))
    # d_x1 = x1
    d_y1 = abs(y1 + (y1 * ( 0.035*(y1 / height))))
    d_x2 = abs(x2 + (x2 * ( 0.02*(x2 / weight))))
    # d_x2 = x2
    d_y2 = abs(y2 + (y2 * ( 0.035*(y2 / height))))
    # print("\tRGB", x1, "Dep", d_x1, "\tRGB", y1, "Dep", d_y1, "\tRGB", x2, "Dep", d_x2, "\tRGB", y2, "Dep", d_y2)
    b_depth_x1 = int(d_x1 * d_width / weight)
    b_depth_y1 = int(d_y1 * d_height / height)
    b_depth_x2 = int(d_x2 * d_width / weight)
    b_depth_y2 = int(d_y2 * d_height / height)
    # print("\tRGB", x1, "Dep", b_depth_x1, "\tRGB", y1, "Dep", b_depth_y1, "\tRGB", x2, "Dep", b_depth_x2, "\tRGB", y2, "Dep", b_depth_y2)



    # 判断是否为横向标签框 根据标签框的形状计算不同的差值
    is_horizontal = abs(b_depth_x2 - b_depth_x1) > abs(b_depth_y2 - b_depth_y1)
    b_w, b_h = b_depth_x2 - b_depth_x1, b_depth_y2 - b_depth_y1  # 获取宽高, 用来以不同方式摘取待分析图像内容

    # 横的比竖的数值大，is_horizontal为Ture，横向标签框
    if is_horizontal:
        ratio = b_h / b_w
        shape = 0
        if b_depth_y2 < 395 or b_depth_y1 < 395:
            b_depth_y1 = b_depth_y1 - int((395 - b_depth_y1) * 0.04)
            b_depth_y2 = b_depth_y2 - int((395 - b_depth_y2) * 0.04)
        else:
            b_depth_y1 = b_depth_y1 + int((b_depth_y1 - 395) * 0.08)
            b_depth_y2 = b_depth_y2 + int((b_depth_y2 - 395) * 0.08)

        c_w = b_w - (b_w * ratio)  # 横的长，多截一点，少留一点
        c_h = b_h * 0.5  # 竖的太短，直接不截掉，全部保留竖着的像素
        depth_x1, depth_y1 = b_depth_x1 + int(c_w / 1.5), b_depth_y1 - int(c_h / 1.5)
        depth_x2, depth_y2 = b_depth_x2 - int(c_w / 1.5), b_depth_y2 + int(c_h / 1.5)
        # 整理坐标点
        depth_x1, depth_x2 = min(depth_x1, depth_x2), max(depth_x1, depth_x2)
        depth_y1, depth_y2 = min(depth_y1, depth_y2), max(depth_y1, depth_y2)

        mean = depth_y2 - depth_y1 - 10
        counts = [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
        # counts.append([b_depth_x1, b_depth_y1, b_depth_x2, b_depth_y2, mean, shape])
        count = b_w // (mean // 2)
        for i in range(1, count):
            up_depth_x1, up_depth_y1 = depth_x1 - (mean * i), depth_y1
            up_depth_x2, up_depth_y2 = depth_x2 - (mean * i), depth_y2

            dw_depth_x1, dw_depth_y1 = depth_x1 + (mean * i), depth_y1
            dw_depth_x2, dw_depth_y2 = depth_x2 + (mean * i), depth_y2

            # 整理坐标点
            up_depth_x1, up_depth_x2 = min(up_depth_x1, up_depth_x2), max(up_depth_x1, up_depth_x2)
            up_depth_y1, up_depth_y2 = min(up_depth_y1, up_depth_y2), max(up_depth_y1, up_depth_y2)
            dw_depth_x1, dw_depth_x2 = min(dw_depth_x1, dw_depth_x2), max(dw_depth_x1, dw_depth_x2)
            dw_depth_y1, dw_depth_y2 = min(dw_depth_y1, dw_depth_y2), max(dw_depth_y1, dw_depth_y2)
            if up_depth_x1 > b_depth_x1 + 10:
                counts.append([up_depth_x1, up_depth_y1, up_depth_x2, up_depth_y2, mean, shape])
            if dw_depth_x2 < b_depth_x2 - 10:
                counts.append([dw_depth_x1, dw_depth_y1, dw_depth_x2, dw_depth_y2, mean, shape])
    # 竖的比横的数值大，is_horizontal为False，竖向标签框
    else:
        # 纵向标签框
        shape = 1
        ratio = b_w / b_h
        if b_depth_y2 < 395 or b_depth_y1 < 395:

            b_depth_y1 = b_depth_y1 - int((395 - b_depth_y1) * 0.04)
            b_depth_y2 = b_depth_y2 - int((395 - b_depth_y2) * 0.04)
        else:
            b_depth_y1 = b_depth_y1 + int((b_depth_y1 - 395) * 0.08)
            b_depth_y2 = b_depth_y2 + int((b_depth_y2 - 395) * 0.08)

        c_w = b_w * 0.5  # 横的太短，直接不截掉，全部保留竖着的像素
        c_h = b_h - (b_h * ratio)  # 竖的长，多截一点，少留一点
        depth_x1, depth_y1 = b_depth_x1 - int(c_w / 1.5), b_depth_y1 + int(c_h / 1.5)
        depth_x2, depth_y2 = b_depth_x2 + int(c_w / 1.5), b_depth_y2 - int(c_h / 1.5)
        # 整理坐标点
        depth_x1, depth_x2 = min(depth_x1, depth_x2), max(depth_x1, depth_x2)
        depth_y1, depth_y2 = min(depth_y1, depth_y2), max(depth_y1, depth_y2)

        mean = depth_x2 - depth_x1 - 10
        counts = [[depth_x1, depth_y1, depth_x2, depth_y2, mean, shape]]
        # counts.append([b_depth_x1, b_depth_y1, b_depth_x2, b_depth_y2, mean, shape])
        count = b_h // (mean // 2)
        for i in range(1, count):
            up_depth_x1, up_depth_y1 = depth_x1, depth_y1 - (mean * i)
            up_depth_x2, up_depth_y2 = depth_x2, depth_y2 - (mean * i)

            dw_depth_x1, dw_depth_y1 = depth_x1, depth_y1 + (mean * i)
            dw_depth_x2, dw_depth_y2 = depth_x2, depth_y2 + (mean * i)

            # 整理坐标点
            up_depth_x1, up_depth_x2 = min(up_depth_x1, up_depth_x2), max(up_depth_x1, up_depth_x2)
            up_depth_y1, up_depth_y2 = min(up_depth_y1, up_depth_y2), max(up_depth_y1, up_depth_y2)
            dw_depth_x1, dw_depth_x2 = min(dw_depth_x1, dw_depth_x2), max(dw_depth_x1, dw_depth_x2)
            dw_depth_y1, dw_depth_y2 = min(dw_depth_y1, dw_depth_y2), max(dw_depth_y1, dw_depth_y2)
            if up_depth_y1 > b_depth_y1 + 10:
                # print(up_depth_y1,"----------------------", b_depth_y1+20)
                counts.append([up_depth_x1, up_depth_y1, up_depth_x2, up_depth_y2, mean, shape])
            if dw_depth_y2 < b_depth_y2 - 10:
                # print(dw_depth_y2,"----------------------",b_depth_y2-10)
                counts.append([dw_depth_x1, dw_depth_y1, dw_depth_x2, dw_depth_y2, mean, shape])
    # print("counts：：：：：：\n",counts)
    for cut in counts:
        c_x1, c_y1, c_x2, c_y2, c_mean, c_shape = cut
        if 80 < c_y1 and c_y2 < 1230:
            if 875 < c_y1 and c_x1 < 358:
                if c_y1 > 500:
                    c_x1, c_x2 = c_x1 - 7, c_x2 - 7
            c_box.append([c_x1, c_y1, c_x2, c_y2, c_mean, c_shape])
    return c_box


# dets: 检测出的图中某一类别的bbox及对应的置信度，列表中的每个元素为[x1, y1, x2, y2, confidence]；
# thresh: 设定的IoU阈值
def nms(dets, thresh):
    """
    处理tensor格式的nms
    :param dets: 坐标点
    :param thresh: IOU阈值
    :return:去重后的坐标点、置信度、类别
    """
    # 确保 dets 在 CPU 上，如果你确定它在 GPU 上，你可以省略这一步
    dets = dets.cpu()
    # 预处理
    # 提取各个bbox的位置，即左上角和右下角坐标，用于后续计算IoU里的各种面积
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]  # 提取各个bbox的置信度
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 计算各个bbox的面积
    # Step1：将原始列表中的所有bbox按照置信度从高到低进行排序
    print("scores",scores.argsort())
    # order = scores.argsort()[::-1]  # 在从大到小排序后返回索引值，即order[0]表示scores列表里最大值的索引
    order = scores.argsort(descending=True)
    keep = []  # 保存筛选后重叠度低于阈值的bbox，注意，返回的是原始列表中要保留的bbox的索引
    # Step5：重复Step2-4，直到原始列表中不再有bbox
    # print("order",order.numel())
    # while order.size > 0:
    while order.numel() > 0:
        # Step2：选取当前置信度最高的bbox，记为$b$，并将其放到最终的结果列表里
        i = order[0]  # scores列表里置信度最高的bbox对应的的索引
        keep.append(i)  # 将当前这个框得最准的bbox保存到输出结果里
        # Step3：计算剩余所有bbox与$b$的IoU，将IoU大于阈值的bbox全部删除
        # 首先要计算出bbox重叠部分的左上角和右下角坐标
        # 即取两个bbox中左上角值较大者和右下角值较小者
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算重叠部分的宽和高，如果算出来是负值则说明两个bbox不重叠，因此要把相应的宽/高置0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        # 计算重叠部分的面积
        inter = w * h
        # 计算$b$与剩余所有bbox的IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 将IoU大于阈值的bbox全部删除，也就是把重叠度较小的bbox给保留下来
        inds = np.where(ovr <= thresh)[0]
        # Step4：从原始列表中删除$b$
        # 由于之前的操作都是算的剩余的bbox与$b$的关系，也就是排除了原始列表的首个元素（从算xx1开始）
        # 所以上面得到的inds要+1才是真正对应到原始列表中的索引，这个过程也就自动地把$b$拿掉了
        order = order[inds + 1]

    # Step6：返回结果列表，即为NMS筛选后的结果
    keep_tensor  = torch.tensor(keep, device=dets.device)
    xyxy = torch.index_select(dets,0, keep_tensor)
    return xyxy

import cv2
def rotate_image_and_boxes(image, boxes, angle=180):
    """
        对图像进行180°旋转，并在镜像后的图像上绘制边界框。

        参数:
        image -- 原始图像（numpy数组）
        boxes -- 边界框列表，每个边界框是(x_min, y_min, x_max, y_max)格式的元组

        返回:
        rotated, rotated_boxes -- 镜像后并绘制了边界框的图像（numpy数组）,转换后的检测框坐标点
    """
    # 旋转图像
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

        # 转换边界框坐标
    rotated_boxes = []
    x1, y1, x2, y2 = boxes[0], boxes[1], boxes[2], boxes[3]
    x1, y1 = w - x2, h - y2
    x2, y2 = w - x1, h - y1
    rotated_boxes.append((x1, y1, x2, y2))

    return rotated, rotated_boxes