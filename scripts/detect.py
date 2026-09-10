import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import glob
import os
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import math
import re


def get_number(filename):
    return int(''.join(filter(str.isdigit, filename)))

# 将小数形式的经度转为度分秒形式的字符串


def lng_to_dms(lng):
    d = int(lng)
    m = int((lng - d) * 60)
    s = round((lng - d - m / 60) * 3600, 2)
    return f"{d}°{m}'{s}\"E"

# 将小数形式的纬度转换为度分秒形式的字符串


def lat_to_dms(lat):
    d = int(lat)
    m = int((lat - d) * 60)
    s = round((lat - d - m / 60) * 3600, 2)
    return f"{d}°{m}'{s}\"N"


class Image:
    def __init__(self, img, image_name):
        self.img = img
        self.image_name = image_name

    def save_result(self, save_dir, class_count):
        save_path = os.path.join(save_dir, self.image_name + ".txt")
        with open(save_path, "w") as f:
            f.write("Image name: {}\n".format(self.image_name))
            f.write("Class counts: {}\n".format(class_count))
            f.write("\n")


def detect(opt):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith(
        '.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name,
                    exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    print('imgsz', imgsz)
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        print('dataset', dataset)
    else:
        dataset = LoadImages(source, img_size=(3956, 5280), stride=stride)
        print('dataset', dataset)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    t0 = time.time()

    # 假设大图中心点的经纬度为center_lng,center_lat
    center_lng = 111.151111  # 东经111'9'44'
    center_lat = 32.349722  # 北纬32'20'59'

    # 假设大图的分辨率为img_width*img_height
    img_width = 8192
    img_height = 5460

    # 假设将大图切割为n_col列，n_row行的小图
    n_col = 6
    n_row = 5
    #假设每个小图的分辨率为sub_img_width *sub_img_height
    sub_img_width = img_width // n_col
    sub_img_height = img_height // n_row

    # 假设拍摄高度为h
    h = 100

    # 循环访问包含图像路径、调整大小的图像、原始图像和视频捕获的数据据
    for path, img, im0s, vid_cap in dataset:
        # Cut images (1920*1439 -> [640*480 + 640*480 + 640*479]*3)
        # 创建临时变量来存储原始图像和调整大小的图像
        tmp_img = img
        tmp_im0s = im0s
        # 创建嵌套循环以裁剪图像。有6行和5列的小块，切成
        for m in range(n_col):
            for n in range(n_row):
                # 根据循环的当前迭代计算垂直和水平裁剪坐标的最小值和最大值。
                ycrop_min = imgsz*n
                ycrop_max = imgsz*(n+1)
                xcrop_min = imgsz*m
                xcrop_max = imgsz*(m+1)

                sub_img_center_x = (n+0.5) * sub_img_width
                sub_img_center_y = (m+0.5) * sub_img_height

                # 计算每个小图的中心点的经纬度
                dx = (sub_img_center_x-img_width/2) * 0.00001/h
                dy = (img_height / 2 - sub_img_center_y)*0.00001/h
                sub_img_center_lng = center_lng+dx
                sub_img_center_lat = center_lat+dy

                # 将小数形式的经纬度转换为度分秒形式的字符串
                sub_img_center_lng_str = lng_to_dms(sub_img_center_lng)
                sub_img_center_lat_str = lat_to_dms(sub_img_center_lat)
                # 输出每个小图的中心点经纬度
                print(
                    f"第{m+1}行第{n+1}列小图中心点经纬度为: ({sub_img_center_lng_str}, {sub_img_center_lat_str})")
                # 如果循环的当前迭代位于最后一列或最后一行，请相应地调整裁剪坐标以避免超出范围。
                if(m == 5):
                    xcrop_min = 5280-1-imgsz
                    xcrop_max = 5280-1
                if(n == 4):
                    ycrop_min = 3956-1-imgsz
                    ycrop_max = 3956-1
                # 根据计算的裁剪坐标裁剪原始图像和调整大小的图像
                # 将原始图像裁剪存储在im0s中
                # 记录裁剪图像所需的时间
                tic = time_synchronized()
                print('ycrop_min', ycrop_min)
                print('ycrop_max', ycrop_max)
                print('xcrop_min', xcrop_min)
                print('xcrop_max', xcrop_max)
                im0s = tmp_im0s[ycrop_min:ycrop_max, xcrop_min:xcrop_max, :]
                img = tmp_img[:, ycrop_min:ycrop_max, xcrop_min:xcrop_max]
                tok = time_synchronized()

                # Print time (Crop)
                print(f'Crop time: ({tok - tic:.5f})')

                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(
                    pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                t2 = time_synchronized()

                # Apply Classifier
                if classify:
                    pred = apply_classifier(pred, modelc, img, im0s)

                class_ids = []

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # 获取检测到的类别的id
                        det_class_ids = det[:, -1].detach().cpu().numpy()
                        class_ids.extend(list(det_class_ids))
                    if webcam:  # batch_size >= 1
                        p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                        ), dataset.count
                    else:
                        p, s, im0, frame = path, '', im0s, getattr(
                            dataset, 'frame', 0)
                    class_count = {}
                    class_count = {class_id: 0 for class_id in range(1)}

                    for class_id in class_ids:
                        if class_id in class_count:
                            class_count[class_id] += 1
                        else:
                            class_count[class_id] = 1
                    print(class_count)

                    # print("Class counts:", class_count[class_id])
                    p = Path(p)  # to Path

                    # save_path = str(save_dir / 'result')+str(m)+str(n)+'.jpg'  # img.jpg p.name
                    save_path = str(
                        save_dir / ('result'+str(m)+str(n)+'_'+p.stem+'.jpg'))
                    print('save_path', save_path)

                    # print('save_path',save_path)
                    txt_path = str(save_dir / 'labels' / p.stem) + \
                        ('' if dataset.mode ==
                         'image' else f'_{frame}')  # img.txt
                    # txt_path=Path(save_dir)/'result'/(p.stem+'.txt')
                    # print('txt_path',txt_path)
                    # print('p.stem',p.stem)
                    # print(f"第{m+1}行第{n+1}列小图中心点经纬度为: ({sub_img_center_lng}, {sub_img_center_lat})")
                    # 将检测到的类别数量写入txt文件
                    filename = os.path.basename(path)
                    # with open(str(save_dir / 'result') +filename + str(m+1) + str(n+1) + '.txt', 'w',encoding='utf-8') as f:
                    #     f.write('Image name: '+filename + '\n' + 'class counts: '+str(class_count[class_id]) + '\n'
                    #         +"第{}行第{}列小图中心点经纬度为{}{}:".format(m,n,sub_img_center_lng,sub_img_center_lat))
                    with open(str(save_dir / 'result') + filename + str(m+1) + str(n+1) + '.txt', 'w', encoding='utf-8') as f:
                        f.write('Image name: ' + filename + '\n' + 'class counts: ' + str(class_count[class_id]) + '\n'
                                + "经纬度:{}  {}".format(sub_img_center_lng_str, sub_img_center_lat_str))

                    # with open(txt_path,'w') as f:
                    #     f.write('\n'.join([f'{k}: {v}' for k, v in class_count.items()]))
                    s += '%gx%g ' % img.shape[2:]  # print string
                    # normalization gain whwh
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # add to string
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]}{conf:.2f}'
                        im0s = np.ascontiguousarray(im0s)
                        plot_one_box(xyxy, im0s, label=label,
                                     color=colors[int(cls)], line_thickness=3)

                        # img = Image(im0s, os.path.basename(path))
                        # img.save_result(save_dir,class_count[class_id])

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(
                                1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            # label format
                            line = (
                                cls, *xywh, conf) if opt.save_conf else (cls, *xywh)

                            # with open(txt_path + '.txt', 'a') as f:
                            #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            im0 = np.ascontiguousarray(im0s)
                            plot_one_box(xyxy, im0, label=label,
                                         color=colors[int(cls)], line_thickness=3)

                    # Print time (inference + NMS)
                    print(f'{s}Done. ({t2 - t1:.3f}s)')

                    # Stream results
                    if view_img:
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'image':
                            # cv2.imwrite(save_path, im0)
                            pass
                        else:  # 'video' or 'stream'
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(
                                        cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(
                                        cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path += '.mp4'
                                vid_writer = cv2.VideoWriter(
                                    save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

    # 指定要遍历的目录路径和合并后的txt文件路径
    txt_dir = '/home/xcn/new-data/z/YOLO1/runs/detect/exp30'
    merged_txt_path = '/home/xcn/new-data/z/YOLO1/runs/detect/1.txt'

    # 遍历目录中的所有txt文件
    txt_files = sorted(glob.glob(os.path.join(txt_dir, '*.txt')),
                       key=lambda x: int(os.path.basename(x).split('.')[1].split('JPG')[1]))
    # print('txt_files',txt_files)

    # 存储每个图像的class counts的字典
    total_counts = {}
    info = []

    # 合并所有的txt文件到一个大的txt文件中
    with open(merged_txt_path, 'w') as f:
        for txt_file in txt_files:
            # 从文件名中获取图像名称
            img_name = os.path.splitext(os.path.basename(txt_file))[0]
            # print('经度',sub_img_center_lng_str)
            # 读取txt文件内容并获取class counts
            with open(txt_file, 'r') as f_txt:
                lines = f_txt.readlines()
                jingwei = lines[2]
                # print('lines',lines)
                # print('jingwei',jingwei)
                counts = lines[-2].split(':')[1].strip()
                # print('counts',counts)
        # 将class counts添加到total_counts字典中
            total_counts[img_name] = counts
            # 将图像名称和class counts写入到大的txt文件中
            f.write('image name:{}   class counts:{}   {}\n'.format(
                img_name, counts, jingwei))
            # f.write('class counts:{}\n'.format(counts))
            # f.write('经纬度:{}\n'.format(jingwei))
            # f.write(f' {img_name}:   {counts}   {jingwei}\n')
            # f.write("第{}行第{}列小图中心点经纬度为{}{}".format(m, n, sub_img_center_lng_str, sub_img_center_lat_str))

            # print('info',info)
    # 计算总的class counts并写入到大的txt文件中
    total_class_count = sum(map(int, total_counts.values()))
    with open(merged_txt_path, 'a') as f:
        f.write(f'Total class counts: {total_class_count}\n')

    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='runs/train/exp64/weights/best.pt', help='model.pt path(s)')
    # parser.add_argument('--weights', nargs='+', type=str, default='weights/v5lite-s.onnx', help='model.pt path(s)')
    # file/folder, 0 for webcam
    #parser.add_argument(
    #    '--source', type=str, default='/home/xcn/new-data/z/YOLO1/data/wubeizi/test', help='source')
    parser.add_argument(
        '--source', type=str, default='/home/taylor/Ljm/Psdk30/Payload-SDK-master/build/bin/image', help='source')        
        
        
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        default=True, help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        default=False, help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3,
                        type=int, help='bounding box thickness(pixels)')
    parser.add_argument('--hide-labels', default=False,
                        action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True,
                        action='store_true', help='hide confidences')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt=opt)
