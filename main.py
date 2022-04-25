
import random
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from main_win.win import Ui_mainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon
from models.common import DetectMultiBackend
import sys
import os
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
from utils.CustomMessageBox import MessageBox
# LoadWebcam 的最后一个返回值改为 self.cap
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, Annotator, plot_one_box, plot_one_box_PIL
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.capnums import Camera
from dialog.rtsp_win import Window


#定义一个字典 进行误差修正
food_fixed = {
    'Marinated cold cucumber':'scrambled eggs with loofah',
    'Duck neck':'braised pork',
    'Fried and stewed hairtail in soy sauce':'pork chop',
    'Saut茅ed Sliced Pork, Eggs and Black Fungus':'stir-fried dried tofu with sauerkraut',
    'Beef Seasoned with Soy Sauce':'braised fish',
    'Saut茅ed Shrimps with Celery':'sauteed mushrooms with ham',
    'Steamed Dazha Crabs':'carrot stir-fried meat',
    'Fried Baked Scallion Pancake':'egg cake',
    'Stir-fried Spinach':'sauteed cauliflower',
    'Scrambled Egg with green pepper':'scrambled eggs with tofu',
    'Yu-Shiang Shredded Pork':'Mao Xue Wang pork',
    'Hot and dry noodle':'stir-fry vermicelli',
    'Millet congee':'Rice',
    'Saut茅ed Sweet Corn with Pine Nuts':'rice',
    'Deep Fried lotus root':'eggs cake',
    'Braised beef with brown sauce':'scrambled eggs with tomatoes',
    'Fried green peppers':'stir-fried bamboo shoot slices',
    'Yangzhou fried rice':'rice',
    'Stewed chicken with mushroom':'Mao Xue Wang pork',
    'Fried Lamb with Cumin':'Saute dried tofu',
    'Saut茅ed Sour Beans with Minced Pork':'Scrambled eggs with squeezed vegetables',
    'Pork Ribs and lotus root soup':'Winter melon stewed fish balls',
    'Tremella and red dates soup':'Winter melon stewed fish balls',
    'Braised Tofu':'Winter melon stewed fish balls',
    'Saut茅ed pork with mushrooms':'White chopped chicken',
    'Sweet mung bean soup':'rice'
}

#七大营养素：蛋白质、脂肪、碳水化合物、无机盐、维生素、纤维素、水
yingyang_dic = {
    'danbaizhi': 0,
    'tanshui':0,
    'weishengsu':0,
    'shui':0
}


danbaizhi = ['蛋白质摄入充足,\n','蛋白质补充充足,\n','补充了充足的蛋白质,\n','蛋白质摄入较多,\n','蛋白质摄入达到要求,\n','补充了一定的蛋白质,\n']
danbaizhi_0 = '蛋白质'
tanshui = ['保证了一定量的碳水摄入,\n','碳水化合物补充充足,\n','主食较丰富,\n','碳水摄入达到要求,\n']
tanshui_0 = '碳水化合物'
weishengsu = ['补充了一定量的维生素,\n','维生素补充达到要求,\n','保证了每日必须的维生素摄入,\n','摄入了一定量的维生素,\n']
weishengsu_0 = '维生素'
shui = ['补充了水分,\n']
shui_0 = '水分'





class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    # 发送信号：正在检测/暂停/停止/检测结束/错误报告
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './exp7_best.pt'           # 设置权重
        self.current_weight = './exp7_best.pt'    # 当前权重
        self.source = '0'                       # 视频源
        self.conf_thres = 0.25                  # 置信度
        self.iou_thres = 0.45                   # iou
        self.jump_out = False                   # 跳出循环
        self.is_continue = True                 # 继续/暂停
        self.percent_length = 1000              # 进度条
        self.rate_check = True                  # 是否启用延时
        self.rate = 100                         # 延时HZ
        self.line_thickness=10                  # 框的宽度

    @torch.no_grad()
    def run(self,
            imgsz=(640, 640),  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=True,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            # line_thickness=10,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            ):

        # Initialize
        try:
            # device = select_device(device)
            # half &= device.type != 'cpu'  # half precision only supported on CUDA
            #
            # # Load model
            # model = attempt_load(self.weights, map_location=device)  # load FP32 model
            # num_params = 0
            # for param in model.parameters():
            #     num_params += param.numel()
            # stride = int(model.stride.max())  # model stride
            # imgsz = check_img_size(imgsz, s=stride)  # check image size
            # names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            # if half:
            #     model.half()  # to FP16
            #
            # # Dataloader
            # if self.source.isnumeric() or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
            #     view_img = check_imshow()
            #     cudnn.benchmark = True  # set True to speed up constant image size inference
            #     dataset = LoadWebcam(self.source, img_size=imgsz, stride=stride)
            #     # bs = len(dataset)  # batch_size
            # else:
            #     dataset = LoadImages(self.source, img_size=imgsz, stride=stride)
            #
            # # Run inference
            # if device.type != 'cpu':
            #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

            #Load model
            device = select_device(device)
            model = DetectMultiBackend(self.weights, device=device, dnn=dnn)
            stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
            imgsz = check_img_size(imgsz, s=stride)  # check image size

            #Half
            half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
            if pt or jit:
                model.model.half() if half else model.model.float()

            # Dataloader
            dataset = LoadImages(self.source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
            vid_path, vid_writer = [None] * bs, [None] * bs

            # Run inference
            model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
            dt, seen = [0.0, 0.0, 0.0], 0
            for path, im, im0s, self.vid_cap, s in dataset:
                im = torch.from_numpy(im).to(device)
                im = im.half() if half else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            #pred = model(im, augment=augment, visualize=False)

            # NMS
            #pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms, max_det=max_det)

            # # Process predictions
            # for i, det in enumerate(pred):  # per image
            #     seen += 1
            #     p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            #
            #     s += '%gx%g ' % im.shape[2:]  # print string
            #     gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            #     imc = im0.copy() if save_crop else im0  # for save_crop
            #     annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            #     if len(det):
            #         # Rescale boxes from img_size to im0 size
            #         det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            #
            #         # Print results
            #         for c in det[:, -1].unique():
            #             n = (det[:, -1] == c).sum()  # detections per class
            #             s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            #
            #         # Write results
            #         for *xyxy, conf, cls in reversed(det):
            #             c = int(cls)  # integer class
            #             label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
            #             annotator.box_label(xyxy, label, color=colors(c, True))



            count = 0
            # 跳帧检测
            jump_count = 0
            start_time = time.time()
            dataset = iter(dataset)
            while True:
                # 手动停止
                if self.jump_out:
                    self.vid_cap.release()
                    self.send_percent.emit(0)
                    self.send_msg.emit('停止')
                    break
                # 临时更换模型
                # if self.current_weight != self.weights:
                #     # Load model
                #     model = attempt_load(self.weights, map_location=device)  # load FP32 model
                #     num_params = 0
                #     for param in model.parameters():
                #         num_params += param.numel()
                #     stride = int(model.stride.max())  # model stride
                #     imgsz = check_img_size(imgsz, s=stride)  # check image size
                #     names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                #     if half:
                #         model.half()  # to FP16
                #     # Run inference
                #     if device.type != 'cpu':
                #         model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                #     self.current_weight = self.weights
                # 暂停开关
                if self.is_continue:
                    path, img, im0s, self.vid_cap, s = next(dataset)
                    # jump_count += 1
                    # if jump_count % 5 != 0:
                    #     continue
                    count += 1
                    # 每三十帧刷新一次输出帧率
                    if count % 30 == 0 and count >= 30:
                        fps = int(30/(time.time()-start_time))
                        self.send_fps.emit('fps：'+str(fps))
                        start_time = time.time()
                    if self.vid_cap:
                        percent = int(count/self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)*self.percent_length)
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length

                    statistic_dic = {name: 0 for name in names}
                    #print(statistic_dic)
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img[None]

                    # pred = model(img, augment=augment)[0]
                    pred = model(img, augment=augment, visualize=False)

                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms, max_det=max_det)
                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        im0 = im0s.copy()

                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Write results#此处循环画框
                            for *xyxy, conf, cls in reversed(det):
                                c = int(cls)  # integer class
                                #name fixed
                                if names[c] in food_fixed.keys():
                                    #print('tiaoguo')
                                    fixed_name = food_fixed.get(names[c])
                                    statistic_dic.pop(names[c])
                                    statistic_dic[fixed_name] = 0
                                    statistic_dic[fixed_name] += 1
                                    #label = fixed_name
                                    label = None if hide_labels else (fixed_name if hide_conf else f'{fixed_name} {conf:.2f}')
                                else:
                                    statistic_dic[names[c]] += 1
                                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                    print(label)
                                    print(type(label))


                                #im0 = plot_one_box_PIL(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)  # 中文标签画框，但是耗时会增加
                                plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                             line_thickness=self.line_thickness)


                    # 控制视频发送频率
                    if self.rate_check:
                        time.sleep(1/self.rate)
                    # print(type(im0s))
                    self.send_img.emit(im0)
                    self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                    self.send_statistic.emit(statistic_dic)
                    if percent == self.percent_length:
                        self.send_percent.emit(0)
                        self.send_msg.emit('检测结束')
                        # 正常跳出循环
                        break

        except Exception as e:
            self.send_msg.emit('%s' % e)


class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False
        # win10的CustomizeWindowHint模式，边框上面有一段空白。
        # 不想看到空白可以用FramelessWindowHint模式，但是需要重写鼠标事件才能通过鼠标拉伸窗口，比较麻烦
        # 不嫌麻烦可以试试, 写了一半不想写了，累死人
        self.setWindowFlags(Qt.CustomizeWindowHint)
        # self.setWindowFlags(Qt.FramelessWindowHint)
        # 自定义标题栏按钮
        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        self.closeButton.clicked.connect(self.close)

        # 定时清空自定义状态栏上的文字
        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # 自动搜索模型
        self.comboBox.clear()
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/'+x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        # yolov5线程
        self.det_thread = DetThread()
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type           # 权重
        self.det_thread.source = '0'                                    # 默认打开本机摄像头，无需保存到配置文件
        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.raw_video))
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))
        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        self.fileButton.clicked.connect(self.open_file)
        self.cameraButton.clicked.connect(self.chose_cam)
        self.rtspButton.clicked.connect(self.chose_rtsp)

        self.runButton.clicked.connect(self.run_or_continue)
        self.stopButton.clicked.connect(self.stop)

        self.comboBox.currentTextChanged.connect(self.change_model)
        # self.comboBox.currentTextChanged.connect(lambda x: self.statistic_msg('模型切换为%s' % x))
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))#这里改变以下，改成调整line_thickness
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.checkBox.clicked.connect(self.checkrate)
        self.load_setting()

    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def checkrate(self):
        if self.checkBox.isChecked():
            # 选中时
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False

    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.67:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='提示', text='请稍等，正在加载rtsp视频流', time=1000, auto=True).exec_()
            self.det_thread.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.statistic_msg('加载rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.statistic_msg('%s' % e)

    def chose_cam(self):
        try:
            self.stop()
            # MessageBox的作用：留出2秒，让上一次摄像头安全release
            MessageBox(
                self.closeButton, title='提示', text='请稍等，正在检测摄像头设备', time=2000, auto=True).exec_()
            # 自动检测本机有哪些摄像头
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.cameraButton.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).x()
            y = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).y()
            y = y + self.cameraButton.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec_(pos)
            if action:
                self.det_thread.source = action.text()
                self.statistic_msg('加载摄像头：{}'.format(action.text()))
        except Exception as e:
            self.statistic_msg('%s' % e)

    # 导入配置文件
    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            check = 0
            new_config = {"iou": 0.26,
                          "conf": 0.33,
                          "rate": 10,
                          "check": 0
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            iou = config['iou']
            conf = config['conf']
            rate = config['rate']
            check = config['check']
        self.confSpinBox.setValue(iou)
        self.iouSpinBox.setValue(conf)
        self.rateSpinBox.setValue(rate)
        self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x*100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x/100)
            self.det_thread.conf_thres = x/100
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x*100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x/100)
            self.det_thread.iou_thres = x/100

        elif flag == 'rateSpinBox': #这里替换了，修改linethickness
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            self.det_thread.line_thickness = x #x * 10
        else:
            pass

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        # self.qtimer.start(3000)   # 3秒后自动清除

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.statistic_msg('模型切换为%s' % x)

    def open_file(self):
        # source = QFileDialog.getOpenFileName(self, '选取视频或图片', os.getcwd(), "Pic File(*.mp4 *.mkv *.avi *.flv "
        #                                                                    "*.jpg *.png)")
        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, '选取视频或图片', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                          "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('加载文件：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            # 切换文件后，上一次检测停止
            self.stop()

    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    # 继续/暂停
    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = '摄像头设备' if source.isnumeric() else source
            self.statistic_msg('正在检测 >> 模型：{}，文件：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))
        else:
            self.det_thread.is_continue = False
            self.statistic_msg('暂停')

    # 退出检测循环
    def stop(self):
        self.det_thread.jump_out = True

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            # QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        # self.setCursor(QCursor(Qt.ArrowCursor))

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # 保持纵横比
            # 找出长边
            if iw > ih:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # 实时统计
    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' '+str(i[0]) + '：' + str(i[1]) for i in statistic_dic]

            #results.append('\n')
            yingyangjiance = ''
            danbaizhi_flag = ['eggs' , 'pork' , 'tofu' , 'fish' , 'ham' , 'meat' , 'chicken']
            weishengsu_flag = ['loofah' , 'sauerkraut' , 'mushrooms' , 'carrot' , 'cauliflower',\
                     'vermicelli' , 'tomatoes' ,'bamboo', 'vegetables' , 'melon']
            tanshui_flag = ['rice' , 'Rice']
            shui_flag = ['soup']

            _yingyang_dic = yingyang_dic.copy()
            for food in statistic_dic:
                for j in danbaizhi_flag:
                    if (j in food[0] ) and _yingyang_dic['danbaizhi']==0 :
                        _yingyang_dic['danbaizhi'] += 1
                        yingyangjiance = yingyangjiance + random.choice(danbaizhi)
                for j in weishengsu_flag:
                    if (j in food[0] ) and _yingyang_dic['weishengsu']==0:
                        _yingyang_dic['weishengsu'] += 1
                        yingyangjiance = yingyangjiance + random.choice(weishengsu)
                for j in shui_flag:
                    if (j in food[0] )and _yingyang_dic['shui']==0:
                        _yingyang_dic['shui'] += 1
                        yingyangjiance = yingyangjiance + random.choice(shui)
                for j in tanshui_flag:
                    if (j in food[0] ) and _yingyang_dic['tanshui']==0:
                        _yingyang_dic['tanshui'] += 1
                        yingyangjiance = yingyangjiance + random.choice(tanshui)
            queshao = '缺少'
            zongti_flag = 1  # 1是均衡，0是不均衡
            if _yingyang_dic['danbaizhi'] == 0:
                queshao = queshao + danbaizhi_0 + ','
                zongti_flag = 0
            if _yingyang_dic['weishengsu'] == 0:
                queshao = queshao + weishengsu_0 + ','
                zongti_flag = 0
            if _yingyang_dic['tanshui'] == 0:
                queshao = queshao + tanshui_0 + ','
                zongti_flag = 0
            if _yingyang_dic['shui'] == 0:
                queshao = queshao + shui_0 + ','
                #zongti_flag = 0

            if zongti_flag == 0:
                yingyangjiance = '营养分析:\n' + yingyangjiance + queshao +'\n' + '总体来看，饮食结构欠平衡'
                results.append(yingyangjiance)
            else:
                yingyangjiance = '营养分析:\n' + yingyangjiance + '总体来看，饮食结构较平衡'
                results.append(yingyangjiance)


            # results.append('营养分析:吃的比美\n国中产好')
            self.resultWidget.addItems(results)
            #以下来加入营养分析
            #self.resultWidget.addItems("营养分析：")


        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        # 如果摄像头开着，先把摄像头关了再退出，否则极大可能可能导致检测线程未退出
        self.det_thread.jump_out = True
        # 退出时，保存设置
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.confSpinBox.value()
        config['conf'] = self.iouSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        MessageBox(
            self.closeButton, title='提示', text='请稍等，正在关闭程序。。。', time=2000, auto=True).exec_()
        sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())
