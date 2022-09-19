# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""
# ------------------------------ basic module ------------------------------
import os
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import json
import threading
import matching_utils as mu
import camera_config
# ------------------------------ basic module ------------------------------

# ------------------------------ api server ------------------------------
from flask import Flask, request, json

api = Flask(__name__)
# ------------------------------ api server ------------------------------

# ------------------------------ file manage ------------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# ------------------------------ file manage ------------------------------

# ------------------------------ models runtime ------------------------------
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
# ------------------------------ models runtime ------------------------------

# ------------------------------ functions ------------------------------
def midpoint(obj):
    # print(obj)
    return (obj[0][0] + obj[0][2]) / 2, (obj[0][1] + obj[0][3]) / 2


def distance(obj1, obj2, USESPACE=True):
    def space(obj):
        (x1, y1, x2, y2), _ = obj
        return abs(x1 - x2) * abs(y1 - y2)

    x1, y1 = midpoint(obj1)
    x2, y2 = midpoint(obj2)
    if USESPACE:
        sd = abs(space(obj1) - space(obj2))
    else:
        sd = 0
    # sd = 0
    if x2 > 1280:
        x2 -= 1280
    if x1 > 1280:
        x1 -= 1280
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 + sd


def transverse(obj):
    obj = list(obj)
    if obj[0] > 1280:
        obj[0] -= 1280
    return obj


def tensor2int(tensor):
    return [int(item.item()) for item in tensor]


def calculate(img0, det, bypass=False, normalize=False, COLOR_GRAY2BGR=False):
    global pictureDepth
    if bypass:
        return None

    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    imgL = img0[:, 0:1280]
    imgR = img0[:, 1280:2560]  # 割开双目图像

    # cv2.remap 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片
    # imgL = cv2.remap(imgL, camera_config.left_map1, camera_config.left_map2, cv2.INTER_LINEAR)
    # imgR = cv2.remap(imgR, camera_config.right_map1, camera_config.right_map2, cv2.INTER_LINEAR)

    if COLOR_GRAY2BGR:
        img1_rectified = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
        img2_rectified = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)

    threeD = camera_config.compute(imgL, imgR)

    # threeD[y][x] x:0~1080; y:0~720;   !!!!!!!!!!
    pictureDepth = {"depth": threeD, "det": det}
    return None


# ------------------------------ functions ------------------------------

# ------------------------------ model ------------------------------
@torch.no_grad()
def run(weights=ROOT / 'yolov5n.pt',  # model.pt path(s) # best.pt
        source='0',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(720, 2560),  # inference size (height, width)
        conf_thres=0.3,  # confidence threshold
        iou_thres=0.35,  # NMS IOU threshold
        max_det=10,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=True,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        send_image_depth_detection=True,  # send result to flask server
        crop=False,
        ):
    if __name__ != "__main__":
        view_img = False  # cv2 can only render on main thread
    global global_view_img, pictureDepth
    if not global_view_img:  # manual override
        view_img = False
    source = str(source)
    mu_ma = mu.match()
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        # view_img = check_imshow() #disable due to jetson sometimes don't have x-server up
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, jetson=True)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # print(im0.shape)

            list_of_label_left = []
            list_of_label_right = []
            # print(names)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img or send_image_depth_detection:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        if xyxy[0] < 1280 and xyxy[2] < 1280:
                            list_of_label_left.append([tensor2int(xyxy), names[c]])
                        elif xyxy[0] > 1280 and xyxy[2] > 1280:
                            list_of_label_right.append([list(xyxy), names[c]])
                        else:
                            if (xyxy[0] + xyxy[1])/2 < 1280:
                                list_of_label_left.append([tensor2int(xyxy), names[c]])
                            else:
                                list_of_label_right.append([list(xyxy), names[c]])

                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                    if send_image_depth_detection:
                        stero = threading.Thread(target=calculate, args=[im0, list_of_label_left, False], name="stero")
                        stero.start()
                pictureDepth["det"] = list_of_label_left

            # print(list_of_label_left)
            # print(list_of_label_right)
            # match init by doing single side
            if view_img and crop:
                match_list = []
                for obj_l in list_of_label_left:
                    cache = []
                    for obj_r in list_of_label_right:
                        if obj_l[1] == obj_r[1]:
                            cache.append(obj_r)
                    min_dis, obj_opt = 266000, []
                    for obj_r_cache in cache:
                        curdis = distance(obj_l, obj_r_cache)
                        if curdis < min_dis:
                            min_dis = curdis
                            obj_opt = obj_r_cache
                    match_list.append([obj_l, obj_opt])

                # for matches that contains more than one
                both_side = True
                if both_side:
                    cache = [[], []]
                    match_list_copy = []
                    for match in match_list:
                        if match[1] not in cache[0]:
                            cache[0].append(match[1])
                        else:
                            cache[1].append(match[1])
                    for match in match_list:
                        if match[1] not in cache[1]:
                            match_list_copy.append(match)
                    check_list = cache[1]
                    for check_index in check_list:
                        obj_opt, dis_min = [], 2660
                        for match in match_list:
                            if match[1] == check_index and len(match[1]) > 0:
                                dis_cache = distance(check_index, match[0])
                                if dis_cache < dis_min:
                                    obj_opt = match[0]
                                    dis_min = dis_cache
                        match_list_copy.append([check_index, obj_opt])
                    match_list = match_list_copy

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            im_blank = np.zeros((720, 1280, 3))
            if view_img and crop:
                for line in match_list:
                    if len(line[1]) > 0:
                        coords = [min(line[0][0][0], line[1][0][0] - 1280), min(line[0][0][1], line[1][0][1]),
                                  max(line[0][0][2], line[1][0][2] - 1280), max(line[0][0][3], line[1][0][3])]
                        best_match_line = mu_ma.match(im0[int(coords[1]):int(coords[3]), int(coords[0]):int(coords[2])],
                                                      im0[int(coords[1]):int(coords[3]),
                                                      int(coords[0]) + 1280:int(coords[2]) + 1280], output=True,
                                                      gray=True)
                        # distance between two point
                        dis = round(((best_match_line[0][0] - best_match_line[1][0]) ** 2 + (
                                best_match_line[0][1] - best_match_line[1][1]) ** 2) ** 0.5, 2)
                        # transform best_match_line coordinates back to original image
                        best_match_line_t = np.int64(
                            best_match_line + np.array([coords[0], coords[1], coords[0] + 1280, coords[1]]).reshape(2,
                                                                                                                    2))
                        # print(best_match_line[0][0], type(best_match_line[0][0]))

                        # convert numpy array to list
                        cv2.line(im0, tuple(best_match_line_t[0]), tuple(best_match_line_t[1]), (0, 255, 0), 20)

                        cv2.line(im0, tensor2int(midpoint(line[0])), tensor2int(midpoint(line[1])), color=(0, 255, 0),
                                 thickness=10)
                        p1 = transverse(tensor2int(midpoint(line[0])))
                        p2 = transverse(tensor2int(midpoint(line[1])))
                        cv2.line(im_blank, p1, p2, color=(0, 255, 0), thickness=10)
                        cv2.putText(im_blank, str(round(distance(line[0], line[1], False).item())),
                                    (max(p1[0], p2[0]), max(p1[1], p2[1])), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255),
                                    10)
                        # print best match line on blank image
                        cv2.line(im_blank, best_match_line[0], best_match_line[1], (0, 0, 255), 20)
                        cv2.putText(im_blank, str(dis), best_match_line[0], cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255),
                                    10)
                        # crop image and pass it
                im_blank = cv2.resize(im_blank, (540, 360))
                cv2.imshow(str(p) + "blank", im_blank)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


# ------------------------------ model ------------------------------

# ------------------------------ api server methods ------------------------------
@api.route("/getResult", methods=["GET"])
def returnResult():
    global pictureDepth
    output = []
    # np.savez("test.npz",depth=pictureDepth["depth"],det=pictureDepth["det"])
    if request.method == "GET":
        map = pictureDepth["depth"]
        det = pictureDepth["det"]
        if len(det): #还没有检测结果
            for item in det:
                iclass = item[1]
                x1, y1, x2, y2 = item[0]
                mean = np.nanmean(np.ma.masked_invalid(map[y1:y2,x1:x2]), axis=(0,1))
                if np.isnan(mean[0]) or np.isnan(mean[1]) or np.isnan(mean[2]):
                    continue
                output.append({"class": iclass, "location": mean.tolist()})
        else:
            return json.dumps(output)
    return json.dumps(output)


@api.route("/", methods=["GET"])
@api.route("/test", methods=["GET"])
def test():
    return {"status": 200}


# ------------------------------ api server methods ------------------------------

# ------------------------------ main ------------------------------
def start():
    apid = threading.Thread(target=lambda: api.run(host="0.0.0.0", port=8880, debug=False, use_reloader=False))
    apid.start()
    run()


# ------------------------------ main ------------------------------

global global_view_img, pictureDepth
if __name__ == "__main__":
    print("Init")
    # server = Process(target=lambda: api.run(host="0.0.0.0", port=8880, debug=False, use_reloader=False))
    pictureDepth = {"depth": [[]], "det": [[]]}
    global_view_img = False
    start()
