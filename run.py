import os.path
import time

from PIL import Image
from flask import Flask, request
import torch
import argparse
import os
import platform
import sys
from pathlib import Path
from models.common import DetectMultiBackend
from models.experimental import attempt_load
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

app = Flask(__name__)


def detect(
        weights='runs/train/300/best.pt',
        source='',
        project='runs/detect',
        imgsz=(640, 640),
        device='0',
        conf_thres=0.4,
        iou_thres=0.45,
        max_det=1000
):
    source = str(source)
    save_dir = Path(project)
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data='data/coco128.yaml', fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    bs = 1
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    labels = []
    visualize = False

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    windows, dt = [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=False, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, None, True, max_det=max_det)
        for i, det in enumerate(pred):
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())) + '.jpg'  # im.jpg
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy()
            annotator = Annotator(im0, line_width=3, example=str(names))
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                labels.append(names[int(c)])
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = None if False else (names[c] if False else f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
            # Write results
            # for *xyxy, conf, cls in reversed(det):
            #     if save_txt:  # Write to file
            #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            #         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            #         with open(f'{txt_path}.txt', 'a') as f:
            #             f.write(('%g ' * len(line)).rstrip() % line + '\n')
            #
            #     if save_img or save_crop or view_img:  # Add bbox to image
            #         c = int(cls)  # integer class
            #         label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
            #         annotator.box_label(xyxy, label, color=colors(c, True))
            #     if save_crop:
            #         save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

        # Stream results
            im0 = annotator.result()
            cv2.imwrite(save_path, im0)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    return labels


@app.route('/image_post', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        global filename
        filename = f.filename
        # global file_path

        file_path = 'runs/imagepost/' + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + '.jpg'

        f.save(file_path)
        labels = detect(source=file_path)
        return labels


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8088)
