import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


@torch.no_grad()
def detect(opt):


    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
    avg_severity_list = []
    fps_list=[]

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    try:
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model


    except:
        print("Cuda not found")
        return
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        # cudnn.benchmark = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    colors = [[225, 198, 33], [15, 89, 16], [138, 71, 164], [205, 15, 16], [11, 92, 180], [179, 117, 127],
              [13, 76, 154], [246, 111, 10]]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    severity_total = 0
    count_mine = 0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                   agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections

        for i, det in enumerate(pred):  # detections per image
            det_cls = []
            severity_list = []
            quality_score = 0
            severity_score = -2
            my_class = 0
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + (
                '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            ori_img_path = str(save_dir / 'labels' / p.name)  # img.jpg
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            im0_copy = im0.copy()

            if len(det):
                # Rescale boxes from img_size to im0 size
                count_mine += 1
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    ###Creating Qulaity Matrix of Image
                    ##Determine detected classes in each image and assign a score for each detection
                    if int(c) not in det_cls:
                        det_cls.append(int(c))
                if any(s in det_cls for s in (0, 1, 2, 3, 4)):
                    quality_score += 45
                    if 0 in det_cls:
                        severity_list.append(4)
                    if all(p in det_cls for p in (1, 5)):
                        severity_list.append(0)
                    if all(p in det_cls for p in (2, 5)):
                        severity_list.append(1)
                    if all(p in det_cls for p in (3, 5)):
                        severity_list.append(2)
                    if 4 in det_cls:
                        severity_list.append(3)
                    if any(q in det_cls for q in (1, 2, 3)) and 5 not in det_cls:
                        if 0 not in det_cls and 4 not in det_cls:
                            quality_score = 0
                            severity_list.append(-1)
                if 5 in det_cls:
                    quality_score += 30
                    severity_list.append(-1)
                if 6 in det_cls:
                    quality_score += 15
                    severity_list.append(-1)
                if 7 in det_cls:
                    quality_score += 10
                    severity_list.append(-1)

                if (quality_score >= 90):
                    image_quality = 'Excellent'
                elif (quality_score >= 75 and quality_score < 90):
                    image_quality = 'Good'
                elif (quality_score >= 45 and quality_score < 70):
                    image_quality = 'Average'
                elif (quality_score >= 30 and quality_score < 45):
                    image_quality = 'Below Average'
                else:
                    image_quality = 'Bad'
                
                if dataset.mode == 'image':
                    if(quality_score>70):
                        cv2.imwrite(ori_img_path, im0)
                else:  # 'video'
                    ori_vid_path = txt_path + '.jpg'
                    if(quality_score>70):
                        cv2.imwrite(ori_vid_path, im0)
                
                # if dataset.mode == 'image':
                #     if(quality_score>=45 & max(severity_list)>=3):
                #         cv2.imwrite(ori_img_path, im0)
                # else:  # 'video'
                #     ori_vid_path = txt_path + '.jpg'
                #     if(quality_score>=45 & max(severity_list)>=1):
                #         cv2.imwrite(ori_vid_path, im0)

                # # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, conf, *xywh) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        # print(xywh[0])
                        # print(f'{names(int[cls]):.2f}')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

                try:
                    severity_score = max(severity_list)
                except:
                    severity_score = -1
                    print("enterd here")

                ##Write Image Quality and Quality score
                if save_txt:
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(image_quality + ', severity score: ' + '%g' % severity_score + '\n')
                        f.write('Image quality score: ' + '%g' % quality_score + '\n')
                        if dataset.mode == 'image':
                            f.write('\n')
            print(f'{s}AllDone. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)

            print("Severity Score=",severity_score)
            if save_img:
                if dataset.mode == 'image':
                    if(quality_score>45):
                        cv2.imwrite(save_path, im0)                        


                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                            # avg_severity = severity_total / nfrms
                            # fps_list.append(nfrms)
                            # avg_severity=severity_total/nfrms
                            if (count_mine != 0):
                                avg_severity = severity_total / (count_mine - 1)
                            else:
                                avg_severity = 0
                            fps_list.append(count_mine - 1)
                            avg_severity_list.append(avg_severity)
                            severity_total = 0
                            count_mine = 1

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        my_nfrms = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

                    try:
                        if (severity_score < 0):
                            severity_score = 0
                        severity_total = severity_total + severity_score
                    except:
                        print("An ërror occured while calculating severity total")
                    if(quality_score>45):
                        vid_writer.write(im0)
    if dataset.mode == 'video':
        # avg_severity=severity_total/nfrms
        if (count_mine != 0):
            avg_severity = severity_total / count_mine
        else:
            avg_severity = 0
        fps_list.append(count_mine)
        avg_severity_list.append(avg_severity)
        # fps_list.append(nfrms)
        if save_txt:
            txt_files = list(save_dir.glob('labels/*.txt'))
            num = 0
            j = 0
            for file in txt_files:
                num += 1
                if (num <= fps_list[j]):
                    with open(file, 'a') as f:
                        f.write('Average severity score: ' + '%g' % avg_severity_list[j] + '\n')
                else:
                    num = 1
                    j += 1
                    with open(file, 'a') as f:
                        f.write('Average severity score: ' + '%g' % avg_severity_list[j] + '\n')
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     print(f"Results saved to {save_dir}{s}")
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=r'data', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_false', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_false', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', type=bool, default=False, help='use FP16 half-precision inference')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    if opt.update:  # update all models (to fix SourceChangeWarning)
        for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            detect(opt=opt)
            strip_optimizer(opt.weights)
    else:
        detect(opt=opt)
