from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
from mmdet.visualization.local_visualizer import DetLocalVisualizer

import torch
import cv2
import time
import numpy as np

deploy_cfg = '/home/alan_khang/dev/mmdeploy/configs/mmdet/instance-seg/instance-seg_rtmdet-ins_tensorrt_static-640x640.py'
model_cfg = '/home/alan_khang/dev/YOLOF-MaskV2-mmcv/configs/unipercepnet_s_regnetx_4gf_se_sam_3x_coco.py'
device = 'cuda'
backend_model = ['/home/alan_khang/dev/mmdeploy/work_dir/mmdet/unipercepnet_s_regnetx_4gf_se_sam_coco/end2end.engine']
video_path = './car_driving_demo_video.mp4'

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

input_shape = get_input_shape(deploy_cfg)

visualizer = DetLocalVisualizer(alpha=0.5)

coco_classes = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
]

fps = 0
frame_cnt = 0
start_time = time.time()

cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = frame
    show_img = img.copy()

    model_inputs, _ = task_processor.create_input(img, input_shape)
    with torch.no_grad():
        result = model.test_step(model_inputs)

    pred_instances = result[0].pred_instances.cpu()
    pred_instances = pred_instances[pred_instances.scores > 0.5]

    drawed_img = visualizer._draw_instances(
        show_img[..., ::-1],
        pred_instances,
        coco_classes,
        None)

    elapsed_time = time.time() - start_time
    frame_cnt += 1
    if elapsed_time > 1:
        fps = frame_cnt / elapsed_time
        start_time = time.time()
        frame_cnt = 0

    drawed_img = np.ascontiguousarray(drawed_img[..., ::-1])

    show_img = cv2.putText(
        show_img,
        f'FPS: {fps:.2f}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2)

    cv2.imshow('input', show_img)
    cv2.imshow('output', drawed_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
