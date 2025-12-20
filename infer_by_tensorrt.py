from mmdeploy_runtime import Detector
import numpy as np
import cv2

img = cv2.imread('/home/alan_khang/dev/mmdeploy/demo/resources/det.jpg')

detector = Detector(model_path='/home/alan_khang/dev/mmdeploy/work_dir/mmdet/unipercepnet_s_regnetx_4gf_se_sam_coco', device_name='cuda', device_id=0)
bboxes, labels, masks = detector(img)

indices = [i for i in range(len(bboxes))]
for index, bbox, label_id in zip(indices, bboxes, labels):
    [left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
    if score < 0.3:
        continue

    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))

    if masks[index].size:
        mask = masks[index]
        blue, green, red = cv2.split(img)
        if mask.shape == img.shape[:2]:  # rtmdet-inst
            mask_img = blue
        else:  # maskrcnn
            x0 = int(max(math.floor(bbox[0]) - 1, 0))
            y0 = int(max(math.floor(bbox[1]) - 1, 0))
            mask_img = blue[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
        cv2.bitwise_or(mask, mask_img, mask_img)
        img = cv2.merge([blue, green, red])

cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
