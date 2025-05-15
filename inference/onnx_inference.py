import cv2
import numpy as np
import onnxruntime as ort

from typing import Optional


class Model:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is not None:
            self.load_model(model_path)
        
    def load_model(self, model_path):
        self.model = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.input_name = [x.name for x in self.model.get_inputs()]
        _, _, h, w = self.model.get_inputs()[0].shape
        self.model_inpsize = (w, h)
        
    def run(self, image, det_thresh=0.25):
        input_tensor, ratio, dwdh = self.preprocessing(image)
        predicts = self.model.run(None, dict(zip(self.input_name,[input_tensor])))[0]
        outputs = self.postprocessing(predicts, image, det_thresh, dwdh, ratio)
        
        return outputs
        
    def preprocessing(self, im):
        shape = im.shape[:2]  # get current shape
        new_shape = (640, 640)  # default shape

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw, dh = dw / 2, dh / 2  # divide padding into 2 sides
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, 0)
        im = np.ascontiguousarray(im, dtype=np.float32)
        im /= 255

        return im, r, (dw, dh)
    
    def postprocessing(self, predicts, image, det_thres, dwdh, ratio):
        if isinstance(predicts, list):
            predicts = np.array(predicts)

        predicts = predicts[predicts[:, 6] > det_thres] # get sample higher than threshold

        padding = dwdh*2
        bboxes, score, cls  = predicts[:,1:5], predicts[:, 6], predicts[:, 5]
        bboxes = ((bboxes- np.array(padding)) / ratio).round()

        self.clip_coords(bboxes, image.shape)

        result = []
        for _box, _score, _cls in zip(bboxes, score, cls):
            result.append(np.concatenate((_box, _score, _cls), axis=None))

        return result
    
    def clip_coords(self, boxes, img_shape):
          # Clip bounding xyxy bounding boxes to image shape (height, width)
          boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1
          boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1
          boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2
          boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2
    

def draw(image, predictions, thickness=2, font_scale=0.5, font_color=(0, 255, 0), font_thickness=1):
    for line in predictions:
        x_min, y_min, x_max, y_max, score, cls_id = line[:6]
        score = round(score, 2)

        x_min, y_min, x_max, y_max, cls_id = map(int, [x_min, y_min, x_max, y_max, cls_id])
        label = f"{score:.2f}"
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=font_color, thickness=thickness)
        cv2.putText(image, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

    return image

    

if __name__ == '__main__':
    model = Model('weights/v9-c-end2end.onnx')
    image = cv2.imread("demo/images/inference/image.png")
    
    predicts = model.run(image)
    
    draw(image, predicts)
    
    cv2.imwrite("demo/images/output/image.png", image)
    
    print("Predict: ", predicts)
