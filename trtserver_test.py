import os
import argparse
from functools import partial
import cv2
from PIL import Image
import numpy as np
from grpc_client import init_tensorrt_connection,compose_request, process_yolo_response, request_generator


class TestTRTServer():
    def __init__(self, args):
        self.args = args
        self.class_names = [c.strip() for c in open('coco.names').readlines()]
        self.grpc_stub = init_tensorrt_connection(args.tensorrt_grpc)
        print('Initialized TensorRT gRPC connection')

        self.yolo_request = compose_request('input_1', 'yolo_nms_0,yolo_nms_1,yolo_nms_2',
                3, args.yolo_size, args.yolo_size, args.yolo_model, args.yolo_ver)
        self.wh = None

    def bgr_to_rgb(self,frame, yolo_size):
        """
        Convert openCV's BGR format to RGB for YOLOv3
        Parameters
        ---------
            frame : opencv frame in BGR format
            yolo_size : YOLOv3 input shape (416, 416)
        Returns
        -------
            frame_rgb : frame in RGB format
        """
        # cv2 BGR to RGB and resize for yolo
        frame_rgb = Image.fromarray(frame[...,::-1])
        boxed_image = frame_rgb.resize((yolo_size, yolo_size))
        frame_rgb = np.array(boxed_image, dtype='float32')
        frame_rgb = frame_rgb / 255
        return frame_rgb

    def draw_outputs(self, img, outputs):
        boxes, objectness, classes, nums = outputs
        boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
        for i in range(nums):
            x1y1 = tuple((np.array(boxes[i][0:2]) * self.wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * self.wh).astype(np.int32))
            img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
            img = cv2.putText(img, '{} {:.4f}'.format(
                self.class_names[int(classes[i])], objectness[i]),
                x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        return img

    def process(self):
        img = cv2.imread(self.args.img)
        if self.wh is None:
            self.wh = np.flip(img.shape[0:2],0)

        frame_rgb = self.bgr_to_rgb(img, self.args.yolo_size)
        frames = []
        frames.append(frame_rgb)
        responses = []
        yolo_generator = partial(request_generator, self.yolo_request, frames)
        responses = self.grpc_stub.StreamInfer(yolo_generator())
        for response in responses:
            boxes, scores, classes, nums = process_yolo_response(response)
            img_out = self.draw_outputs(img, (boxes, scores, classes, nums))
            cv2.imwrite('img_out.jpg', img_out)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test TensorRT Server and Object detection using YOLOv3')
    # TensorRT Server YOLOv3 configurations
    parser.add_argument('--tensorrt_grpc', type=str, default='localhost:8001', help='TensorRT Server grpc URI')
    parser.add_argument('--yolo_model', type=str, default='yolov3', help='YOLO model name in TensorRT server')
    parser.add_argument('--yolo_ver', type=int , default=1, help='YOLO model version in TensorRT server')
    parser.add_argument('--yolo_size', type=int , default=416, help='width and height for YOLO')

    parser.add_argument('--img', type=str, default='./imgs/meme.jpg', help='Input image for object detection')

    args = parser.parse_args()
    test = TestTRTServer(args)
    test.process()




