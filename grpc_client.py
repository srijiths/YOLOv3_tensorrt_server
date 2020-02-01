import grpc
from tensorrtserver.api import api_pb2
from tensorrtserver.api import grpc_service_pb2
from tensorrtserver.api import grpc_service_pb2_grpc
import tensorrtserver.api.model_config_pb2 as model_config
import numpy as np
import tensorflow as tf
from datetime import datetime


def init_tensorrt_connection(url):
    '''
    Iniitlaize connection to TensorRT Server using gRPC
    Arguments :
        url : gRPC TensorRT Server URL (host:port)
    Returns :
        grpc_stub
    '''
    channel = grpc.insecure_channel(url, options=[('grpc.max_receive_message_length', 7000000)])
    grpc_stub = grpc_service_pb2_grpc.GRPCServiceStub(channel)
    return grpc_stub


def compose_request(input_name, output_name, c, w, h, model_name, model_version):
    '''
    Compose gRPC request for different models.
    Arguments :
        input_name  : input placeholder for model
        output_name : output placeholders for model ( one or more)
        c           : channel
        w           : width
        h           : height
        model_name  : model name to connect
        model_version: model version
    Returns:
        gRPC request
    '''
    request = grpc_service_pb2.InferRequest()
    request.model_name = model_name
    request.model_version = model_version
    request.meta_data.batch_size = 1
    request.meta_data.input.add(dims=[w,h,c], name=input_name)

    output_names = output_name.split(',')
    out_len = len(output_names)
    if out_len > 1:
        for i in range(out_len):
            output_message = api_pb2.InferRequestHeader.Output()
            output_message.name = output_names[i]
            request.meta_data.output.extend([output_message])
    else:
        output_message = api_pb2.InferRequestHeader.Output()
        output_message.name = output_name
        request.meta_data.output.extend([output_message])

    return request

def process_yolo_response(response):
    """
    Process YOLO response from TensorRT Server and return boxes, classes, scores and nums.
    Parameters:
    ----------
        response : gRPC response from YOLOv3 model
    Returns:
    -------
        boxes : bounding boxes of each object
        scores : score of each object
        classes : class label for each object
        nums : totoal number of object detections
    """   
    boxes = np.frombuffer(response.raw_output[0], dtype=np.float32)
    boxes = np.reshape(boxes, ((10647, 1, 4)))
    boxes = np.expand_dims(boxes, axis=0)

    scores = np.frombuffer(response.raw_output[1], dtype=np.float32)
    scores = np.reshape(scores, ((10647, 80)))
    scores = np.expand_dims(scores, axis=0)
    start_time = datetime.now()
    boxes, scores, classes, nums = tf.image.combined_non_max_suppression(
            boxes=boxes,
            scores=scores,
            max_output_size_per_class=100,
            max_total_size=200,
            iou_threshold=0.5,
            score_threshold=0.5)

    return boxes, scores, classes, nums

def request_generator(request, images):
    """
    Generic gRPC request raw input generator for TensorRT Server
    Parameters:
    ----------
        request : gRPC request with all meta data and header
        images : list of images
    Returns:
    -------
        request: request with raw input/s appended
    """
    idx = 0
    last_request = False
    while not last_request:
        input_bytes = None
        del request.raw_input[:]
        if len(images) > 0:
            if input_bytes is None:
                input_bytes = images[idx].tobytes()
            else:
                input_bytes += images[idx].tobytes()
            idx = (idx + 1) % len(images)
            request.raw_input.extend([input_bytes])
        if idx == 0:
            last_request = True
        yield request
