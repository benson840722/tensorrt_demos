"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from control import drone_controller
from depth import find_depth

WINDOW_NAME = 'TrtYOLODemo'


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.3,
        help='set the detection confidence threshold')
    
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish|yolov4-p5]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis, writer, img_width, img_height):
    """Continuously capture images from camera and do object detection.
    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    connection_string = '/dev/ttyACM0'
    #drone = drone_controller(connection_string)
    #drone.takeoff(1.25)

    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        """
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        """
        img = cam.read()
        if img is None:
            break
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        img , bb_coor = vis.draw_bboxes(img, boxes, confs, clss)
        
        if (len(boxes) > 0):
            print("control ready!")
            
            # calculate bbox center
            bb_center_w = (bb_coor[2] - bb_coor[0]) / 2 + bb_coor[0]
            bb_center_h = (bb_coor[3] - bb_coor[1]) / 2 + bb_coor[1]
            Area = (bb_coor[2] - bb_coor[0]) * (bb_coor[3] - bb_coor[1])
            print(f"Area: {Area}")
            # calculate img center
            img_center_w = img_width / 2
            img_center_h = img_height / 2
            print(f"bb_center_w: {bb_center_w}")
            print(f"bb_center_h: {bb_center_h}")
            print(f"img_center_w: {img_center_w}")
            print(f"img_center_h: {img_center_h}")
            
            if ((bb_center_h - img_center_h) != 0):
                if (bb_center_h < img_center_h):
                    print("move up!")
                    #drone.move_up()
                    time.sleep(0.5)

                elif (bb_center_h > img_center_h):
                    print("move down!")
                    #drone.move_down()
                    time.sleep(0.5)

            if ((bb_center_w - img_center_w) != 0):
                if (bb_center_w < img_center_w):
                    print("move left!")
                    #drone.move_left()
                    time.sleep(0.5)

                elif (bb_center_w > img_center_w):
                    print("move right")
                    #drone.move_right()
                    time.sleep(0.5)

            if (Area > 5000):
                #drone.move_backward()
                time.sleep(0.5)

            elif (Area < 3000):
                #drone.move_forward()
                time.sleep(0.5)
        else:
            print("No object detect")

        writer.write(img)
        
        """
        img = show_fps(img, fps)
        cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc             
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)
        """

def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)
    
    distance = threading.Thread(target = find_depth)
    distance.start()

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    """
    open_window(
        	WINDOW_NAME, 
        	'Camera TensorRT YOLO Demo',
        	cam.img_width, cam.img_height
        	)
    """ 
    writer = cv2.VideoWriter(
    				#args.output,
    				'result.mp4',
    				cv2.VideoWriter_fourcc(*'mp4v'), 
    				30, 
    				(cam.img_width, cam.img_height)
    			      )
    loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis, writer=writer, img_width=cam.img_width, img_height=cam.img_height)

    writer.release()
    cam.release()
    distance.join()
    print('Done.')
    """
    cv2.destroyAllWindows()
    """

if __name__ == '__main__':
    main()
