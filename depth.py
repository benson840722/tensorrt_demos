import numpy as np
import cv2
import cv2.cuda
import time

from performance_model import performance_monitor

baseline = 6
focal = 2.6
alpha = 73
bias = 10 # minus

# 定义平移translate函数
def translate(image, x, y):
    # 定义平移矩阵
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
 
    # 返回转换后的图像
    return shifted

def gstreamer_pipeline(
        sensor_id,
        sensor_mode=3,
        capture_width=640,
        capture_height=480,
        display_width=640,
        display_height=480,
        framerate=20,
        flip_method=0,
):
    return (
            "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                sensor_mode,
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
     )

#def update(val = 0):
     
    #blockSize = cv2.getTrackbarPos('blockSize', 'disparity')
 
    #stereo.setP1(8*3*blockSize**2);
    #stereo.setP2(32*3*blockSize**2);
 
    #print ('computing disparity...')

def find_depth():
    l_camera = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0),cv2.CAP_GSTREAMER)
    r_camera = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1),cv2.CAP_GSTREAMER)
         
    #create windows
    #cv2.namedWindow('left_Webcam', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('right_Webcam', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
         
    #设置视频格式大小
    #l_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640);   #设置视频宽度
    #l_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480);  #设置视频长度
         
    #r_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640);   #设置视频宽度
    #r_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480);  #设置视频长度
                 
         
    blockSize = 5
         
    #cv2.createTrackbar('blockSize', 'disparity', blockSize, 60, update)   
    
    create_s = time.time()
    stereo = cv2.StereoSGBM_create(
                     minDisparity=6,
                     numDisparities=54,
                     blockSize=5,
                     uniquenessRatio = 5,
                     speckleWindowSize = 50,
                     speckleRange = 1,
                     disp12MaxDiff = 50,
                     P1 = 8*3*blockSize**2,
                     P2 = 32*3*blockSize**2)
    create_e = time.time()
                 
    while(cv2.waitKey(1) & 0xFF != ord('q')):
        read_s = time.time()
        ret1, left_frame = l_camera.read()
        read_e = time.time()
        ret2, right_frame = r_camera.read()
                  
    #    #对图像进行X,Y轴平移(Img,x,y)
        right_frame = translate(right_frame,0,0)
             
        # our operations on the frame come here
        #left_frame = cv2.flip(left_frame,0)
        #right_frame = cv2.flip(right_frame,0)
        #cv2.imwrite('left.jpg',left_frame)
        #cv2.imwrite('right.jpg',right_frame)
        cvt_s = time.time()
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        cvt_e = time.time()
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        flip_s = time.time()
        gray_left = cv2.flip(gray_left,0)
        flip_e = time.time()
        gray_right = cv2.flip(gray_right,0)
        #cv2.imshow('left_Webcam', gray_left)
        #cv2.imshow('right_Webcam', gray_right)
        
        
        compute_s = time.time()
        disparity = stereo.compute(gray_left, gray_right)
        compute_e = time.time()
        disparity_normal = cv2.normalize(disparity, disparity ,0, 255,cv2.NORM_MINMAX)
        image = np.array(disparity_normal, dtype = np.uint8)

        colormap_s = time.time()
        disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        colormap_e = time.time()

        #disparity = np.hstack((disparity_color, left_frame))
        disparity = cv2.flip(disparity_color,0)
        erode_s = time.time()
        disparity = cv2.erode(disparity, None, iterations=1)
        erode_e = time.time()
        dilate_s = time.time()
        disparity = cv2.dilate(disparity, None, iterations=1)
        dilate_e = time.time()
        disparity = cv2.flip(disparity,0)
            
        ######
            
        height_right, width_right = gray_right.shape
        height_left, width_left = gray_left.shape
            
        if width_right == width_left:
            f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)
        else:
            print('Left and right camera frames do not have the same pixel width')
        
        height, width = disparity.shape[:2]
        center = (int(width/2), int(height/2))
        #disparity_center = disparity[center[0]][center[1]]
        disparity_center = disparity[320,240]
        disp = int(np.mean(disparity_center))
        if disp:
            depth = ((baseline * f_pixel) / disp) - bias
            print(f"Depth: {depth} cm")
        else:
            pass
        
        """
        print("------------------------------------------------------")
        print("cvread: {:.6f} s".format(read_e - read_s))
        print("cvt: {:.6f} s".format(cvt_e - cvt_s))
        print("flip: {:.6f} s".format(flip_e - flip_s))
        print("stereo_create: {:.6f} s".format(create_e - create_s))
        print("stereo_compute: {:.6f} s".format(compute_e - compute_s))
        print("colormap: {:.6f} s".format(colormap_e - colormap_s))
        print("erode: {:.6f} s".format(erode_e - erode_s))
        print("dilate: {:.6f} s".format(dilate_e - dilate_s))
        print("------------------------------------------------------")
        """
        #cv2.imshow('disparity', disparity)
            
        ######
        # When everything done, release the capture
    l_camera.release()
    r_camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    find_depth()
