import cv2
import time
import math
import numpy as np


class OpticalFlowCalculator:
    '''
    A class for optical flow calculations using OpenCV
    '''
    def __init__(self, perspective_angle=0,
                 move_step=16,
                 window_name=None):
        '''
        Creates an OpticalFlow object for images with specified width and height.
        Optional inputs are:
          perspective_angle - perspective angle of camera, for reporting flow in meters per second 摄影机透视角,用于报告以米每秒为单位的流量
          move_step         - step size in pixels for sampling the flow image 用于采样流图像的步长(以像素为单位)
          window_name       - window name for display 显示窗口名
          flow_color_rgb    - color for displaying flow 流显示颜色
        '''

        self.move_step = move_step
        self.perspective_angle = perspective_angle
        self.window_name = window_name
        self.size = (112, 112)
        self.prev_gray = None
        self.prev_time = None

    def processBytes(self, rgb_bytes, distance=None, timestep=1):
        '''
        Processes one frame of RGB bytes, returning summed X,Y flow.
        Optional inputs are:
          distance - distance in meters to image (focal length) for returning flow in meters per second
          timestep - time step in seconds for returning flow in meters per second
         '''

        frame = np.frombuffer(rgb_bytes, np.uint8)
        frame = np.reshape(frame, (self.size[1], self.size[0], 3))
        return self.processFrame(frame, distance, timestep)

    def processFrame(self, frame, distance=None, timestep=1):
        '''
        Processes one image frame, returning summed X,Y flow and frame.
        Optional inputs are:
          distance - distance in meters to image (focal length) for returning flow in meters per second
          timestep - time step in seconds for returning flow in meters per second
        '''

        frame2 = cv2.resize(frame, self.size)
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        xsum, ysum = 0, 0
        xvel, yvel = 0, 0
        flow = None

        if not self.prev_gray is None:
            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, flow, pyr_scale=0.5, levels=3,
                                                winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            for y in range(0, flow.shape[0], self.move_step):
                for x in range(0, flow.shape[1], self.move_step):
                    fx, fy = flow[y, x]
                    xsum += fx
                    ysum += fy

            # Default to system time if no timestep
            curr_time = time.time()
            if not timestep:
                timestep = (curr_time - self.prev_time) if self.prev_time else 1
            self.prev_time = curr_time
            xvel = self._get_velocity(flow, xsum, flow.shape[1], distance, timestep)
            yvel = self._get_velocity(flow, ysum, flow.shape[0], distance, timestep)
        self.prev_gray = gray
        return xvel, yvel

    def _get_velocity(self, flow, sum_velocity_pixels, dimsize_pixels, distance_meters, timestep_seconds):
        count = (flow.shape[0] * flow.shape[1]) / self.move_step**2
        average_velocity_pixels_per_second = sum_velocity_pixels / count / timestep_seconds

        return self._velocity_meters_per_second(average_velocity_pixels_per_second, dimsize_pixels, distance_meters) \
            if self.perspective_angle and distance_meters \
            else average_velocity_pixels_per_second

    def _velocity_meters_per_second(self, velocity_pixels_per_second, dimsize_pixels, distance_meters):
        distance_pixels = (dimsize_pixels / 2) / math.tan(self.perspective_angle / 2)
        pixels_per_meter = distance_pixels / distance_meters

        return velocity_pixels_per_second / pixels_per_meter

