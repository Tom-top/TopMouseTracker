import os
import time
import threading
import queue

import pyautogui
import psutil
import ctypes
import numpy as np
import cv2
import matplotlib.pyplot as plt

from pykinect2 import PyKinectV2, PyKinectRuntime


class KinectException(Exception):
    pass


class Kinect:
    def __init__(self, experiment, animal, verbose=False):
        self._experiment = experiment
        self._animal = animal
        self._verbose = verbose
        self._monitor_size = pyautogui.size()

        self.initialize_kinect()
        self._color_width, self._color_height = self.get_color_dimensions()
        self._area_color = self._color_width * self._color_height
        self._color_space_point = PyKinectV2._ColorSpacePoint
        self._resizing_factor_color = self.divide_until_smaller(self._color_width, self._monitor_size.width/4)
        self._window_width_color = int(self._color_width/self._resizing_factor_color)
        self._window_height_color = int(self._color_height/self._resizing_factor_color)

        self._depth_width, self._depth_height = self.get_depth_dimensions()
        self._area_depth = self._depth_width * self._depth_height
        self._depth_space_point = PyKinectV2._DepthSpacePoint
        self._depth_x, self._depth_y = self.determine_registration_parameters()
        self._resizing_factor_depth = self.divide_until_smaller(self._depth_width, self._monitor_size.width/4)
        self._window_width_depth = int(self._depth_width / self._resizing_factor_depth)
        self._window_height_depth = int(self._depth_height / self._resizing_factor_depth)

        self._roi_reference = []
        self.range_for_conversion_is_set = False

    def initialize_kinect(self):
        self._kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Depth |
            PyKinectV2.FrameSourceTypes_Color)
        time.sleep(2)  # This is not a mistake, if this is disabled the kinect
        # Fails to load the first frame.

    def turn_off(self):
        self._kinect.close()

    def get_raw_color_frame(self):
        if self._kinect.has_new_color_frame():
            color_frame = self._kinect.get_last_color_frame()
            return color_frame
        else:
            if self._verbose:
                print("[ERROR] no new color frame available!")
            return None

    def get_raw_depth_frame(self):
        if self._kinect.has_new_depth_frame():
            depth_frame = self._kinect.get_last_depth_frame()
            return depth_frame
        else:
            if self._verbose:
                print("[ERROR] no new depth frame available!")
            return None

    def get_color_dimensions(self):
        width = self._kinect.color_frame_desc.Width
        height = self._kinect.color_frame_desc.Height
        return width, height

    def get_depth_dimensions(self):
        width = self._kinect.depth_frame_desc.Width
        height = self._kinect.depth_frame_desc.Height
        return width, height

    def get_color_frame(self, flip=True):
        color_frame = self.get_raw_color_frame()
        if color_frame is None:
            return color_frame
        else:
            color_frame = self.reshape_frame(color_frame,
                                             self._color_width,
                                             self._color_height,
                                             depth=-1)
            if flip:
                color_frame = self.flip_frame_horizontally(color_frame)
            return color_frame

    def get_depth_frame(self, flip=True, return_raw=False):
        raw_depth_frame = self.get_raw_depth_frame()
        if raw_depth_frame is None:
            return raw_depth_frame
        else:
            raw_depth_frame = self.reshape_frame(raw_depth_frame,
                                                 self._depth_width,
                                                 self._depth_height,
                                                 depth=None)
            if flip:
                raw_depth_frame = self.flip_frame_horizontally(raw_depth_frame)
            if return_raw:
                return raw_depth_frame
            else:
                depth_frame = self.frame_to_8bit(raw_depth_frame)
                return depth_frame

    def determine_registration_parameters(self):
        """Inspired by https://github.com/KonstantinosAng/
        PyKinect2-Mapper-Functions/blob/master/mapper.py
        """
        color_to_depth_points_type = self._depth_space_point * self._area_color
        color_to_depth_points = ctypes.cast(color_to_depth_points_type(),
                                            ctypes.POINTER(self._depth_space_point))
        self._kinect._mapper. \
            MapColorFrameToDepthSpace(ctypes.c_uint(self._area_depth),
                                      self._kinect._depth_frame_data,
                                      ctypes.c_uint(self._area_color),
                                      color_to_depth_points)
        depth_x_y = np.copy(np.ctypeslib.as_array(color_to_depth_points,
                                                  shape=(self._area_color,)))
        depth_x_y = depth_x_y.view(np.float32).reshape(depth_x_y.shape + (-1,))
        depth_x_y += 0.5
        depth_x_y = depth_x_y.reshape(self._color_height,
                                      self._color_width,
                                      2).astype(int)
        depth_x = np.clip(depth_x_y[:, :, 0], 0, self._depth_width - 1)
        depth_y = np.clip(depth_x_y[:, :, 1], 0, self._depth_height - 1)

        return depth_x, depth_y

    def register_depth_to_rgb(self, flip=True):

        raw_depth_frame = self.get_depth_frame(flip=False, return_raw=True)
        registered_depth_img = np.zeros((self._color_height, self._color_width),
                                        dtype=np.uint16)
        self._depth_x, self._depth_y = self.determine_registration_parameters()
        registered_depth_img[:, :] = raw_depth_frame[self._depth_y, self._depth_x]
        if flip:
            registered_depth_img = self.flip_frame_horizontally(registered_depth_img)

        return registered_depth_img

    def set_cropping_params(self, upper_left, lower_right):
        self._roi_reference = [upper_left, lower_right]
        self._roi_width = abs(self._roi_reference[0][0] - self._roi_reference[1][0])
        self._roi_height = abs(self._roi_reference[0][1] - self._roi_reference[1][1])

    def set_roi(self):
        self._roi_reference = []
        self.rgb_frame_roi = self.get_color_frame()
        self.rgb_clone_roi = self.rgb_frame_roi.copy()

        cv2.namedWindow("rgb_frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("rgb_frame", (self._color_width, self._color_height))
        # cv2.resizeWindow("rgb_frame", (self._window_width_color, self._window_height_color))
        cv2.setMouseCallback("rgb_frame", self.select_and_crop)

        while True:
            cv2.imshow("rgb_frame", self.rgb_frame_roi)
            key = cv2.waitKey(10) & 0xFF

            if key == ord("r"):
                if self._verbose:
                    print("[INFO] ROI was reset")
                self._roi_reference = []
                self.rgb_frame_roi = self.rgb_clone_roi.copy()
            elif key == ord("c"):
                if self._verbose:
                    print("[INFO] ROI successfully set")
                break

        cv2.destroyAllWindows()

        for _ in range(1, 5):
            cv2.waitKey(1)

    def select_and_crop(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._roi_reference = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            self._roi_reference.append((x, y))

        if len(self._roi_reference) == 2 and self._roi_reference is not None:
            self._roi_width = abs(self._roi_reference[0][0] - self._roi_reference[1][0])
            self._roi_height = abs(self._roi_reference[0][1] - self._roi_reference[1][1])
            cv2.rectangle(self.rgb_frame_roi, self._roi_reference[0], self._roi_reference[1], (0, 0, 255), 2)
            cv2.imshow("rgb_frame", self.rgb_frame_roi)

    def crop_frame_from_params(self, frame):
        if self._roi_reference:
            return frame[self._roi_reference[0][1]:self._roi_reference[1][1],
                         self._roi_reference[0][0]:self._roi_reference[1][0]]
        else:
            return frame

    def set_16bit_data_range(self, vmin=0, vmax=256):
        self.range_for_conversion_is_set = True
        if vmin < 0 and vmax < 0:
            raise KinectException("vmin and vmax can't both be negative")
        if vmin > vmax:
            raise KinectException("vmin has to be superior to vmax!")
        if vmin == -1:
            vmin = vmax - 256
        if vmax == -1:
            vmax = vmin + 256
        if vmax - vmin < 256:
            print("[WARNING] vmax-vmin is superior to 256 clipping the range automatically")
            vmin = vmax - 256
        self._depth_vmin = vmin
        self._depth_vmax = vmax

    def check_depth_histogram(self, zoom_raw=(), bins=300):
        depth_frame = self.register_depth_to_rgb()
        cropped_depth_frame = self.crop_frame_from_params(depth_frame)
        scaled_depth_frame = self.scale_image_to_8bit_from_data_range(cropped_depth_frame,
                                                                      self._depth_vmin,
                                                                      self._depth_vmax)

        plt.figure(figsize=(4, 7))
        y_max = 10000

        ax0 = plt.subplot(4, 1, 1)
        if zoom_raw:
            ax0.imshow(cropped_depth_frame, vmin=zoom_raw[0], vmax=zoom_raw[-1], aspect='auto')
        else:
            ax0.imshow(cropped_depth_frame, vmin=0, vmax=2**16, aspect='auto')
        ax0.set_xticks([])
        ax0.set_yticks([])

        ax1 = plt.subplot(4, 1, 2)
        ax1.hist(cropped_depth_frame.flatten(), bins=bins)
        ax1.vlines([self._depth_vmin, self._depth_vmax],
                   ymin=0,
                   ymax=y_max,
                   color="red",
                   linewidth=0.5)
        if zoom_raw:
            ax1.set_xlim(zoom_raw[0], zoom_raw[1])
        else:
            ax1.set_xlim(0, 2**16)
        ax1.set_ylim(0, y_max)

        ax2 = plt.subplot(4, 1, 3)
        ax2.imshow(scaled_depth_frame, vmin=0, vmax=2**8, aspect='auto')
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3 = plt.subplot(4, 1, 4)
        ax3.hist(scaled_depth_frame.flatten(), bins=bins)
        ax3.set_xlim(0, 2**8)
        ax3.set_ylim(0, y_max)

        plt.tight_layout()
        plt.show()

    def check_camera_feed(self):
        cv2.namedWindow("color", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("color", (self._window_width_color, self._window_height_color))
        cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("depth", (self._window_width_color, self._window_height_color))
        while True:
            color_frame = self.get_color_frame(flip=True)
            color_frame = self.crop_frame_from_params(color_frame)
            color_frame = color_frame[:, :, :3]

            depth_frame = self.register_depth_to_rgb(flip=True)
            depth_frame = self.crop_frame_from_params(depth_frame)
            if self.range_for_conversion_is_set:
                depth_frame = self.scale_image_to_8bit_from_data_range(depth_frame,
                                                                       self._depth_vmin,
                                                                       self._depth_vmax)
            depth_frame = depth_frame.astype("uint8")

            cv2.imshow("color", color_frame)
            cv2.imshow("depth", depth_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def set_video_writers(self, fps, saving_directory, color=True, depth=True):
        if color:
            color_video_path = os.path.join(saving_directory,
                                            "color_video_{}_{}.mp4"
                                            .format(self._experiment,
                                                    self._animal))
            if self._roi_reference:
                color_width, color_height = self._roi_width, self._roi_height
            else:
                color_width, color_height = self._color_width, self._color_height
            self.video_writer_color = cv2.VideoWriter(color_video_path,
                                                      cv2.VideoWriter_fourcc(*"DIVX"),
                                                      fps,
                                                      (color_width,
                                                       color_height))
        if depth:
            depth_video_path = os.path.join(saving_directory,
                                            "depth_video_{}_{}.mp4"
                                            .format(self._experiment,
                                                    self._animal))
            if self._roi_reference:
                depth_width, depth_height = self._roi_width, self._roi_height
            else:
                depth_width, depth_height = self._depth_width, self._depth_height
            self.video_writer_depth = cv2.VideoWriter(depth_video_path,
                                                      cv2.VideoWriter_fourcc(*"DIVX"),
                                                      fps,
                                                      (depth_width,
                                                       depth_height),
                                                      False)

    def capture_camera_feed(self, fps, saving_directory,
                            recording_duration, color=True,
                            depth=True):
        if not any([color, depth]):
            raise ValueError("Both color and depth were set to False!\n\
                             at least one camera feed has to be enabled")
        if not os.path.exists(saving_directory):
            os.mkdir(saving_directory)
        self.set_video_writers(fps, saving_directory, color=color, depth=depth)
        queue_video_writer_color = VideoWriter(self.video_writer_color,
                                               saving_dir=saving_directory,
                                               fn="color",
                                               fps=fps,
                                               duration=recording_duration)
        queue_video_writer_depth = VideoWriter(self.video_writer_depth,
                                               saving_dir=saving_directory,
                                               fn="depth",
                                               fps=fps,
                                               duration=recording_duration)
        delay_between_frames = 1 / fps
        start_timer = time.time()
        fps_timer = time.time()

        while True:
            cur_timer = time.time()
            if (cur_timer - fps_timer) >= delay_between_frames:
                if self._verbose:
                    percentage_error = (((cur_timer - fps_timer) -
                                         delay_between_frames) /
                                        delay_between_frames) * 100
                    print(psutil.virtual_memory().percent)
                    print(percentage_error)
                if color:
                    color_frame = self.get_color_frame(flip=True)
                    color_frame = self.crop_frame_from_params(color_frame)
                    color_frame = color_frame[:, :, :3]
                    queue_video_writer_color.write(color_frame)
                if depth:
                    depth_frame = self.register_depth_to_rgb(flip=True)
                    depth_frame = self.crop_frame_from_params(depth_frame)
                    if self.range_for_conversion_is_set:
                        depth_frame = self.scale_image_to_8bit_from_data_range(depth_frame,
                                                                               self._depth_vmin,
                                                                               self._depth_vmax)
                    depth_frame = depth_frame.astype("uint8")
                    queue_video_writer_depth.write(depth_frame)
                fps_timer = cur_timer
            if (cur_timer - start_timer) > recording_duration:
                break

    @staticmethod
    def frame_to_8bit(frame):
        return frame.astype(np.uint8)

    @staticmethod
    def resize_frame(frame, resizing_factor):
        pass

    @staticmethod
    def reshape_frame(frame, width, height, depth=None):
        if depth is not None:
            return frame.reshape(height, width, depth)
        else:
            return frame.reshape(height, width)

    @staticmethod
    def flip_frame_horizontally(frame):
        return cv2.flip(frame, 1)

    @staticmethod
    def flip_frame_vertically(frame):
        return cv2.flip(frame, 0)

    @staticmethod
    def frame_bgr_to_rgb(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def frame_rgb_to_bgr(frame):
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    @staticmethod
    def blend_frames(frame1, frame2):
        return cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0.0)

    @staticmethod
    def show_frame(frame, range_im_1=(None, None)):
        plt.figure(figsize=(5, 5))
        ax0 = plt.subplot(1, 1, 1)
        ax0.imshow(frame, vmin=range_im_1[0], vmax=range_im_1[1])

    @staticmethod
    def show_two_frames(frame1, frame2, range_im_1=(None, None), range_im_2=(None, None)):
        plt.figure(figsize=(15, 5))
        ax0 = plt.subplot(1, 2, 1)
        ax0.imshow(frame1, vmin=range_im_1[0], vmax=range_im_1[1])
        ax1 = plt.subplot(1, 2, 2)
        ax1.imshow(frame2, vmin=range_im_2[0], vmax=range_im_2[1])

    @staticmethod
    def show_histogram_pixel_intensity(frame, bins=200):
        if len(frame.shape) > 1:
            frame = frame.flatten()
        plt.figure(figsize=(5, 5))
        ax0 = plt.subplot(1, 1, 1)
        ax0.hist(frame, bins=bins)

    @staticmethod
    def threshold_image(frame, low, high):
        frame[frame <= low] = low
        frame[frame >= high] = high
        return frame

    @staticmethod
    def scale_image_to_8bit(frame):
        return ((frame - min(frame)) / (max(frame) - min(frame))) * 256

    @staticmethod
    def scale_image_to_8bit_from_data_range(frame, vmin, vmax):
        frame_copy = frame.copy()
        frame_copy[frame_copy < vmin] = vmin
        frame_copy[frame_copy > vmax] = vmax
        scaled_frame = ((frame_copy - vmin) / (vmax - vmin)) * 256
        return scaled_frame

    @staticmethod
    def crop_frame(frame, upper_left, lower_right):
        y0, x0 = upper_left
        y1, x1 = lower_right
        return frame[x0:x1, y0:y1]

    @staticmethod
    def divide_until_smaller(source, target):
        res = 1
        while source > target:
            source = source/2
            res *= 2
        return res


class VideoWriter:
    def __init__(self, video_writer, fn=0, fcc='XVID', duration=1 * 60, fps=15, saving_dir=os.path.expanduser("~")):
        self._fourcc = fcc
        self._video_duration = duration
        self._video_fps = fps
        self._dt = 1. / self._video_fps
        self._video_writer = video_writer
        self._queue = queue.Queue()
        self._stop = False
        self._n = 0
        self._saving_dir = saving_dir
        self._fn = fn
        self._video_path = os.path.join(self._saving_dir, "{}.avi".format(self._fn))
        self._wrtr = threading.Thread(target=self.queue_writer)
        self._wrtr.start()

    def queue_writer(self):
        current_frame = None
        start_time = None
        while True:
            if self._stop:
                break
            while not self._queue.empty():
                current_frame = self._queue.get_nowait()
            if current_frame is not None:
                if start_time is None:
                    start_time = time.time()
                self._video_writer.write(current_frame)
                self._n += 1
                if self._n == self._video_duration * self._video_fps:
                    self.stop()
                    self.release()
            dt = self._dt if start_time is None else \
                max(0, start_time + self._n * self._dt - time.time())
            time.sleep(dt)

    def write(self, frm):
        self._queue.put(frm)

    def release(self):
        self._video_writer.release()

    def stop(self):
        self._stop = True