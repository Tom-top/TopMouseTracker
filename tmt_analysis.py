#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:06:08 2019

@author: tomtop
"""

import os
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import clips_array
from scipy.signal import savgol_filter

import tmt_tracker as tracker


class Plot(tracker.TopMouseTracker):

    def __init__(self, color_capture, depth_capture, time_limit, **kwargs):

        tracker.TopMouseTracker.__init__(self, color_capture, depth_capture, **kwargs)

        self.nesting_raster = None
        self.time_resolution = self._framerate_color * self._args["plot"]["general"]["resolution"]

        self._positions = np.load(os.path.join(self._args["general"]["segmentation_directory"],
                                               "data_{0}_positions.npy"
                                               .format(self._args["general"]["general"]["animal_tag"])))
        self._roi_ref_points = np.load(os.path.join(self._args["general"]["segmentation_directory"],
                                                    "data_{0}_ref_points.npy"
                                                    .format(self._args["general"]["general"]["animal_tag"])))
        self._nm_average_height = np.load(os.path.join(self._args["general"]["segmentation_directory"],
                                                       "data_{0}_nesting_material_average_height.npy"
                                                       .format(self._args["general"]["general"][
                                                                   "animal_tag"])))
        self._segmentation_video = os.path.join(self._args["general"]["segmentation_directory"],
                                                "tracking_{0}.{1}".format(
                                                    self._args["general"]["general"]["animal_tag"],
                                                    self._args["general"]["general"][
                                                        "extension_video"]))
        if time_limit is not None:
            self._time_limit = time_limit
        else:
            self._time_limit = self._recording_duration

        self.up_left_x = int(self._roi_ref_points[0][0])  # Defines the Up Left ROI corner X coordinates
        self.up_left_y = int(self._roi_ref_points[0][1])  # Defines the Up Left ROI corner Y coordinates
        self.low_right_x = int(self._roi_ref_points[1][0])  # Defines the Low Right ROI corner X coordinates
        self.low_right_y = int(self._roi_ref_points[1][1])  # Defines the Low Right ROI corner Y coordinates

        self.distance_conversion_factor = (abs(self.up_left_x - self.low_right_x) / self._args["general"]["cage"][
            "length"] +
                                           abs(self.up_left_y - self.low_right_y) / self._args["general"]["cage"][
                                               "width"]) / 2  # Defines the resizing factor for the cage

        self.roi_width = abs(self.low_right_x - self.up_left_x)
        self.roi_length = abs(self.low_right_y - self.up_left_y)

        self.distance_px = [sqrt((self._positions[n + 1][0] - self._positions[n][0]) ** 2 +
                                 (self._positions[n + 1][1] - self._positions[n][1]) ** 2) for n in
                            range(len(self._positions)) if n + 1 < len(self._positions)]
        self.distance_m = [dist / self.distance_conversion_factor for dist in self.distance_px]
        self.distance_corrected = [dist if self._args["plot"]["general"]["minimal_distance"] < dist <
                                           self._args["plot"]["general"]["maximal_distance"] else 0 for dist
                                   in self.distance_m]
        self.distance_cropped = np.array(self.distance_corrected)[0:self._time_limit * int(self._framerate_color)]
        self.distance = np.sum(self.distance_cropped)
        self.distance_cumulative = np.cumsum(self.distance_cropped)
        self.nm_average_height_cropped = np.array(self._nm_average_height)[
                                         0:int(self._time_limit * self._framerate_color)]
        self.nm_average_height_smooth = self.smooth(self.nm_average_height_cropped,
                                                    win=5 * int(self._framerate_color),
                                                    order=3)[0:self._time_limit * int(self._framerate_color)]
        self.positions_cropped = np.array(self._positions)[
                                 0:int(self._time_limit * self._framerate_color)]

    def detect_nesting_events(self):
        nesting = [np.mean(self.nm_average_height_cropped[int(i):int(i + self.time_resolution)]) for i in
                   np.arange(0, len(self.nm_average_height_cropped), self.time_resolution)]

        detected_events = []
        for i, data in enumerate(nesting):
            if i <= len(nesting) - self._args["plot"]["general"]["peak_distance"] - 1:
                if abs(data - nesting[i + self._args["plot"]["general"]["peak_distance"]]) \
                        >= self._args["plot"]["general"]["peak_threshold"]:
                    detected_events.append(i)

        event_start = []
        event_end = []
        if self._args["plot"]["general"]["event_distance"] is not None:
            for pos, event in enumerate(detected_events):
                if not event_start:
                    event_start.append(detected_events[pos])
                if pos < len(detected_events) - 1:
                    if detected_events[pos + 1] - detected_events[pos] \
                            >= self._args["plot"]["general"]["event_distance"]:
                        if event_start[-1] == detected_events[pos]:
                            event_end.append(detected_events[pos] + self._duration_between_two_frames)
                        else:
                            event_end.append(detected_events[pos])
                        event_start.append(detected_events[pos + 1])
                    else:
                        pass
                else:
                    if event_start[-1] == detected_events[pos]:
                        event_end.append(detected_events[pos] + self._duration_between_two_frames)
                    else:
                        event_end.append(detected_events[pos])

        self.nesting_events = np.column_stack([event_start, event_end])

    def generate_raster_file(self):
        sink = np.zeros(int(self._recording_duration * self._framerate_color))
        for s, e in self.nesting_events:
            s = int(s * self._framerate_color)
            e = int(e * self._framerate_color)
            sink[s:e] = 1
        np.save(os.path.join(self._args["general"]["segmentation_directory"],
                             "data_{0}_raster.npy".format(self._args["general"]["general"]["animal_tag"])), sink)

    def load_raster_file(self):
        raster_file_path = os.path.join(self._args["general"]["segmentation_directory"],
                                        "data_{0}_raster.npy".format(self._args["general"]["general"]["animal_tag"]))
        if os.path.exists(raster_file_path):
            self.nesting_raster = np.load(raster_file_path)
        else:
            raise RuntimeError("Please generate raster file first: generate_raster_file")

    def create_tracking_animal(self, ax, lg=True):
        lg4, = self.plot_tracking = ax.plot([x[0] for x in self.positions_cropped],
                                            [y[1] for y in self.positions_cropped],
                                            '-o',
                                            markersize=0.5,
                                            alpha=0.1,
                                            solid_capstyle="butt",
                                            color="red",
                                            )
        lg4_b, = self.plot_tracking = ax.plot(self.positions_cropped[-1][0],
                                              self.positions_cropped[-1][1],
                                              'o',
                                              markersize=5,
                                              alpha=1,
                                              solid_capstyle="butt",
                                              color="red",
                                              )
        if lg:
            self.lg.append(lg4)
            self.lg.append(lg4_b)
        # self.segmentation_video = mpy.VideoFileClip(self._segmentation_video)
        # self.segmentation_video_resized = self.segmentation_video.resize(0.25)
        # frame_to_display = cv2.resize(self.segmentation_video.get_frame(-1),
        #                               dsize=(int(self.roi_width), int(self.roi_length)),
        #                               interpolation=cv2.INTER_CUBIC)
        # ax.imshow(self.segmentation_video_resized.get_frame(-1))
        # str_distance = "%.2f" % (len(self.distance_cropped) / 100)
        # h, m, s = utils.get_h_m_s(len(self.positions_cropped) / self._framerate_color)
        # font = {"size": 5,
        #         "color": "black",
        #         "alpha": 0.5,
        #         }
        # ax.text(int(self.roi_width) / 3, -27, "Time: {0}h {1}m {2}s".format(str(h), str(m), str(s)),
        #         fontdict=font)
        # ax.text(int(self.roi_width) / 3, -19, "Dist: {0}m".format(str_distance),
        #         fontdict=font)

        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
        ax.set_title("Tracking Animal {0}".format(self._animal_tag), fontsize=7)

        ax.set_xlim([0, int(self.roi_width)])
        ax.set_ylim([0, int(self.roi_length)])
        plt.gca().invert_yaxis()

    def plot_animal_tracking(self):
        fig = plt.figure(figsize=(7, 5), dpi=300)
        ax0 = plt.subplot()
        self.create_tracking_animal(ax0, lg=True)
        plt.tight_layout()

        if self._args["plot"]["general"]["save"]:
            plt.savefig(os.path.join(self._args["general"]["segmentation_directory"],
                                     "tracking_animal_{0}".format(self._animal_tag)), dpi=300)

        plt.show()

    def plot_complete_tracking(self, track_length=20):
        self.track_length = track_length
        self.fig = plt.figure(figsize=(7, 5), dpi=300)
        self.lg = []
        ax0 = plt.subplot2grid((4, 4), (0, 3))
        lg0, = ax0.plot(np.arange(0, len(self.distance_cropped)),
                        self.distance_cropped,
                        linewidth=0.5,
                        color="black",
                        )
        self.lg.append(lg0)
        ax0.tick_params(axis='both', labelsize=5)
        ax0.set_xlim([0, self._time_limit * self._framerate_color])
        ax0.set_title("Speed over time (cm/s)", fontsize=5)
        ax0.set_ylabel("Speed (cm/s)", fontsize=5)
        ax0.tick_params(bottom=False, labelbottom=False)

        ax1 = plt.subplot2grid((4, 4), (1, 3), sharex=ax0)
        lg1, = ax1.plot(np.arange(0, len(self.distance_cumulative)),
                        self.distance_cumulative,
                        linewidth=0.5,
                        color="black",
                        )
        self.lg.append(lg1)
        ax1.tick_params(axis='both', labelsize=5)
        ax1.set_xlim([0, self._time_limit * self._framerate_color])
        ax1.set_title("Cumulative distance over time", fontsize=5)
        ax1.set_ylabel("Cumulative distance (cm)", fontsize=5)
        ax1.tick_params(bottom=False, labelbottom=False)

        if self._args["plot"]["general"]["nesting_subplots"]:
            ax2 = plt.subplot2grid((4, 4), (2, 3), sharex=ax0)
            lg2, = ax2.plot(np.arange(0, len(self.nm_average_height_smooth)),
                            self.nm_average_height_smooth,
                            linewidth=0.25,
                            color="black"
                            )
            self.lg.append(lg2)
            ax2.set_ylim([np.min(self.nm_average_height_smooth),
                          np.max(self.nm_average_height_smooth)])
            ax2.tick_params(axis='both', labelsize=5)
            ax2.set_xlim([0, self._time_limit * self._framerate_color])
            ax2.set_title("Object height over time", fontsize=5)
            ax2.set_ylabel("Average pixel intensity *8bit (a.u)", fontsize=5)
            ax2.tick_params(bottom=False, labelbottom=False)

            ax3 = plt.subplot2grid((4, 4), (3, 3), sharex=ax0)
            self.load_raster_file()
            lg3, = ax3.plot(np.arange(0, len(self.nesting_raster)),
                            self.nesting_raster,
                            linewidth=0.5,
                            color="black",
                            )
            self.lg.append(lg3)
            # for bout in raster_data:
            #     ax3.add_patch(patches.Rectangle((bout[0] * self._framerate_color, 0),
            #                                     bout[1] * self._framerate_color - bout[0] * self._framerate_color,
            #                                     1,
            #                                     linewidth=0,
            #                                     color=self._args["plot"]["general"]["color_after"],
            #                                     alpha=1)
            #                   )
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(["no nesting", "nesting"])
            ax3.tick_params(axis='both', labelsize=5)
            ax3.set_title("Behavioral bouts over time", fontsize=5)
            ax3.set_ylabel("Behavioral bouts", fontsize=5)
            ax3.set_xlabel("time (mins)", fontsize=5)
            ax3.set_xticks(np.arange(0, (self._time_limit * self._framerate_color)
                                     / self._args["plot"]["general"]["resolution"] + 60 * self._framerate_color,
                                     60 * self._framerate_color / self._args["plot"]["general"]["resolution"]))
            ax3.set_xticklabels(np.arange(0, self._time_limit / 60 + 1, 1), fontsize=5)
            ax3.set_xlim([0, self._time_limit * self._framerate_color])

        self.ax4 = plt.subplot2grid((4, 4), (0, 0), rowspan=4, colspan=3)
        self.create_tracking_animal(self.ax4, lg=True)
        plt.tight_layout()

        if self._args["plot"]["general"]["save"]:
            plt.savefig(os.path.join(self._args["general"]["segmentation_directory"],
                                     "complete_tracking_animal_{0}".format(self._animal_tag)), dpi=300)

        self.slider()
        plt.show()

    def slider(self):
        ax = plt.axes([0.1, 0.02, 0.6, 0.02], facecolor='white')
        self.slider_widget = Slider(ax,
                                    "time",
                                    0,
                                    self._args["general"]["video"]["end"] - self._args["general"]["video"]["start"] - 1,
                                    valinit=self._args["general"]["video"]["end"] - self._args["general"]["video"][
                                        "start"],
                                    valstep=1,
                                    color='black')
        self.slider_widget.label.set_size(5)
        val_text = "{0}s".format(int("".join([x for x in self.slider_widget.valtext.get_text() if x.isdigit()])) + 1)
        self.slider_widget.valtext.set_text(val_text)
        self.slider_widget.valtext.set_fontsize(5)
        self.slider_widget.on_changed(self.update_slider)

    def update_slider(self, val):
        val_text = "{0}s".format(int("".join([x for x in self.slider_widget.valtext.get_text() if x.isdigit()])) + 1)
        self.slider_widget.valtext.set_text(val_text)
        distance = self.distance_cropped.copy()
        distance[int(self.slider_widget.val * self._framerate_color):] = None
        self.lg[0].set_ydata(distance)
        distance_cumulative = self.distance_cumulative.copy()
        distance_cumulative[int(self.slider_widget.val * self._framerate_color):] = None
        self.lg[1].set_ydata(distance_cumulative)
        nm_average_height = self.nm_average_height_smooth.copy().astype("float")
        nm_average_height[int(self.slider_widget.val * self._framerate_color):] = None
        self.lg[2].set_ydata(nm_average_height)
        nm_raster = self.nesting_raster.copy()
        nm_raster[int(self.slider_widget.val * self._framerate_color):] = None
        self.lg[3].set_ydata(nm_raster)
        positions = self.positions_cropped.copy()
        if int(self.slider_widget.val * self._framerate_color) > self.track_length * self._framerate_color:
            positions[:int((self.slider_widget.val - self.track_length) * self._framerate_color)] = \
                positions[int((self.slider_widget.val - self.track_length) * self._framerate_color)]
        positions[int(self.slider_widget.val * self._framerate_color):] = \
            positions[int(self.slider_widget.val * self._framerate_color)]
        self.lg[4].set_data(positions[:, 0], positions[:, 1])
        self.lg[5].set_data(positions[-1, 0], positions[-1, 1])
        # frame_to_display = self.segmentation_video_resized.get_frame(self.slider_widget.val)
        # self.ax4.imshow(frame_to_display)
        self.fig.canvas.draw_idle()

    def make_frame_mpl(self, t):
        i = int(t)
        if i < self.thresh_display:
            try:
                self.nm_height_graph.set_data(self.live_plot_x[0:i],
                                              self.live_plot_nm[self.start_live_plot:self.start_live_plot + i])
                self.raster_graph.set_data(self.live_plot_x[0:i],
                                           self.live_plot_raster[self.start_live_plot:self.start_live_plot + i])
            except:
                pass
            last_frame = mplfig_to_npimage(self.live_figure)
            return last_frame
        else:
            delta = i - self.thresh_display
            xtick_labels = np.arange(delta / self._framerate_color,
                                     i / self._framerate_color
                                     + self.thresh_display / (self._framerate_color * 5),
                                     self.thresh_display / (self._framerate_color * 5))
            xtick_labels = [int(j) for j in xtick_labels][:self.n_x_tick_labels]
            self.live_ax0.set_xticks(np.arange(self.start_live_plot + delta,
                                               self.start_live_plot + i + self.thresh_display / 5,
                                               self.thresh_display / 5))
            self.live_ax0.set_xticklabels(xtick_labels)
            self.live_ax0.set_xlim(self.start_live_plot + delta, self.start_live_plot + i)
            self.live_ax1.set_xticks(np.arange(self.start_live_plot + delta,
                                               self.start_live_plot + i + self.thresh_display / 5,
                                               self.thresh_display / 5))
            self.live_ax1.set_xticklabels(xtick_labels)
            self.live_ax1.set_xlim(self.start_live_plot + delta, self.start_live_plot + i)
            try:
                self.nm_height_graph.set_data(self.live_plot_x[delta:i], self.live_plot_nm[
                                                                         self.start_live_plot + delta:self.start_live_plot + i])
                self.raster_graph.set_data(self.live_plot_x[delta:i], self.live_plot_raster[
                                                                      self.start_live_plot + delta:self.start_live_plot + i])
            except:
                pass
            last_frame = mplfig_to_npimage(self.live_figure)
            return last_frame

    def smooth(self, source, win=11, order=3):
        sink = savgol_filter(source, win, order)
        return sink

    def live_tracking_plot(self, res=1, start_live_plot=0, end_live_plot=5 * 60, acceleration=1):
        self.start_live_plot = int((start_live_plot) * self._framerate_color)
        self.end_live_plot = int((end_live_plot) * self._framerate_color)
        self.duration_live_plot = abs(self.end_live_plot - self.start_live_plot)
        self.thresh_display = int(100 * self._framerate_color)
        if self.duration_live_plot == 0:
            raise RuntimeError('Video length has to be > 0 !')
        self.segmentation_video = mpy.VideoFileClip(self._segmentation_video)
        self.segmentation_video_small = self.segmentation_video.resize(0.25)
        self.segmentation_video_subclip = self.segmentation_video_small.subclip(t_start=start_live_plot,
                                                                                t_end=end_live_plot)

        if self.segmentation_video_subclip.duration == 0:
            raise RuntimeError('videoClip is empty !')

        self.live_figure = plt.figure(figsize=(5, 3), facecolor='white')
        self.gs = self.live_figure.add_gridspec(2, 2)
        self.live_ax0 = self.live_figure.add_subplot(self.gs[0, 0:])
        self.live_ax1 = self.live_figure.add_subplot(self.gs[1, 0:])
        self.live_ax0.invert_yaxis()
        self.live_ax1.invert_yaxis()
        self.live_plot_x = np.arange(self.start_live_plot, self.end_live_plot)
        self.live_plot_nm = [np.mean(self._nm_average_height[int(i):int(i + self._framerate_color / res)])
                             for i in np.arange(0, len(self._nm_average_height), self._framerate_color / res)]
        self.live_plot_nm = self.smooth(np.array(self.live_plot_nm), win=int(5 * self._framerate_color))
        self.live_plot_raster = [np.mean(self.nesting_raster[int(i):int(i + self._framerate_color / res)])
                                 for i in np.arange(0, len(self.nesting_raster), self._framerate_color / res)]

        self.live_ax0.set_title("Distance of nesting material to camera")
        self.live_ax0.set_ylabel("Mean pixel value")
        self.live_ax0.set_ylim(min(self.live_plot_nm), max(self.live_plot_nm))
        self.live_ax0.set_xticks(np.arange(self.start_live_plot,
                                           self.start_live_plot + self.thresh_display + self.thresh_display / 5,
                                           int(self.thresh_display / 5)))
        self.live_ax0.set_xticklabels(np.arange(0,
                                                self.thresh_display / self._framerate_color + self.thresh_display / (
                                                        self._framerate_color * 5),
                                                self.thresh_display / (self._framerate_color * 5)))
        self.live_ax0.set_xlim(self.start_live_plot, self.start_live_plot + self.thresh_display)

        self.live_ax1.set_title("Nest-building raster")
        self.live_ax1.set_yticks([0, 1])
        self.live_ax1.set_yticklabels(["no nesting", "nesting"])
        self.live_ax1.set_ylim(-0.5, 1.5)
        self.live_ax1.set_xticks(np.arange(self.start_live_plot,
                                           self.start_live_plot + self.thresh_display + self.thresh_display / 5,
                                           self.thresh_display / 5))
        x_tick_labels = np.arange(0,
                                  self.thresh_display / self._framerate_color + self.thresh_display / (
                                          self._framerate_color * 5),
                                  self.thresh_display / (self._framerate_color * 5))
        self.n_x_tick_labels = len(x_tick_labels)
        self.live_ax1.set_xticklabels(x_tick_labels)
        self.live_ax1.set_xlim(self.start_live_plot, self.start_live_plot + self.thresh_display)
        self.live_ax1.set_xlabel("Time (s)")

        self.nm_height_graph, = self.live_ax0.plot(self.live_plot_x[0:1], self.live_plot_nm[0:1],
                                                   lw=1., c="blue", alpha=0.5)
        self.raster_graph, = self.live_ax1.plot(self.live_plot_x[0:1], self.live_plot_raster[0:1],
                                                lw=2, c="blue", alpha=0.5)
        plt.tight_layout()
        self._acceleration = acceleration
        self.animation = mpy.VideoClip(self.make_frame_mpl,
                                       duration=(self.duration_live_plot * res) / self._framerate_color)
        self.finalClip = clips_array([[clip.margin(2, color=[255, 255, 255]) for clip in
                                       [(self.segmentation_video_subclip.resize(0.6).speedx(self._acceleration)),
                                        self.animation.speedx(self._acceleration * res)]]],
                                     bg_color=[255, 255, 255])
        self.finalClip.write_videofile(os.path.join(self._args["general"]["segmentation_directory"],
                                                    'live_tracking_{0}.mp4'
                                                    .format(self._args["general"]["general"]["animal_tag"])),
                                       fps=10)
