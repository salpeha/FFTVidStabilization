# -*- coding: utf-8 -*-
"""
@author: spena
Based on OpenPIV and on OpenCV
"""


import wx
import os
import cv2
import numpy as np
from  scipy import ndimage
import subprocess
import json
import sys
import linecache
import shutil

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as\
     NavigationToolbar

def checkFolders(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    working_dir = os.path.dirname(video_path)
    frame_folder = working_dir + '/' + video_name + '_frames/'
    stab_folder = working_dir + '/' + video_name + '_stabilized/'
    file_folder = working_dir + '/' + video_name + '_res/'
    plot_folder = working_dir + '/' + video_name + '_plots/'

    if not os.path.isdir(frame_folder):
        os.mkdir(frame_folder)
    if not os.path.isdir(stab_folder):
        os.mkdir(stab_folder)
    if not os.path.isdir(file_folder):
        os.mkdir(file_folder)
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)    

    for r in range(0, 4):
        tmp_folder  = working_dir + '/' + video_name + '_piv' + str(r)
        if not os.path.isdir(tmp_folder):
            os.mkdir(tmp_folder)

    return working_dir, frame_folder, stab_folder, file_folder, plot_folder


class VidStab(object):
    def __init__(self, video_path):
        self.video_path = video_path
        self.checkVidExits()

        self.piv = PIV()

        self.getFPS()
        self.checkFolders()

        self.frames_piv = []
        self.msgs = []

    def getROI(self):
        self.rois = []
        for r in range(0, len(self.gcp)):
            tmp_roi = self.gcp[r]
            xy_array = np.zeros((4, 2))
            
            xy_array[0, 0] = tmp_roi[1] - self.win_size / 2
            xy_array[0, 1] = tmp_roi[0] + self.win_size / 2

            xy_array[1, 0] = tmp_roi[1] + self.win_size / 2
            xy_array[1, 1] = tmp_roi[0] + self.win_size / 2
            
            xy_array[2, 0] = tmp_roi[1] - self.win_size / 2
            xy_array[2, 1] = tmp_roi[0] - self.win_size / 2
            
            xy_array[3, 0] = tmp_roi[1] + self.win_size / 2
            xy_array[3, 1] = tmp_roi[0] - self.win_size / 2

            if len(xy_array[xy_array<0]) > 0:
                wx.MessageBox("Negative ROI %d. Image cannot be cropped" %r,
                              'getRoi',
                              wx.OK | wx.ICON_INFORMATION)
                return False

            self.rois.append(xy_array)

    def checkVidExits(self):
        if not os.path.exists(self.video_path):
            sys.exit("Video %r was not found!" % (self.video_path,))

    def getFPS(self):
        vid = cv2.VideoCapture(self.video_path)

        self.fps = vid.get(cv2.cv.CV_CAP_PROP_FPS)
        self.video_length = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

        print "Video FPS: ", self.fps
        print "Num frames: ", self.video_length

    def checkFolders(self):
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.working_dir = os.path.dirname(self.video_path)
        self.frame_folder = self.working_dir + '/' + video_name + '_frames/'
        self.stab_folder = self.working_dir + '/' + video_name + '_stabilized/'
        self.file_folder = self.working_dir + '/' + video_name + '_res/'
        self.plot_folder = self.working_dir + '/' + video_name + '_plots/'

        if not os.path.isdir(self.frame_folder):
            os.mkdir(self.frame_folder)
        if not os.path.isdir(self.stab_folder):
            os.mkdir(self.stab_folder)
        if not os.path.isdir(self.file_folder):
            os.mkdir(self.file_folder)
        if not os.path.isdir(self.plot_folder):
            os.mkdir(self.plot_folder)    

        for r in range(0, 4):
            tmp_folder  = self.working_dir + '/' + video_name + '_piv' + str(r)
            if not os.path.isdir(tmp_folder):
                os.mkdir(tmp_folder)

    def emptyPivFolders(self, rois='all'):
        if rois == 'all':
            for r in range(len(self.rois)):
                tmp_folder  = self.working_dir + '/piv' + str(r)
                files_del = os.listdir(tmp_folder)
                for f in files_del:
                    os.remove(os.path.join(tmp_folder, f))

    def emptyFolder(self, folder):
        files_del = os.listdir(folder)
        for f in files_del:
            os.remove(os.path.join(folder, f))

    def extractFrames(self):
        self.emptyFolder(self.frame_folder)

        vidcap = cv2.VideoCapture(self.video_path)
        success,image = vidcap.read()
        count = 0

        while success:
            cv2.imwrite(self.frame_folder + "img%03d.jpg" % count, image)
            success, image = vidcap.read()
            if success:
                print "Extracting frame: %d" %count
                pass
            else:
                print "Error extracting frame"
            count += 1

    def getFrames(self):
        files = os.listdir(self.frame_folder)

        if len(files) < 2:
            sys.exit("There are not enough frames in:" % (self.frame_folder,))

        return np.sort(files)

    def getDisplacement(self, gcp, rois='all', win_size=64, save_img=False):
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.win_size = win_size
        self.gcp = gcp
        self.getROI()

        sorted_files = self.getFrames()

        if rois == 'all':
            start = 0
            end = len(self.rois)
        else:
            start = int(rois)
            end = int(rois) + 1
        
        count = 0

        frame_1 = cv2.imread(self.frame_folder + str(sorted_files[0]))
        frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)

        for r in range(start, end):
            disp = []
            print "========================= PIV roi: ", str(r)
            tmp_folder  = self.working_dir + '/' + video_name + '_piv' + str(r)
            disp_file = self.file_folder  + '/piv' + str(r) + '.txt'

            frame_a = self.cropImg(frame_1, self.rois[r])

            for i in range(0, (len(sorted_files)-1)):
                print "Processing images: ", sorted_files[0], sorted_files[i+1]

                frame_2 = cv2.imread(self.frame_folder + str(sorted_files[i+1]))
                frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
                frame_b = self.cropImg(frame_2, self.rois[r])

                x, y, u, v, s2n = self.calcDisplacement(frame_a, frame_b,
                                                        win_size=self.win_size,
                                                        overlap=0)

                print "u: %0.4f, v: %0.4f, s2n: %0.4f\n" %(np.nanmean(u),
                                                        np.nanmean(v),
                                                        np.average(s2n))

                if save_img:
                    fig, ax = plt.subplots()
                    ax.imshow(frame_a, cmap='gray')
                    ax.imshow(frame_b, cmap='jet', alpha=0.3)
                    ax.quiver(x, y, u, v, color='r', scale_units='xy', scale=1)

                    fig.savefig(tmp_folder + '/fig_' + str(i))
                    plt.close(fig)

                if s2n <= 0:
                    msg = "ROI %d, frame %d s2n is 0. Too much movement\n" % (r, i)
                    print msg
                    self.msgs.append(msg)
                    break

                disp.append([np.nanmean(u), np.nanmean(v)])
               
                count += 1

            disp = np.asarray(disp)
            self.frames_piv.append(i)

            if disp.size > 0:
                disp = self.interpolateNAN(disp)
            self.saveDisp(disp_file, disp)

            print "frames pived: ", self.frames_piv
            print self.msgs

    def calcDisplacement(self, frame_a, frame_b, win_size, overlap):
        # - PIV cross-correlation algorithm
        uu, vv, sig2noise = self.piv.piv(frame_a, frame_b)

        # - Computes the x, y coord of the centers of the interrogation windows
        x, y = self.piv.getCoords(image_size=frame_a.shape,
                                  window_size=win_size,
                                  overlap=overlap)

        return x, y, uu, vv, sig2noise

    def cropImg(self, img, roi):
        #Check there are only 4 points
        if roi.shape[0] > 4:
            sys.exit("ROI has more than 4 points")

        start_i = int(min(roi[:, 1]))
        end_i = int(max(roi[:, 1]))
        start_j = int(min(roi[:, 0]))
        end_j = int(max(roi[:, 0]))

        img_cut = img[start_j:end_j, start_i:end_i]

        return img_cut

    def saveDisp(self, file_name, values):
        tmp_file = open(file_name, 'w')
        for v in range(0, len(values)):
            tmp_file.write(str(values[v, 0]) + ',' + str(values[v, 1]) + '\n')
        tmp_file.close()

    def interpolateNAN(self, values):
        # - check in which lines are nan's -
        index_nan = np.argwhere(np.isnan(values[:, 0]))
        if index_nan.size != 0:
            print "WARNNG: %d nan's were found and will be interpolated" % len(index_nan)
        else:
            return values

        # - check if there are more than 1 nan in between two valid values -
        dist_nan = index_nan[range(1, index_nan.size)] - \
                   index_nan[range(0, index_nan.size - 1)]
        dist_nan = np.append(dist_nan, [0])

        # - get how many consequtive nan's -
        consec = []
        count = 0
        for i in range(0, len(dist_nan)):
            if dist_nan[i] == 1:
                count += 1
            else:
                if count != 0:
                    consec.append(count)
                    count = 0

        # - interpolate -
        count_consec = 0
        count_nan = 0
        while count_nan < index_nan.size:
            if dist_nan[count_nan] != 1:
                dist = 1
                values = self.linearInterpolation(values, index_nan,
                                                   count_nan, dist)
                # print "Count nan", count_nan
                count_nan += 1
            else:
                dist = consec[count_consec] + 1
                values = self.linearInterpolation(values, index_nan,
                                                   count_nan, dist)
                # print "Count nan", count_nan
                # print "Consecutive", consec[count_consec]
                count_nan += consec[count_consec] + 1
                count_consec += 1
        print "Nans removed"

        return values

    def linearInterpolation(self, values, index_nan, count_nan, dist):
        for d in range(0, dist):
            v_before = values[index_nan[count_nan] - 1, 0]
            v_after = values[index_nan[count_nan] + dist, 0]
            values[index_nan[count_nan + d], 0] = \
                (v_before + v_after) * (d + 1) / (1 + dist)

            u_before = values[index_nan[count_nan] - 1, 1]
            u_after = values[index_nan[count_nan] + dist, 1]
            values[index_nan[count_nan + d], 1] = \
                (u_before + u_after) * (d + 1) / (1 + dist)

        return values

    def stabilize(self, gcp, rois_to_use, transformation='affine'):
        self.getTrajectories(gcp, rois_to_use)
 
        files_del = os.listdir(self.stab_folder)
        for f in files_del:
            os.remove(self.stab_folder + f)

        shutil.copy(self.frame_folder + "img001.jpg", self.stab_folder +
                    "img000.jpg")

        #sorted_imgs, ext = self.getFrames()
        sorted_imgs = self.getFrames()

        pived_frames = []
        for t in range(len(self.all_trajec)):
            pived_frames.append(len(self.all_trajec[t]))
        min_pived = min(pived_frames)

        for i in range(1, min_pived):
            img = cv2.imread(self.frame_folder + str(sorted_imgs[i]))
            rows, cols, ch = img.shape
            pts_org, pts_disp = self.getTransfPoints(i)

            if transformation == 'affine':
                if len(self.gcp) < 3:
                    sys.exit('Not enough control points. 3 needed')

                M = cv2.getAffineTransform(pts_disp, pts_org)
                img_stabi = cv2.warpAffine(img, M, (cols, rows))

            elif transformation == 'perspective':
                if len(self.gcp) < 4:
                    sys.exit('Not enough control points. 4 needed')

                M = cv2.getPerspectiveTransform(pts_disp, pts_org)
                img_stabi = cv2.warpPerspective(img, M, (cols, rows))

            img_name = self.stab_folder + str("img%03d" % (i)) + '.jpg'
            cv2.imwrite(img_name, img_stabi)
            print "Image stabilized: ", i

    def getTrajectories(self, gcp, rois_to_use):
        self.gcp = gcp
        self.readDisplacement(rois_to_use)

        self.all_trajec = []
        for r in range (0, len(self.disp_gcp)):
            pts_trajec = np.zeros(shape=(len(self.disp_gcp[r]), 2))
            ini_point = self.gcp
            pts_trajec[0, 0] = ini_point[r][0]
            pts_trajec[0, 1] = ini_point[r][1]

            for p in range(0, len(self.disp_gcp[r])):
                pts_trajec[p, 0] = ini_point[r][0] + self.disp_gcp[r][p, 0]
                pts_trajec[p, 1] = ini_point[r][1] + self.disp_gcp[r][p, 1] * -1

            self.all_trajec.append(pts_trajec)

    def getTransfPoints(self, img_num):
        pts_org = np.zeros((len(self.gcp), 2))
        pts_dst = np.zeros((len(self.gcp), 2))

        for r in range(len(self.gcp)):
            pts_org[r, :] = self.all_trajec[r][0, :]
            pts_dst[r, :] = self.all_trajec[r][img_num, :]

        return np.float32(pts_org), np.float32(pts_dst)

    def plotTrajectories(self, gcp, rois_to_use, show_img=False):
        sorted_imgs = self.getFrames()
        img = cv2.imread(self.frame_folder + str(sorted_imgs[0]))
        self.getTrajectories(gcp, rois_to_use)

        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')

        for r in range(0, len(self.all_trajec)):
            ax.plot(self.all_trajec[r][:, 0], self.all_trajec[r][:, 1],
                    marker='o')
        ax.set_xlabel('i')
        ax.set_xlabel('j')
        # ax.set_xlim([0, img.shape[1]])
        # ax.set_ylim([0, img.shape[0]])
        plt.savefig(self.plot_folder + '/trajectories.png')
        if show_img:
            plt.show()
        plt.close()

    def readDisplacement(self, rois_to_use):
        self.disp_gcp = []
       
        for r in range(len(rois_to_use)):
            if rois_to_use[r]:
                tmp_file = self.file_folder  + 'piv' + str(r) + '.txt'
                values = self.readFile(tmp_file)
                self.disp_gcp.append(values)

    def plotDisplacement(self, gcp, rois_to_use, show_img=False):
        self.getTrajectories(gcp, rois_to_use)

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

        for p in range(0, len(self.disp_gcp)):
            plt.plot(self.disp_gcp[p][:, 0], color=colors[p],
                     label='roi:%d, dx'%p)
            plt.plot(self.disp_gcp[p][:, 1], color=colors[p], linestyle='--',
                     label='roi:%d, dy'%p)
        plt.ylabel('pixels')
        plt.xlabel('frame number')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                   ncol=len(self.disp_gcp), prop={'size': 10})
        plt.savefig(self.plot_folder + '/displacements.png')
        if show_img:
            plt.show()
        plt.close()

    def centerROI(self, roi):
        start_i = int(min(roi[:, 1]))
        end_i = int(max(roi[:, 1]))
        start_j = int(min(roi[:, 0]))
        end_j = int(max(roi[:, 0]))

        return np.array(((start_i + end_i) / 2, (start_j + end_j) / 2))

    def readFile(self, file_name):
        num_lines = 0
        ffile = open(file_name)
        for line in ffile:
            num_lines += 1
        ffile.close()

        values = np.zeros(shape=(num_lines, 2))
        for l in range(1, num_lines):
            part_line = linecache.getline(file_name, l)
            tmp = part_line.split(',')
            values[l, 0] = float(tmp[0])
            values[l, 1] = float(tmp[1].split('\n')[0])
        linecache.clearcache()

        return values

    def videoFromImgs(self):
        print "=== Creating video..."
        dst_name = str(self.working_dir + "/vid_stabilized.avi")
        if os.path.isfile(dst_name):
            os.remove(dst_name)
    
        frame_array = []

        imgs = os.listdir(self.stab_folder)
        files = np.sort(imgs)
       
        count = 0
        for i in range(len(files)):
            filename = self.stab_folder + files[i]

            img = cv2.imread(filename)
            height, width, _ = img.shape
            size = (width, height)
            count += 1
            frame_array.append(img)
    
        out = cv2.VideoWriter(dst_name, cv2.cv.CV_FOURCC('M','J','P','G'),
                              self.fps, size)
    
        for i in range(len(frame_array)):
            out.write(frame_array[i])
            count += 1
        out.release()

        print "=== Video created ==="


class PIV(object):
    def __init__(self):
        pass

    def correlateWin(self, win_a, win_b):
        win_a = np.subtract(win_a, np.mean(win_a))
        win_b = np.subtract(win_b, np.mean(win_b))

        win_a[win_a < 0] = 0
        win_b[win_b < 0] = 0

        win_bb = win_b[::-1]
        win_bb = np.fliplr(win_bb)

        FFT_a = np.fft.rfft2(win_a)
        FFT_b = np.fft.rfft2(win_bb)
        FFT_a_b = FFT_a * FFT_b
        IFFT_a_b = np.fft.irfft2(FFT_a_b)
        corr = np.fft.fftshift(IFFT_a_b.real, axes=(0, 1))
        corr[corr < 0] = 0

        return corr

    def findFirstPeak(self, corr):
        ind = corr.argmax()
        s = corr.shape[1]

        i = ind // s
        j = ind % s

        return i, j, corr.max()

    def findSecondPeak(self, corr, i=None, j=None, width=2):
        if i is None or j is None:
            i, j, tmp = self.findFirstPeak(corr)

        tmp = corr.view(np.ma.MaskedArray)

        iini = max(0, i - width)
        ifin = min(i + width + 1, corr.shape[0])
        jini = max(0, j - width)
        jfin = min(j + width + 1, corr.shape[1])
        tmp[iini:ifin, jini:jfin] = np.ma.masked
        i, j, corr_max2 = self.findFirstPeak(tmp)

        return i, j, corr_max2

    def findSubpixPeakPosition(self, corr, subpixel_method='gaussian'):
        default_peak_position = (corr.shape[0] / 2, corr.shape[1] / 2)

        peak1_i, peak1_j, dummy = self.findFirstPeak(corr)

        try:
            c = corr[peak1_i, peak1_j]
            cl = corr[peak1_i-1, peak1_j]
            cr = corr[peak1_i+1, peak1_j]
            cd = corr[peak1_i, peak1_j - 1]
            cu = corr[peak1_i, peak1_j + 1]

            if np.any(np.array([c, cl, cr, cd, cu]) < 0) and subpixel_method == 'gaussian':
                subpixel_method = 'centroid'

            try:
                if subpixel_method == 'centroid':
                    subp_peak_position = (((peak1_i-1)*cl+peak1_i*c+(peak1_i+1)*cr)/(cl+c+cr),
                                          ((peak1_j-1)*cd+peak1_j*c+(peak1_j+1)*cu)/(cd+c+cu))

                elif subpixel_method == 'gaussian':
                    subp_peak_position = (peak1_i + ((np.log(cl)-np.log(cr)) / (2*np.log(cl) - 4*np.log(c) + 2*np.log(cr))),
                                          peak1_j + ((np.log(cd)-np.log(cu)) / (2*np.log(cd) - 4*np.log(c) + 2*np.log(cu))))

                elif subpixel_method == 'parabolic':
                    subp_peak_position = (peak1_i + (cl - cr) / (2 * cl - 4*c + 2*cr),
                                        peak1_j + (cd - cu) / (2 * cd - 4*c + 2*cu))

            except:
                subp_peak_position = default_peak_position

        except IndexError:
                subp_peak_position = default_peak_position

        return subp_peak_position[0], subp_peak_position[1]

    def piv(self, frame_a, frame_b):
        corr = self.correlateWin(frame_a, frame_b)
        row, col = self.findSubpixPeakPosition(corr, subpixel_method='gaussian')
        u, v = -(col + 1 - corr.shape[1] / 2), (row + 1 - corr.shape[0] / 2)
        s2n = self.sig2noise(corr, sig2noise_method='peak2peak', width=2)

        return u, v, s2n

    def sig2noise(self, corr, sig2noise_method='peak2peak', width=2):
        peak1_i, peak1_j, corr_max1 = self.findFirstPeak(corr)

        if sig2noise_method == 'peak2peak':
            peak2_i, peak2_j, corr_max2 = self.findSecondPeak(corr ,
                                                              peak1_i,
                                                              peak1_j,
                                                              width=width)

            if corr_max1 < 1e-3 or (peak1_i == 0 or peak1_j == corr.shape[0] or peak1_j == 0 or peak1_j == corr.shape[1] or
                                    peak2_i == 0 or peak2_j == corr.shape[0] or peak2_j == 0 or peak2_j == corr.shape[1]):
                # return zero, since we have no signal.
                return 0.0

        elif sig2noise_method == 'peak2mean':
            corr_max2 = corr.mean()

        else:
            raise ValueError('wrong sig2noise_method')

        try:
            sig2noise = corr_max1 / corr_max2
        except ValueError:
            sig2noise = np.inf

        return sig2noise 

    def getCoords(self, image_size, window_size, overlap):
        field_shape = self.getFieldShape(image_size, window_size, overlap)

        x = np.arange(field_shape[1]) * (window_size-overlap) + (window_size-1) / 2.0
        y = np.arange(field_shape[0]) * (window_size-overlap) + (window_size-1) / 2.0

        return np.meshgrid(x, y[::-1])

    def getFieldShape(self, image_size, window_size, overlap):
        return ((image_size[0] - window_size)//(window_size-overlap)+1,
                (image_size[1] - window_size)//(window_size-overlap)+1)


if __name__ == '__main__':   
    ###########################################################################
    # - path to video -
    input_vid = '/path/to/video/vid.mp4'
    
    # - GCP: [i, j]
    gcp = [[3333, 508], [3294, 1888], [770, 264], [722, 1703]]
  
    # - window size -
    win_size = 64

    # - GCP to use -
    rois_to_use = [True, True, True, True]

    # - transformation type. options 'perspective' (4gcp), 'affine' (3gcp) -
    transformation = 'perspective'
    ###########################################################################

    # - INITIALIZE -
    stabilizer = VidStab(input_vid)

    # - Extract frames -
    stabilizer.extractFrames()

    # - PIVS AND WRITES DISPLACEMENT -
    stabilizer.getDisplacement(gcp, rois='all', save_img=False, win_size=win_size)

    # - PLOTS -
    stabilizer.plotDisplacement(gcp, rois_to_use, show_img=False)
    stabilizer.plotTrajectories(gcp, rois_to_use, show_img=False)

    # - STABILIZE IMAGES affine, perspective-
    stabilizer.stabilize(gcp, rois_to_use, transformation=transformation)

    # - CREATE VIDEO -
    stabilizer.videoFromImgs()