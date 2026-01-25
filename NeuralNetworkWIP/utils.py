import numpy as np
from PIL import Image
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt


class SplitImage():
    '''
    Usage example:

    from Utils import SplitImage

    si = SplitImage('some/file/path')
    handwritten_area = si.handwritten_area()
    handwritten_area.save('handwritten.png')
    '''

    def __init__(self, path):
        self.path = path
        self.image = Image.open(path)
        self.width, self.height = self.image.size
        self.transitions = None
    
    def black_and_white(self):
        threshold = 195
        fn = lambda x : 255 if x > threshold else 0
        return self.image.convert('L').point(fn, mode='1')
    
    def header_area(self):
        transitions = self._area_transitions()
        area = (0, 0, self.width, transitions[0])
        return self.image.crop(area)
    
    def printed_area(self):
        transitions = self._area_transitions()
        area = (0, transitions[1], self.width, transitions[2])
        return self.image.crop(area)

    def handwritten_area(self):
        transitions = self._area_transitions()
        area = (0, transitions[3], self.width, transitions[4])
        return self.image.crop(area)

    def footer_area(self):
        transitions = self._area_transitions()
        area = (0, transitions[5], self.width, self.height)
        return self.image.crop(area)

    def _horizontal_runs_of_black(self):
        bw = np.array(self.black_and_white())
        runs_of_black = []
        for i in range(self.height):
            max = 0
            run = 0
            for j in range(self.width):
                if bw[i, j] == 0:
                    run += 1
                else:
                    run = 0

                if run > max:
                    max = run
            runs_of_black.append(max)
        return runs_of_black
    
    def _mask_areas(self):
        runs_of_black = self._horizontal_runs_of_black()
        mask = [1] * self.height
        for i in range(self.height):
            if runs_of_black[i] > 200:
                mask[i-2] *= 0
                mask[i-1] *= 0
                mask[i] *= 0
                mask[i+1] *= 0
                mask[i+2] *= 0
        return mask
    
    def _area_transitions(self):
        if self.transitions is None:
            mask = self._mask_areas()
            transitions = []
            for i in range(self.height-1):
                if abs(mask[i] - mask[i+1]) > 0:
                    transitions.append(i)
            self.transitions = transitions
        return self.transitions
    


class LineSegments():
    
    def __init__(self, path, window_len=60, window='flat', line_height_threshold=100):
        self.path = path
        self.img = Image.open(path)
        self.pixelized = np.array(self.img)
        self.window_len = window_len
        self.window = window
        self.line_height_threshold = line_height_threshold
        self.line_segments = None


    # code from https://www.kaggle.com/code/irinaabdullaeva/text-segmentation
    def smooth(self, x, window_len=45, window='hanning'):
        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.") 
        if window_len<3:
            return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        if window == 'flat': #moving average
            w = np.ones(window_len,'d')
        else:
            w = eval('np.'+window+'(window_len)')

        y = np.convolve(w/w.sum(),s,mode='valid')
        return y

    def crop_lines(self, local_minima, threshold=0):
        x1 = 0
        cropped = []
        diff = []
        for i, min in enumerate(local_minima):
            x2 = min
            #print(f"x1 = {x1}, x2 = {x2}, diff = {x2-x1}")
            if x2-x1 >= threshold:
                cropped.append((x1, x2))
            x1 = min
        return cropped

    def show_crop_lines(self, img, cropped):
        plots = len(cropped)
        for i, l in enumerate(cropped):
            line = img[l[0]:l[1]]
            plt.subplot(plots, 1, i+1)
            plt.axis('off')
            _ = plt.imshow(line, cmap='gray')
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis


    def segment_lines(self):
        horizontal_projection = np.sum(255 - self.pixelized, axis=1)
        smoothed = self.smooth(horizontal_projection, self.window_len, window='flat')
        local_minima = argrelextrema(smoothed, np.less)
        local_minima = np.array(local_minima).flatten()
        cropped = self.crop_lines(local_minima, self.line_height_threshold)
        return cropped


