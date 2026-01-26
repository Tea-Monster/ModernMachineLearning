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

    def __init__(self, path, transitions=None):
        self.path = path
        self.image = Image.open(path)
        self.width, self.height = self.image.size
        self.transitions = transitions
    
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
    

class LineSegment():
    def __init__(self, file_name: str, file_path: str, segment_start: int, segment_end:int, segment_text: str=None):
        self.file_name = file_name
        self.file_path = file_path
        self.segment_start = segment_start
        self.segment_end = segment_end
        self.segment_text = segment_text

    # Returns the line segment cropped out of an image
    def segment_image(self):
        full_image = Image.open(self.file_path)
        _, width = full_image.size
        return full_image.crop((0, self.segment_start, width, self.segment_end))

    # Turn image into black and white
    @staticmethod
    def black_and_white(image: Image, threshold=200):
        fn = lambda x : 255 if x > threshold else 0
        image = image.convert('L').point(fn, mode='1')
        return image

    # Horizontal projection
    @staticmethod
    def horizontal_projection(image: Image):
        pixels = np.array(image)
        horizontal_projection = np.sum(255 - pixels, axis=1)
        return horizontal_projection

    # Normalize values so that they are all between 0 and 1
    @staticmethod
    def normalize_projection(projection):
        minimum, maximum = min(projection), max(projection)
        span = maximum - minimum
        normalized = [(value - minimum) / span for value in projection]
        return normalized

    # Remove noise by setting every value below a certain threshold to 0
    @staticmethod
    def cut_noise_from_projection(projection, threshold=0.05):
        projection = [value if value > threshold else 0 for value in projection]
        return projection

    # Smooth out a the projection
    @staticmethod
    def smoothen_projection(data, window_len=70):
        kernel = np.ones(window_len, 'd')
        smoothed_out = np.convolve(kernel/kernel.sum(), data, mode='same')
        return smoothed_out

    # Calculate Local Minima
    @staticmethod
    def calculate_local_minima(image: Image):
        image = LineSegment.black_and_white(image)

        projection = LineSegment.horizontal_projection(image)
        projection = LineSegment.normalize_projection(projection)
        projection = LineSegment.smoothen_projection(projection)
        projection = LineSegment.cut_noise_from_projection(projection)
        projection = LineSegment.smoothen_projection(projection)

        # Signum of the first order "derivative"
        grad_sign = []
        for i in range(len(projection)-1):
            delta = projection[i+1] - projection[i]
            grad_sign.append(np.sign(delta))

        # Collect minima based on the second order "derivative"
        minima = []
        for i in range(len(projection)-2):
            if grad_sign[i+1] - grad_sign[i] > 0:
                minima.append(i+1)
        return minima

    # Create the list of segments
    @staticmethod
    def calculate_segments(image: Image, threshold=100):
        minima = LineSegment.calculate_local_minima(image)
        list_of_segments = []
        for i in range(len(minima)-1):
            m1 = minima[i]
            m2 = minima[i+1]
            if m2 - m1 > threshold:
                list_of_segments.append((m1, m2))
        return list_of_segments