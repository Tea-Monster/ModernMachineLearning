import numpy as np
from PIL import Image

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