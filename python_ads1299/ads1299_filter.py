from highpass_filter import EnhancedHighPassFilter
from noth_filter import NotchFilter


class EmgFilter:

    def __init__(self):

        self.DictHpFilter = {}
        self.DictHpFilter[500] = EnhancedHighPassFilter(20, 500, 4)
        self.DictHpFilter[1000] = EnhancedHighPassFilter(20, 1000, 4)
        self.DictHpFilter[2000] = EnhancedHighPassFilter(20, 2000, 6)

        self.DictNothFilter = {}
        self.DictNothFilter[500] = NotchFilter(fs=500, filter_type='comb', harmonics=3)
        self.DictNothFilter[1000] = NotchFilter(fs=1000, filter_type='comb', harmonics=3)
        self.DictNothFilter[2000] = NotchFilter(fs=2000, filter_type='comb', harmonics=3)

    def process_buffer(self,rate,buffer):

        output = buffer

        if rate in self.DictHpFilter:
            output = self.DictHpFilter[rate].process_buffer(output)

        if rate in self.DictNothFilter:
            output = self.DictNothFilter[rate].process_buffer(output)

        return output


class EcgFilter:

    def __init__(self):

        self.DictHpFilter = {}
        self.DictHpFilter[125] = EnhancedHighPassFilter(0.5, 125, 2)
        self.DictHpFilter[250] = EnhancedHighPassFilter(0.5, 250, 2)
        self.DictHpFilter[500] = EnhancedHighPassFilter(0.5, 500, 2)
        self.DictHpFilter[1000] = EnhancedHighPassFilter(0.5, 1000, 4)
        self.DictHpFilter[2000] = EnhancedHighPassFilter(0.5, 2000, 6)

    def process_buffer(self,rate,buffer):
        output = buffer
        if rate in self.DictHpFilter:
            output = self.DictHpFilter[rate].process_buffer(buffer)
        return output
