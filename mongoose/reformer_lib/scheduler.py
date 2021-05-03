import torch
from mongoose.slide_lib.simHash import SimHash


class Scheduler:
    def __init__(self, data, D, k=1, l=10, thresh=0.01):
        self.thresh_hash = SimHash(D, k, l)
        self.hash_codes = self.thresh_hash.hash(data)
        self.thresh = thresh

    def detect_change(self, updated_data):
        check = self.thresh_hash.hash(updated_data)
        distance = check - self.hash_codes
        if torch.sum(torch.abs(distance)) > self.thresh*distance.numel():
            self.hash_codes = check
            return True
        else:
            return False
