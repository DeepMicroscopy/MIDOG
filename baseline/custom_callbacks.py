from fastai.vision import *
from fastai.callbacks import TrackerCallback

class UpdateAlphaCallback(TrackerCallback):
    def __init__(self, learn:Learner, max_epochs):
        super().__init__(learn)
        self.max_epochs = max_epochs

    def on_epoch_begin(self,epoch, **kwargs:Any):
        p = (epoch + 1) / self.max_epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        self.learn.model.d5.alpha = alpha

