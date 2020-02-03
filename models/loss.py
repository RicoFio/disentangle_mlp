
from torch.nn.modules.loss import _Loss

class VaeGanLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(VaeGanLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        #TODO implement the loss
        raise NotImplementedError