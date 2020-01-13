import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class AtomicSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super(AtomicSequential, self).__init__(*args, **kwargs)
        self.not_really_sequential = True

## Because in-place operations do not mix well with partial checkpointing
## This function returns a module that does an in-place ReLU operation after
## the given module

def attachReLU(layer):
    class LayerWithReLU(layer):
        def __init__(self, *args, **kwargs):
            super(LayerWithReLU, self).__init__(*args, **kwargs)
            self._opt_chk_relu = nn.ReLU(inplace=True)
        def forward(self, x):
            x = super(LayerWithReLU, self).forward(x)
            x = self._opt_chk_relu(x)
            return x
    return LayerWithReLU

class BothOutputs(nn.Module):

    def __init__(self, module_left, module_right, concat):
        super(BothOutputs, self).__init__()
        self.left = module_left
        self.right = module_right
        self.concat = concat

    def forward(self, x):
        hasRightValue = False
        if self.training and self.right: 
            rightValue = self.right(x)
            hasRightValue = True
        x = self.left(x)
        if hasRightValue:
            return self.concat(x, rightValue)
        return x

    
