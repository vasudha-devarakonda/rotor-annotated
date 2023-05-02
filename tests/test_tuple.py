import rotor
from torch import nn
import torch

class TupleMake(nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, x):
        f = self.first(x)
        s = self.second(x)
        return (f, s)

class TupleApply(nn.Module):
    def __init__(self, module, *args, **kwargs):
        super().__init__()
        self.first = module(*args, **kwargs)
        self.second = module(*args, **kwargs)

    def forward(self, t):
        (x, y) = t
        f = self.first(x)
        s = self.second(y)
        return (f, s)

class TupleAny(nn.Module):
    def __init__(self, *functions):
        super().__init__()
        self.functions = functions

    def forward(self, t):
        result = tuple(self.functions[i](x) for i, x in enumerate(t))
        return result

verbosity = 3

from rotor.algorithms.sequence import *
def hardcode_sequence(chk):
    core_length = len(chk.functions)
    sequence = Sequence(Function("HardCoded"))
    for i in range(core_length-1):
        sequence.insert(ForwardEnable(i))
    sequence.insert(ForwardCheck(core_length-1))
    sequence.insert(Loss())
    sequence.insert(ForwardEnable(core_length-1))
    sequence.insert(Backward(core_length-1))
    for i in reversed(range(core_length-1)):
        sequence.insert(Backward(i))
    chk.sequence = sequence


# works if input is a Tensor
tuple_list = [ TupleApply(nn.Conv2d, 1, 1, 3, padding=(1, 1)) for _ in range(5) ]
model = nn.Sequential(TupleMake(nn.Conv2d(1, 1, 3, padding=(1, 1)),
                                nn.Conv2d(1, 1, 5, padding=(3, 3))),
                      *tuple_list)
model = rotor.Checkpointable(model, verbosity=verbosity)

shape = (1, 1, 100*100, 10)
sample_input = torch.randn(*shape)
print("Measuring")
model.measure(sample_input)
print("Computing")
hardcode_sequence(model)

inp = torch.randn(*shape).requires_grad_()
print("Forward")
res = model(inp)
res = res[0].sum() + res[1].sum()
print("Backward")
res.backward()


# works if input is a tuple
tuple_list = [ TupleApply(nn.Conv2d, 1, 1, 3, padding=(1, 1)) for _ in range(5) ]
model = nn.Sequential(*tuple_list)
model = rotor.Checkpointable(model, verbosity=verbosity)

shape = (1, 1, 100*100, 10)
sample_input = (torch.randn(*shape), torch.rand(*shape))
print("Measuring")
model.measure(sample_input)
print("Computing")
hardcode_sequence(model)

inp = (torch.randn(*shape).requires_grad_(),
       torch.rand(*shape).requires_grad_())
print("Forward")
res = model(inp)
res = res[0].sum() + res[1].sum()
print("Backward")
res.backward()
print(inp[0].grad)


# works if input is a Tensor and some layers are frozen
## Freeze the first 2 layers
def is_frozen(idx):
    return (idx < 2)

def freeze(model, frozen_func):
    for idx, m in enumerate(model):
        if frozen_func(idx):
            for p in m.parameters():
                p.requires_grad = False
def check_correctness(model, frozen_func):
    for i in range(len(model)):
        assert(not frozen_func(i) == any(p.grad is not None for p in model[i].parameters()))
                
tuple_list = [ TupleApply(nn.Conv2d, 1, 1, 3, padding=(1, 1)) for _ in range(5) ]
model = nn.Sequential(TupleMake(nn.Conv2d(1, 1, 3, padding=(1, 1)),
                                nn.Conv2d(1, 1, 5, padding=(3, 3))),
                      *tuple_list)
freeze(model, is_frozen)

model = rotor.Checkpointable(model, verbosity=verbosity)

shape = (1, 1, 100*100, 10)
sample_input = torch.randn(*shape)
print("Measuring")
model.measure(sample_input)
print("Computing")
hardcode_sequence(model)

inp = torch.randn(*shape).requires_grad_()
print("Forward")
res = model(inp)
res = res[0].sum() + res[1].sum()
print("Backward")
res.backward()
check_correctness(model.model, is_frozen)

                
# works if input is a tuple and some layers are frozen?

tuple_list = [ TupleApply(nn.Conv2d, 1, 1, 3, padding=(1, 1)) for _ in range(5) ]
model = nn.Sequential(*tuple_list)
freeze(model, is_frozen)

model = rotor.Checkpointable(model, verbosity=verbosity)

shape = (1, 1, 100*100, 10)
sample_input = (torch.randn(*shape), torch.rand(*shape))
print("Test with frozen layers: Measuring")
model.measure(sample_input)
print("Computing")
hardcode_sequence(model)

inp = (torch.randn(*shape),
       torch.rand(*shape))
print("Forward")
res = model(inp)
res = res[0].sum() + res[1].sum()
print("Backward")
res.backward()
check_correctness(model.model, is_frozen)


exit()

# Does not work if output or input contains a non-Tensor
## Because torch.autograd.backward() requires a Sequence of Tensors
shape = (1, 1, 100*100, 10)
tuple_list = ( [ TupleAny(nn.Conv2d(1, 1, 3, padding=(1, 1)),
                          lambda v: torch.randn(*shape).requires_grad_(),
                          nn.Conv2d(1, 1, 5, padding=(3, 3))) ]
               + [ TupleAny(nn.Conv2d(1, 1, 3, padding=(1, 1)),
                            nn.Conv2d(1, 1, 3, padding=(1, 1)),
                            nn.Conv2d(1, 1, 5, padding=(3, 3)))  for _ in range(5) ] )
print(tuple_list)
model = nn.Sequential(*tuple_list)
print("Params")
for p in model.parameters():
    print(p)
model = rotor.Checkpointable(model, verbosity=verbosity)

sample_input = (torch.randn(*shape), False, torch.rand(*shape))
print("Measuring")
model.measure(sample_input)
print("Computing")
hardcode_sequence(model)

inp = (torch.randn(*shape).requires_grad_(), False,
       torch.rand(*shape).requires_grad_())
print("Forward")
res = model(inp)
res = res[0].sum() + res[1].sum()
print("Backward")
res.backward()

