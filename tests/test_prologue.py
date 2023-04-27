import torch
import rotor
import torch.nn as nn


## Freeze the first 3 layers, and layer 7
def is_frozen(idx):
    return (idx == 7) or (idx < 3)


def make_seq(length):
    seq = nn.Sequential(*(nn.Conv1d(1, 1, 3) for _ in range(length)))

    for idx, m in enumerate(seq):
        if is_frozen(idx):
            for p in m.parameters():
                p.requires_grad = False

    return seq


length = 10

def test_without_checkpoint():
    x = torch.randn([1, 1, 100])
    print("No ckpt test")
    seq = make_seq(length)
    y = seq(x).sum()
    print(y)
    y.backward()
    print(x.requires_grad, x.grad)
    for i in range(length):
        assert(not is_frozen(i) == any(p.grad is not None for p in seq[i].parameters()))


def test_with_seq():
    print("Seq pass")
    seq = make_seq(length)
    x = torch.randn([1, 1, 100])
    import torch.utils.checkpoint as ckpt
    y = ckpt.checkpoint_sequential(seq, 3, x).sum()
    print(y)
    y.backward()
    print(x.requires_grad, x.grad)
    for i in range(length):
        print(i, not is_frozen(i), any(p.grad is not None for p in seq[i].parameters()))


from rotor.algorithms.sequence import *

def test_with_rotor():
    x = torch.randn([1, 1, 100])
    seq = make_seq(length)
    chk = rotor.Checkpointable(seq, mem_limit=1024*1024*1024, verbosity=5)
    chk.measure(x)
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
    print(chk.sequence)
    print(x.requires_grad)
    y = chk(x).sum()
    print(y)
    y.backward()
    print(x.requires_grad, x.grad)
    for i in range(length):
        assert(not is_frozen(i) == any(p.grad is not None for p in seq[i].parameters()))


# Another test, this time without frozen layers, but with integer input
def make_seq_int_input(length):
    seq = nn.Sequential(nn.Embedding(10, 30), *(nn.Linear(30, 30) for _ in range(length-1)))

    return seq


def test_with_rotor_int():
    x = torch.randint(10, [10])
    seq = make_seq_int_input(length)
    chk = rotor.Checkpointable(seq, mem_limit=1024*1024*1024, verbosity=5)
    chk.measure(x)
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
    print(chk.sequence)
    print(x.requires_grad)
    y = chk(x).sum()
    y.backward()
    print(x.requires_grad, x.grad)
    for i in range(length):
        assert(any(p.grad is not None for p in seq[i].parameters()))


if __name__ == "__main__":
    print("-----------TWR-----------")
    test_with_rotor()
    print("-----------TWRI-----------")
    test_with_rotor_int()


## Not part of the test, just showing how it works
shape = [10, 10]
# Those params are used in TestFunc, but not visible as direct arguments of forward()
param = torch.randn(shape).requires_grad_()
param2 = torch.randn(shape).requires_grad_()


class TestFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, arg, bonus): 
        print("In Forward")
        # Two options: you can save the bonus Tensor or just a boolean
        # ctx.bonus = bonus
        ctx.is_bonus = bonus is not None

        # Simulate Fn
        ctx.save_for_backward(arg)
        with torch.no_grad():
            res = arg * param
            
        print(res.requires_grad)
        return res

    @staticmethod
    def backward(ctx, *args):
        print("In Backward")
        [inputs] = ctx.saved_tensors
        # Simulate Fe B
        inputs = inputs.detach().requires_grad_()
        with torch.enable_grad():
            res = inputs * param
        res.backward(*args)

        ## For simplicity, the gradient for the useless bonus is just itself...
        # return inputs.grad, ctx.bonus
        return inputs.grad, torch.ones(1) if ctx.is_bonus else None

# Highlight how things work with torch.autograd.Function:
# Adding a "bonus" unused Tensor that requires grad ensures that
# the backward phase is still called even if no input requires grad
def how_it_works():

    t = torch.randn(shape)
    print("NRG, With bonus")
    bonus = torch.ones([1]).requires_grad_()
    res = TestFunc.apply(t, bonus)
    print("res", res.requires_grad)
    res = res * param2
    print("res", res.requires_grad)
    res.sum().backward()

    t = torch.randn(shape)
    print("NRG, With bonus, deleted")
    bonus = torch.ones([1]).requires_grad_()
    res = TestFunc.apply(t, bonus)
    del bonus
    print("res", res.requires_grad)
    res = res * param2
    print("res", res.requires_grad)
    res.sum().backward()

    t = torch.randn(shape).requires_grad_()
    print("RG, Without bonus")
    res = TestFunc.apply(t, None)
    print("res", res.requires_grad)
    res = res * param2
    print("res", res.requires_grad)
    res.sum().backward()

    ## This case does not work: the TestFunc backward method is not
    ## called, because its output does not require_grad
    t = torch.randn(shape)
    print("NRG, Without bonus")
    res = TestFunc.apply(t, None)
    print("res", res.requires_grad)
    res = res * param2
    print("res", res.requires_grad)
    res.sum().backward()
