import torch
import numpy as np
from .utils import *
from . import timing
from . import memory
torch.manual_seed(seed=42)
## Allows a module that inherits from Sequential
## to specify that it overrides the forward() method
## and thus should be considered as a single block
def is_really_sequential(m):
    b = isinstance(m, torch.nn.Sequential)
    if b:
        try:
            if m.not_really_sequential: return False
            return True
        except AttributeError:
            return True
    else: return False

## Extracts a sequence of modules from a given module, recursively
## Returns a list of (name, module) where running the modules in sequence
## is equivalent to running the input module
## Names are formatted with the same format as onnx does.
def extract_children_from_sequential(m, name="", prefix=""):
    className = type(m).__name__
    fullName = prefix + ("-" if prefix else "")
    fullName += className
    if name: fullName = name + "-" +fullName
    if not (is_really_sequential(m)):
        return [ (fullName, m) ]
    children = (extract_children_from_sequential(c, name=n, prefix=fullName) for (n, c) in m.named_children())
    children = sum(children, [])
    return children


## We assume that ReLUs are made in place, as well as concats
## aten::t nodes are also a question mark for me, they seem used to transpose weights in Linear modules
def node_is_relu_or_concat(node):
    kind = node.kind()
    ns, opName = kind.split('::')
    return (opName.startswith('Relu') or opName.startswith('Concat')
            or opName == "t" or opName == "threshold_" or opName == "relu_" or opName == "cat")

## The weight of a node is the total size of all its Tensor outputs
def node_weight(node):
    return sum(4 * np.prod(o.type().sizes()) for o in node.outputs() if o.isTensor())##  and o.type().scalarType() == 'Float')
    ## For some reason, MaxPool2d has two outputs : the real one, and a Tensor of Longs of the same size. But this output does not appear anywhere...
    ## Hmmm. Probably it is kept to be able to do the backward -- that's why. 
            
def sum_all_weights(nodes, name=None):
    # heavy_nodes = [ (n, node_weight(n)) for n in nodes if not node_is_relu_or_concat(n) and node_weight(n) > 0 ]
    # if name:
    #     print("Heavy nodes for ", name)
    #     for n, w in heavy_nodes:
    #         print("   {:>10} {}".format(w, n))
    # return sum(w for _, w in heavy_nodes)
    return sum(node_weight(n) for n in nodes if not node_is_relu_or_concat(n))

## The output weight of a block is the output of its last node.
## According to csrc/jit/ir.h:359, nodes are ordered in topological order
def output_weight(block, name):
    # for n in reversed(block):
    #     if n.scopeName() == name:
    if block: 
        return node_weight(block[-1])
    else: return 0

def compute_all_weights_of_graph(graph, nameList, warning=True):
    blocks = { n : [] for n in nameList }
    for node in graph.nodes():
        scope = node.scopeName()
        blockNames = [ n for n in nameList if scope.startswith(n) ]
        if blockNames:
            assert(len(blockNames) == 1)
            blocks[blockNames[0]].append(node)
        else:
            if warning: 
                print("Warning: node", node, "does not fit in any block")
            
    ## Compute block weights
    weightList = [ sum_all_weights(blocks[name], name=name) for name in nameList ]

    ## Compute output weights
    outputWeightsList = [ output_weight(blocks[name], name) for name in nameList ]

    return (weightList, outputWeightsList)
        
def compute_all_weights(model, x, nameList, warning=True, verbose=False):
    ## with torch.onnx.set_training(model, False):
    trace, _ = torch.jit.get_trace_graph(model, args=(x,))
    if verbose: 
        print(trace)
        print(trace.graph())
    return compute_all_weights_of_graph(trace.graph(), nameList, warning)




def tensorMsize(t):
    if t is None:
        return 0  # ignore None values
    elif isinstance(t, torch.Tensor):
        return t.element_size() * np.prod(t.shape)
    elif isinstance(t, (list, tuple)):
        return sum(tensorMsize(u) for u in t)  # recursively sum
    else:
        return 0  # ignore other types

def print_tensor_info(x):
    if isinstance(x, torch.Tensor):
        print("Tensor:", x.shape, "dtype:", x.dtype, "requires_grad:", x.requires_grad)
    elif isinstance(x, (list, tuple)):
        print(f"{type(x)} of length {len(x)}")
        for i, t in enumerate(x):
            print(f" element {i}:")
            print_tensor_info(t)
    else:
        print(f"Unknown type: {type(x)}")




def make_gradient_for(x):
    """
    Recursively create grad_tensors matching x.
    Only adds a tensor if it requires grad, else None.
    Supports Tensor, list, or tuple.
    """
    if isinstance(x, torch.Tensor):
        if x.is_floating_point() and x.requires_grad:
            return torch.ones_like(x)
    elif isinstance(x, list):
        return [make_gradient_for(t) for t in x]
    elif isinstance(x, tuple):
        return tuple(make_gradient_for(t) for t in x)
    else:
        raise RuntimeError(f"Unsupported type in make_gradient_for: {type(x).__name__}")
def backward_safe(xbar):
    """
    Recursively flatten xbar and call backward only on tensors
    that are floating-point and require grad.
    """
    def flatten_and_keep_grads(x):
        if isinstance(x, torch.Tensor):
            return [x] if x.is_floating_point() and x.requires_grad else []
        elif isinstance(x, (list, tuple)):
            result = []
            for t in x:
                result.extend(flatten_and_keep_grads(t))
            return result
        else:
            return []

    xbar_grad = flatten_and_keep_grads(xbar)
    if xbar_grad:
        args = [torch.ones_like(t) for t in xbar_grad]
        torch.autograd.backward(xbar_grad, grad_tensors=args)
    else:
        print("Warning: no tensors require grad, skipping backward.")
## Measure execution time and memory usage
## just by running each block in sequence
def measure_everything(named_modules, input, min_duration=30):
    # Zero out grads
    for (_, m) in named_modules:
        for p in m.parameters():
            if p.requires_grad:
                p.grad = torch.zeros_like(p)


    x = detach_variable(input, force_required_grad=False)
    result_xbar = [tensorMsize(input)]
    result_fwdTime = []
    result_bwdTime = []
    result_x = [tensorMsize(input)]
    result_tmpFwd = []
    result_tmpBwd = []

    with torch.enable_grad():
        y = named_modules[0][1](x)

        # Filter tensors that can require gradients
        if isinstance(y, (list, tuple)):
            y_grad = [t for t in y if isinstance(t, torch.Tensor) and t.is_floating_point() and t.requires_grad]
        else:
            y_grad = [y] if y.is_floating_point() and y.requires_grad else []

        if y_grad:
            torch.autograd.backward(y_grad, grad_tensors=make_gradient_for(y_grad))
        else:
            print("Warning: no floating-point tensors require grad, skipping backward.")

    del y

    device = get_device(input)
    timer = timing.make_timer(device)
    memUsage = memory.MeasureMemory(device)

    def perform_measure(func, prologue = None):
        def complete_func():
            if prologue: prologue()
            return func()
        # print("ghere its is called")
        _, usage, maxUsage = memUsage.measure(complete_func)
        duration = timer.measure_median(func)
        if duration < min_duration: 
            number_repetitions = 1 + int(min_duration // duration)
            duration = timer.measure_median(func, iterations = number_repetitions)
        return duration, int(usage), int(maxUsage)

    
    for name, module in named_modules:
        x = detach_variable(x)

        def forwardOp():
            nonlocal xbar
            xbar = None
            with torch.enable_grad(): 
                xbar = module(x)
                # print_tensor_info(xbar)

        fwd_duration, usage, maxUsageFwd = perform_measure(forwardOp)

        xbar_requires_grad = does_require_grad(xbar)
        result_x.append(tensorMsize(xbar))
        xbarSize = max(usage, tensorMsize(xbar))
        result_xbar.append(xbarSize)
        result_fwdTime.append(fwd_duration)

        # with torch.enable_grad(): 
        #     summary = xbar.sum()
        # xbar = None

        def backwardOp():
            remove_gradients(x)
            backward_safe(xbar)
            # summary.backward()

        # memDisplay.printCurrentState("Measuring Bwd" + name)
        # Measure backward only once, because precise timings are not needed since all 
        # backwards are only performed once, no optimization available here
        # Plus running bwd several times would require retain_graph=True, and 
        # it might modify the memory usage
        bwd_duration, _, maxUsageBwd = memUsage.measure(lambda: timer.measure(backwardOp))
        result_bwdTime.append(bwd_duration)

        with torch.no_grad(): 
            xbar = module(x)
        
        result_tmpFwd.append(int(maxUsageFwd) - xbarSize) # input was already in memory when starting experiment
        result_tmpBwd.append(int(maxUsageBwd) - (tensorMsize(x) + tensorMsize(xbar))) # input x_i and xb_i+1 were in memory, y_i+1 and y_i were added.

        x = detach_variable(xbar, force_required_grad=xbar_requires_grad)
        del xbar

    for (_, m) in named_modules: 
        m.zero_grad()
            
    return result_fwdTime, result_bwdTime, result_xbar, result_x, result_tmpFwd, result_tmpBwd
        # args = torch.ones_like(xbar)
        # elapsed = time.perf_counter()
        # usage = torch.cuda.memory_allocated()
        # xbar.backward(args)
        # usage = torch.cuda.memory_allocated() - usage
        # elapsed = time.perf_counter() - elapsed
        # grad = x.grad
        # # print("Stats of module  backward:", name, elapsed, usage)
        # # print("Size of gradient, theory :", name, tensorMsize(grad), tensorMsize(grad) - (xbarSize - tensorMsize(xbar)))
        # del args
        # del xbar

        # xsize = 4*np.prod(x.shape)
        # elapsed = time.perf_counter()
        # usage = torch.cuda.memory_allocated()
        # with torch.no_grad():
        #     x = module(x)
        # usage = torch.cuda.memory_allocated() - usage
        # elapsed = time.perf_counter() - elapsed
        # print("Stats of module   no grad:", name, elapsed, usage)
        # print("Size of xout, x, diff    :", name, tensorMsize(x), xsize, tensorMsize(x) - xsize)
        # print("")
