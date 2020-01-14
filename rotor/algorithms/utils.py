from .sequence import *


def argmin(iterable):
    return min(enumerate(iterable), key = lambda x: x[1])[0]


# Computes the sequential sequence
def chen_sqrt(length, segments = None):
    nb_segments = round(math.sqrt(length)) if segments is None else segments
    segment_size = length // nb_segments
    sequence = Sequence(Function("ChenSqrt", length, nb_segments))
    for start in range(0, segment_size * (nb_segments - 1), segment_size):
        end = start + segment_size - 1
        sequence.insert(ForwardCheck(start))
        for i in range(start + 1, end + 1):
            sequence.insert(ForwardNograd(i))
    for i in range(end + 1, length):
        sequence.insert(ForwardEnable(i))
    sequence.insert(Loss())
    for i in range(length-1, end, -1):
        sequence.insert(Backward(i))
    for start in range(segment_size * (nb_segments - 2), -1, -segment_size):
        end = start + segment_size - 1
        for i in range(start, end + 1):
            sequence.insert(ForwardEnable(i))
        for i in range(end, start-1, -1):
            sequence.insert(Backward(i))

    return sequence

# Computes a sequence which saves everything to memory, and thus does not recompute anything
def no_checkpoint(length):
    sequence = Sequence(Function("noCheckpoint", length, None))
    for i in range(length): 
        sequence.insert(ForwardEnable(i))
    sequence.insert(Loss())
    for i in range(length-1, -1, -1): 
        sequence.insert(Backward(i))
    return sequence
        

## Code to estimate the makespan and memory usage of a given sequence
def elementUsage(e, chain):
    t, i = e.split('_')
    i = int(i)
    if t == "x" or t == "y":
        return chain.cweigth[i]
    else:
        return chain.cbweigth[i]
        
def memUsage(storage, chain):
    if chain:
        return sum(elementUsage(e, chain) for e in storage)
    else:
        return len(storage)

# Simulates the execution of the sequence
# Returns the maximum memory usage 
def simulate_sequence(sequence, l, chain=None, display=True):
    if chain: l = chain.length
    mem = ["x_0"]
    maxUsage = memUsage(mem, chain)
    for op in sequence.list_operations():
        if display: print(memUsage(mem, chain), mem)
        if display: print(op)
        opType = type(op)
        if opType is Loss:
            input = "x_%d"%(l)
            inputalt = "xb_%d"%(l)
            if input not in mem and inputalt not in mem:
                raise ValueError("Before {op}: no {input} or {inputalt} in memory".format(op=op, input=input, inputalt=inputalt))
            mem.append("y_%d"%l)
        else:
            index = op.index
            if opType is ForwardEnable:
                input = "x_%d"%index
                inputalt = "xb_%d"%index
                if input not in mem and inputalt not in mem:
                    raise ValueError("Before {op}: no {input} or {inputalt} in memory".format(op=op, input=input, inputalt=inputalt))
                else:
                    mem.append("xb_%d"%(index+1))
                opUsage = memUsage(mem, chain) + chain.fwd_tmp[index]
            if opType is ForwardNograd:
                input = "x_%d" % index
                inputalt = "xb_%d" % index
                mem.append("x_%d"%(index+1))
                opUsage = memUsage(mem, chain) + chain.fwd_tmp[index]
                if input in mem:
                    mem.remove(input)
                elif inputalt in mem:
                    mem.remove(inputalt)
                else:
                    raise ValueError("Before {op}: no {input} or {inputalt} in memory".format(op=op, input=input, inputalt=inputalt))
            if opType is ForwardCheck:
                input = "x_%d" % index
                inputalt = "xb_%d" % index
                if input not in mem and inputalt not in mem:
                    raise ValueError("Before {op}: no {input} or {inputalt} in memory".format(op=op, input=input, inputalt=inputalt))
                else:
                    mem.append("x_%d"%(index+1))
                opUsage = memUsage(mem, chain) + chain.fwd_tmp[index]
            if opType is Backward:
                yinput = "y_%d"%(index+1)
                xbinput = "xb_%d"%(index+1)
                input = "x_%d"%index
                inputalt = "xb_%d"%index
                if yinput not in mem or xbinput not in mem:
                    raise ValueError("Before {op}: no {yinput} or {xbinput} in memory".format(op=op, yinput=yinput, xbinput=xbinput))
                else:
                    mem.append("y_%d"%index)
                    opUsage = memUsage(mem, chain) + chain.bwd_tmp[index]
                    if input in mem:
                        mem.remove(input)
                    elif inputalt not in mem:
                        raise ValueError("Before {op}: no {input} or {inputalt} in memory".format(op=op, input=input, inputalt=inputalt))
                    mem.remove(yinput)
                    mem.remove(xbinput)
        if display: print(opUsage, mem)
        maxUsage = max(maxUsage, opUsage)
    return maxUsage
    


