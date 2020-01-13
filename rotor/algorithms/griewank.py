#!/usr/bin/python

from . import parameters
from .sequence import *
from .utils import argmin
import argparse

def convert_griewank_to_rotor(seq, params):
    res = Sequence(Function("Conversion", seq.function.name, *seq.function.args), params)
    ops = seq.list_operations()

    idx = 0
    nb_ops = len(ops)
    pleaseSaveNextFwd = True
    while idx < nb_ops:
        op = ops[idx]
        # print(idx, op)
        if isForward(op):
            index = op.index
            if type(index) is int: index = (index, index)
            for i in range(index[0], index[1]+1):
                if pleaseSaveNextFwd:
                    res.insert(ForwardCheck(i))
                    pleaseSaveNextFwd = False
                else: res.insert(ForwardNograd(i))
        elif type(op) is WriteMemory:
            pleaseSaveNextFwd = True
        elif type(op) is DiscardMemory:
            pass
        elif type(op) is ReadMemory:
            if type(ops[idx+1]) is Backward:
                assert op.index == ops[idx+1].index
                res.insert(ForwardEnable(op.index))
                res.insert(Backward(op.index))
                idx += 1
            else:
                assert isForward(ops[idx+1])
                pleaseSaveNextFwd = True
        elif type(op) is Backward:
            if op.index == params.chain.length:
                res.insert(Loss())
            else:
                res.insert(ForwardEnable(op.index))
                res.insert(Backward(op.index))
        else: 
            raise AttributeError("Unknown operation type {t} {op}".format(t=type(op), op=op))
        idx += 1

    return res

def convert_griewank_to_rotor_xbar(seq, params):
    res = Sequence(Function("Conversion Xbar", seq.function.name, *seq.function.args), params)
    ops = seq.list_operations()

    ## lastValueSaved = {i: None for i in range(0, params.chain.length + 1)}
    idx = 0
    nb_ops = len(ops)
    pleaseSaveNextFwd = True
    nextFwdEnable = False
    doNotDoNextFwd = False
    while idx < nb_ops:
        op = ops[idx]
        # print(idx, op)
        if isForward(op):
            index = op.index
            if type(index) is int: index = (index, index)
            for i in range(index[0], index[1]+1):
                if doNotDoNextFwd:
                    doNotDoNextFwd = False
                elif nextFwdEnable: 
                    res.insert(ForwardEnable(i))
                    pleaseSaveNextFwd = True
                    nextFwdEnable = False
                elif pleaseSaveNextFwd:
                    res.insert(ForwardCheck(i))
                    pleaseSaveNextFwd = False
                else: res.insert(ForwardNograd(i))
            pleaseSaveNextFwd = False
        elif type(op) is WriteMemory:
            nextFwdEnable = True
        elif type(op) is DiscardMemory:
            pass
        elif type(op) is ReadMemory:
            if type(ops[idx+1]) is Backward:
                assert op.index == ops[idx+1].index
                res.insert(Backward(op.index))
                idx += 1
            else:
                assert isForward(ops[idx+1])
                doNotDoNextFwd = True
                pleaseSaveNextFwd = True
        elif type(op) is Backward:
            if op.index == params.chain.length:
                res.insert(Loss())
            else:
                res.insert(ForwardEnable(op.index))
                res.insert(Backward(op.index))
        else: 
            raise AttributeError("Unknown operation {t} {op}".format(t=type(op), op=op))
        idx += 1

    return res


def get_opt_0_table(lmax, mmax, params, file_name=None):
    """ Return the Opt_0 table
        for every Opt_0[l][m] with l = 0...lmax and m = 0...mmax
        The computation uses a dynamic program"""
    if file_name != None:
        f = open(file_name, "w")
        f.write("#size max = %d\n" % lmax)
        f.write("#mem max = %d\n" % mmax)
        f.write("#size\tmem\topt_0\n")
        f.flush()
    uf = params.uf
    ub = params.ub
   
    # Build table
    ## print(mmax,lmax)
    opt = [[float("inf")] * (mmax + 1) for _ in range(lmax + 1)]
    # Initialize borders of the table
    for m in range(mmax + 1):
        opt[0][m] = ub
        if file_name != None:
            f.write("%d\t%d\t%.2f\n" % (0, m, opt[0][m]))
    for m in range(1, mmax + 1):
        opt[1][m] = uf + 2 * ub
        if file_name != None:
           f.write("%d\t%d\t%.2f\n" % (1, m, opt[1][m]))
    for l in range(1, lmax + 1):
        opt[l][1] = (l+1) * ub + l * (l + 1) / 2 * uf
        if file_name != None:
            f.write("%d\t%d\t%.2f\n" % (l, 1, opt[l][1]))
    if file_name != None:
        f.flush()
    # Compute everything
    for m in range(2, mmax + 1):
        for l in range(2, lmax + 1):
            opt[l][m] = min([j * uf + opt[l - j][m-1] + opt[j-1][m] for j in range(1, l)])
            if file_name != None:
                f.write("%d\t%d\t%.2f\n" % (l, m, opt[l][m]))
        if file_name != None:
            f.flush()
    return opt


def griewank_rec(l, cmem, params, opt_0 = None):
    """ l : number of forward step to execute in the AC graph
        cmem : number of available memory slots
        Return the optimal sequence of makespan Opt_0(l, cmem)"""
    if cmem == 0:
        raise ValueError("Can not process a chain without memory")
    if opt_0 == None:
        opt_0 = get_opt_0_table(l, cmem, params)
    sequence = Sequence(Function("Griewank", l, cmem), params)
    if l == 0:
        sequence.insert(Backward(0))
        sequence.insert(DiscardMemory(0))
        return sequence
    elif l == 1:
        sequence.insert(WriteMemory(0))
        sequence.insert(Forward(0))
        sequence.insert(Backward(1))
        sequence.insert(ReadMemory(0))
        sequence.insert(Backward(0))
        sequence.insert(DiscardMemory(0))
        return sequence
    elif cmem == 1:
        sequence.insert(WriteMemory(0))
        for index in range(l - 1, -1, -1):
            if index != l - 1:
                sequence.insert(ReadMemory(0))
            sequence.insert(Forwards(0,index))
            sequence.insert(Backward(index + 1))
        sequence.insert(ReadMemory(0))
        sequence.insert(Backward(0))
        sequence.insert(DiscardMemory(0))
        return sequence
    list_mem = [j*params.uf + opt_0[l-j][cmem-1] + opt_0[j-1][cmem] for j in range(1,l)]
    jmin = 1 + argmin(list_mem)
    sequence.insert(WriteMemory(0))
    sequence.insert(Forwards(0, jmin - 1))
    sequence.insert_sequence(griewank_rec(l - jmin, cmem - 1, params, opt_0 = opt_0).shift(jmin))
    sequence.insert(ReadMemory(0))
    sequence.insert_sequence(griewank_rec(jmin-1, cmem, params, opt_0 = opt_0).remove_useless_write())
    return sequence


def griewank(params, useXbar = False, showInputs = False):
    max_peak = max(max(params.chain.fwd_tmp), max(params.chain.bwd_tmp))
    available_mem = params.cm - max_peak
    if useXbar:
        sizes = sorted(params.chain.cbweigth, reverse = True)
    else: 
        sizes = sorted(params.chain.cweigth, reverse = True)
    nb_slots = 0
    sum = 0
    while sum < available_mem and nb_slots < len(sizes):
        sum += sizes[nb_slots]
        nb_slots += 1

    hom_params = argparse.Namespace()
    hom_params.l = params.chain.length
    hom_params.cm = nb_slots
    hom_params.ub = 1
    hom_params.uf = 1
    hom_params.concat = 0
    hom_params.print = None
    hom_params.isHeterogeneous = False
    if showInputs: print("Hom Inputs: {l} {cm}".format(l=hom_params.l, cm=hom_params.cm), file=sys.stderr)
        
    seq = griewank_rec(hom_params.l, hom_params.cm, hom_params)
    if useXbar:
        converted = convert_griewank_to_rotor_xbar(seq, params)
    else: 
        converted = convert_griewank_to_rotor(seq, params)
    return converted


