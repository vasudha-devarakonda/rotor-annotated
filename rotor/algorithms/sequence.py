#!/usr/bin/python

from . import parameters

import math
import os
import sys
import time
import numpy as np
import functools

class Operation:
    def shift(self, value):
        if type(self.index) is tuple:
            self.index = tuple(x + value for x in self.index)
        else: 
            self.index += value

class Forward(Operation):
    def __init__(self, index):
        self.index = index
        self.name = "F"
        
    def __repr__(self):
        return "{n}_{i}".format(n=self.name, i=self.index)

    def cost(self, params):
        if params.isHeterogeneous:
            return params.chain.fweigth[self.index]
        else:
            return params.uf

class ForwardEnable(Forward):
    def __init__(self, index):
        super().__init__(index)
        self.name = "Fe"

class ForwardNograd(Forward):
    def __init__(self, index):
        super().__init__(index)
        self.name = "Fn"

class ForwardCheck(Forward):
    def __init__(self, index):
        super().__init__(index)
        self.name = "CF"
        
class Forwards(Operation):
    def __init__(self, start, end):
        self.index = (start, end)

    def __repr__(self):
        return "F_{i}->{j}".format(i=self.index[0], j=self.index[1])

    def cost(self, params):
        if params.isHeterogeneous:
            return sum(params.chain.fweigth[self.index[0]:self.index[1]+1])
        else:
            return (self.index[1] - self.index[0] + 1) * params.uf

def isForward(op):
    return type(op) is Forward or type(op) is Forwards


class Backward(Operation): 
    def __init__(self, index):
        self.index = index

    def __repr__(self):
        return "B_{i}".format(i=self.index)

    def cost(self, params):
        if params.isHeterogeneous:
            return params.chain.bweigth[self.index]
        else:
            return params.ub
        
class Loss(Operation):
    def __init__(self):
        pass

    def __repr__(self):
        return "L"

    def cost(self, params):
        return 0
    
class MemoryAccess(Operation):
    def __init__(self, index):
        self.index = index
        
    def __repr__(self):
        return "{n}_{i}".format(n=self.name, i=self.index)

    def cost(self, params):
        return 0

class WriteMemory(MemoryAccess):
    def __init__(self, index):
        super().__init__(index)
        self.name = "WM"

class ReadMemory(MemoryAccess):
    def __init__(self, index):
        super().__init__(index)
        self.name = "RM"

class DiscardMemory(MemoryAccess):
    def __init__(self, index):
        super().__init__(index)
        self.name = "DM"
    
    
class Function:
    def __init__(self, name, *args):
        self.name = name
        self.args = args
        self.str_args = ','.join(str(v) for v in self.args)

    def __repr__(self):
        return "{n}({args})".format(n=self.name, args=self.str_args)


class Sequence:
    def __init__(self, function, params):
        self.sequence = [] #List of Operation and Sequence
        self.function = function #Description the function (name and parameters)
        self.makespan = 0 #Makespan to be updated
        self.params = params

    def __repr__(self):
        return repr(self.list_operations())

    def list_operations(self):
        l = []
        for x in self.sequence:
            if isinstance(x, Operation):
                l.append(x)
            else:
                assert isinstance(x, Sequence)
                l += x.list_operations()
        return l

    def insert(self, operation):
        self.sequence.append(operation)
        self.makespan += operation.cost(self.params)

    def remove(self, operation_index):
        self.makespan -= self.sequence[operation_index].cost(self.params)
        del self.sequence[operation_index]

    def insert_sequence(self, sequence):
        self.sequence.append(sequence)
        self.makespan += sequence.makespan

    def shift(self, value):
        for x in self.sequence:
            x.shift(value)
        return self

    def remove_useless_write(self):
        if self.sequence:
            if isinstance(self.sequence[0], WriteMemory):
                self.remove(0)
        return self

    def withoutSuffix(self): 
        ops = self.list_operations()
        endOfFirstPhase = [i for i in range(len(ops)) if type(ops[i]) is Loss][0]
        try:
            lastIndex = max(i for i in range(endOfFirstPhase) if not type(ops[i]) is ForwardEnable)
        except ValueError: 
            lastIndex = -1 
        if lastIndex == endOfFirstPhase - 1:
            return (self, None)
        chainLength = ops[endOfFirstPhase - 1].index   ## Some assumption here about the sequence (finishes with Forward_L
        startOfFwdEnableChain = ops[lastIndex+1].index ## And starts with B_L), but should be fine in practice
        result = Sequence(Function("Strip", self.function.name, *self.function.args, startOfFwdEnableChain), self.params)
        for i in range(lastIndex+1): 
            result.insert(ops[i])
        result.insert(Loss())
        for i in range(chainLength, startOfFwdEnableChain - 1, -1):
            position = endOfFirstPhase + 1 + (chainLength - i)
            assert type(ops[position]) is Backward
            assert ops[position].index == i
        for i in range(endOfFirstPhase + 1 + 1 + chainLength - startOfFwdEnableChain, len(ops)):
            result.insert(ops[i])
        return (result, startOfFwdEnableChain)
    


