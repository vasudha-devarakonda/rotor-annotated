#!/usr/bin/python

import argparse
import sys
import os
import ast





class Architecture:
    def __init__(self, file_name):
        self.file_name = file_name
        try:
            self.file = open(file_name, "r")
        except FileNotFoundError:
            raise FileNotFoundError("The file "+file_name+" describing the architecture is not found.")
        self.sizes = []
        self.wd = []
        self.rd = []
        self.nblevels = -1
        self.read_file()
        if (len(self.sizes) != self.nblevels)or(len(self.wd) != self.nblevels)or(len(self.rd) != self.nblevels):
            raise ImportError("The level in the architecture file does not correspond to the number of line.")
        if (sorted(self.wd) != self.wd)or(sorted(self.rd) != self.rd):
            print("WARNING!!! This code is optimal only if the costs of writing and reading of the architecture are in the increasing order for the levels.")

    def __del__(self):
        try:
            self.file.close()
        except FileNotFoundError:
            pass
        except AttributeError:
            pass

    def __repr__(self):
        l = []
        for i in range(self.nblevels):
            l.append((self.sizes[i], self.wd[i], self.rd[i]))
        return l.__repr__()

    def read_file(self):
        for line in self.file:
            if line[0] == "#":
                continue
            line_list = [x for x in line.split()]
            if len(line_list) > 1 and self.nblevels < 0:
                raise SyntaxError("The first line of the architecture file should be a single integer.")
            if self.nblevels < 0:
                try:
                    self.nblevels = int(line_list[0])
                except:
                    raise SyntaxError("The first line of the architecture file should be an integer.")
                continue
            if len(line_list) != 3:
                raise SyntaxError("Every line of the architecture file should be a triplet (c: integer, wd: float, rd: float).")
            try:
                self.sizes.append(int(line_list[0]))
            except:
                raise SyntaxError("Every line of the architecture file should be a triplet (c: integer, wd: float, rd: float).")
            try:
                self.wd.append(int(line_list[1]))
            except:
                try:
                    self.wd.append(float(line_list[1]))
                except:
                    raise SyntaxError("Every line of the architecture file should be a triplet (c: integer, wd: float, rd: float).")
            try:
                self.rd.append(int(line_list[2]))
            except:
                try:
                    self.rd.append(float(line_list[2]))
                except:
                    raise SyntaxError("Every line of the architecture file should be a triplet (c: integer, wd: float, rd: float).")
        self.file.seek(0)


class Chain:
    def __init__(self, fw, bw, cw, cbw, ftmp, btmp, check=True):
        self.fweigth = fw
        self.bweigth = bw
        self.cweigth = cw
        self.cbweigth = cbw
        self.fwd_tmp = ftmp
        self.bwd_tmp = btmp
        self.length = len(fw)
        if check and not self.check_lengths():
            raise AttributeError("In Chain, input lists do not have consistent lengths")

    def check_lengths(self): 
        return ( (len(self.fweigth) == self.length) 
                 and (len(self.bweigth) == self.length+1)
                 and (len(self.cweigth) == self.length+1)
                 and (len(self.fwd_tmp) == self.length)
                 and (len(self.bwd_tmp) == self.length+1)
                 and (len(self.cbweigth) == self.length+1) )
        
    def __repr__(self):
        l = []
        for i in range(self.length):
            l.append((self.fweigth[i], self.bweigth[i], self.cweigth[i], self.cbweigth[i], self.fwd_tmp[i], self.bwd_tmp[i]))
        i = self.length
        l.append((None, self.bweigth[i], self.cweigth[i], self.cbweigth[i], None, self.bwd_tmp[i]))
        return l.__repr__()

class ChainStr(Chain):
    def __init__(self, str):
        try:
            input = ast.literal_eval(str)
            fw, bw, cw, cbw, ftmp, btmp = map(list, zip(*input))
            if fw[-1] is None:
                fw = fw[:-1]
            if ftmp[-1] is None: 
                ftmp = ftmp[:-1]
            super(ChainStr, self).__init__(fw, bw, cw, cbw, ftmp, btmp)
        except (TypeError, ValueError) as e:
            raise AttributeError("Error when creating ChainStr") from e
                
    
class ChainFile(Chain):
    def __init__(self, file_name):
        super(ChainFile, self).__init__([], [], [], [], check=False)
        self.file_name = file_name
        try:
            self.file = open(file_name, "r")
        except FileNotFoundError:
            raise FileNotFoundError("The file "+file_name+" describing the heterogeneous chain is not found.")
        print("file: ", file_name)
        self.read_file()
        
        if not self.check_lengths():
            raise ImportError("The length in the chain file does not correspond to the number of element per lines.")
        
        self.cbweigth.append(1)

    def __del__(self):
        try:
            self.file.close()
        except FileNotFoundError:
            pass
        except AttributeError:
            pass

        
    def read_file(self):
        real_line = 0
        for line in self.file:
            if line[0] == "#":
                continue
            line_list = [x for x in line.split()]
            if len(line_list) > 1 and real_line == 0:
                raise SyntaxError("The first line of the architecture file should be a single integer.")
            if real_line == 0:
                try:
                    self.length = int(line_list[0])
                except:
                    raise SyntaxError("The first line of the architecture file should be an integer.")
            if real_line == 1:
                for x in line_list:
                    try:
                        if int(x) < 0:
                            raise IndexError
                        self.fweigth.append(int(x))
                    except:
                        raise SyntaxError("The forward values should be positive integers")
            if real_line == 2:
                for x in line_list:
                    try:
                        if int(x) < 0:
                            raise IndexError
                        self.bweigth.append(int(x))
                    except:
                        raise SyntaxError("The backward values should be positive integers")
            if real_line == 3:
                for x in line_list:
                    try:
                        if int(x) < 0:
                            raise IndexError
                        self.cweigth.append(int(x))
                    except:
                        raise SyntaxError("The memory sizes should be positive integers")
            if real_line == 4:
                for x in line_list:
                    try:
                        if int(x) < 0:
                            raise IndexError
                        self.cbweigth.append(int(x))
                    except:
                        raise SyntaxError("The memory sizes should be positive integers")
            real_line += 1
        self.file.seek(0)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Compute HeteRevolve with the chain described in file_name and cm memory slots')
    parser.add_argument("-f", dest="chain", help = "File describing the heterogeneous chain", type = ChainFile, default=None)
    parser.add_argument("cm", help = "Memory Size", type = int)
    parser.add_argument("-s", help = "Specify the chain with a string", type = ChainStr, dest="chain", default=None)
    parser.add_argument("--concat", help="Level of concatenation between 0 and 2? (default: 0)", default = 0, type = int, metavar = "int", dest = "concat")
    parser.add_argument("--print_table", help = "Name of the file to print the table of results", default = None, type = str, metavar = "str", dest = "print_table")
    parser.add_argument("--force_python", help = "Force using the Python implementation even if C is available", action="store_true")
    parameters = parser.parse_args()
    parameters.isHeterogeneous = True
    if not parameters.chain:
        raise ValueError("A chain should be specified, either with '-f' or '-s' option")
    return parameters

def default_arguments(mem_limit, fw, bw, cw, cbw, ftmp, btmp):
    res = argparse.Namespace()
    res.chain = Chain(fw, bw + [0], cw, cbw, ftmp, btmp + [0])
    res.cm = mem_limit
    res.isHeterogeneous = True
    res.print_table = None
    res.concat = 0
    res.force_python = False
    return res

def default_arguments_chain(mem_limit, chain):
    res = argparse.Namespace()
    res.chain = chain
    res.cm = mem_limit
    res.isHeterogeneous = True
    res.print_table = None
    res.concat = 0
    res.force_python = False
    return res

