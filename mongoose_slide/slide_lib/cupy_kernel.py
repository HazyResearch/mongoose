import cupy as cp
from cupy.cuda import function
from cupy.cuda import device
from pynvrtc.compiler import Program
from collections import namedtuple

# CUDA Stream
Stream = namedtuple('Stream', ['ptr'])

class cupyKernel:
    def __init__(self, kernel, func_name):
        self.kernel = kernel
        self.title = func_name + ".cu"
        self.func_name = func_name
        self.compiled = False

    def get_compute_arch():
        return "compute_{0}".format(device.Device().compute_capability)

    def compile(self):
        # Create program
        program = Program(self.kernel, self.title)

        # Compile program
        arch = "-arch={0}".format(cupyKernel.get_compute_arch())
        ptx = program.compile([arch])

        # Load Program
        m = function.Module()
        m.load(bytes(ptx.encode()))

        # Get Function Pointer
        self.func = m.get_function(self.func_name)
        self.compiled = True

    def __call__(self, grid, block, args, strm):
        if not self.compiled:
            self.compile()

        # Run Function
        self.func(grid,
                  block,
                  args,
                  stream=Stream(ptr=strm))
