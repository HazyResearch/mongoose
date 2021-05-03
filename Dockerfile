FROM nvcr.io/nvidian/pytorch:20.03-py3
RUN pip install torch torchvision
RUN pip install scipy
RUN pip install --pre cupy-cuda102
RUN pip install pynvrtc
RUN pip install Cython
RUN pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN pip install local-attention product-key-memory axial-positional-embedding==0.1.0
RUN cd lsh_lib && python setup.py install