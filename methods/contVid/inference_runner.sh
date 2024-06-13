#!/bin/bash

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21'

# /home/dhruv_sri/venvs/contVid/bin/python inference.py num_chunks=6 chunk_index=0
/home/dhruv_sri/venvs/contVid/bin/python inference.py num_chunks=6 chunk_index=1
/home/dhruv_sri/venvs/contVid/bin/python inference.py num_chunks=6 chunk_index=2
/home/dhruv_sri/venvs/contVid/bin/python inference.py num_chunks=6 chunk_index=3
/home/dhruv_sri/venvs/contVid/bin/python inference.py num_chunks=6 chunk_index=4
/home/dhruv_sri/venvs/contVid/bin/python inference.py num_chunks=6 chunk_index=5
