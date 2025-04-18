# TE-CP-Test

### Using Docker:
docker run --name TE-CP-Test --gpus all -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/zhengwang/workspace/TransformerEngine-CP:/workspace/TransformerEngine  nvcr.io/nvidia/pytorch:24.12-py3 


### Intall from source
```
cd TransformerEngine

pip uninstall transformer_engine

git submodule update --init --recursive # run it outside of the docker container
 
export NVTE_FRAMEWORK=pytorch   # Optionally set framework
pip install .                   # Build and install

export GLOBAL_TE_PATH="/workspace/TransformerEngine/build/lib.linux-x86_64-cpython-312/transformer_engine"
```

### Run per-seq CP forward local test:
```
./run_per_seq_test.sh
```