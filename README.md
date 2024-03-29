# Affogato

Affinity based segmentation algorithms:
- [Mutex Watershed](http://openaccess.thecvf.com/content_ECCV_2018/html/Steffen_Wolf_The_Mutex_Watershed_ECCV_2018_paper.html)
- [Semantic Mutex Watershed](https://link.springer.com/chapter/10.1007/978-3-030-58539-6_13)

## Installation 

**From conda:**

```
conda install -c conda-forge affogato
```

**From source:**

 - create conda environment with the necessary dependencies:
 - `conda create -n mws -c conda-forge xtensor-python`
 - activate the env:
 - `source activate mws`
 - clone this repository, enter it and build the repository via cmake:
 - `git clone https://github.com/constantinpape/affogato`
 - `cd affogato`
 - `mkdir -p build && cd build`
 - `cmake -DCMAKE_PREFIX_PATH=/path/to/conda/env ..`
 - `make`

## How can I run Mutex Watershed?

You can follow along with this [colab tutorial](https://github.com/constantinpape/affogato/blob/master/example/MutexWatershed.ipynb)
The example data is available [here](https://oc.embl.de/index.php/s/sXJzYVK0xEgowOz).

The demo for the interactive mutex watershed can be run with [example/interactive/interactive_napari.py](https://github.com/constantinpape/affogato/blob/master/example/interactive/interactive_napari.py). Exapme data is available [here](https://oc.embl.de/index.php/s/yoBbHp7n2PAMrXm/download?path=%2Fdata&files=data.h5).
