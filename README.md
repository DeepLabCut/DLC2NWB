# Welcome to the DeepLabCut 2 Neurodata Without Borders Repo

Here we provide utilities to convert DeepLabCut (DLC) output to/from Neurodata Without Borders (NWB) format. This repository also elaborates a way for how pose estimation data should be represented in NWB. 

Currently this is in alpha mode!

Specifically, this package allows you to convert [DLC's predictions on videos (*.h5 files)](https://github.com/DeepLabCut/DLC2NWB/blob/main/examples/README.md) into NWB format. This is best explained with an example

# Example use:

```
from dlc2nwb.utils import convert_h5_to_nwb, convert_nwb_to_h5

# Convert DLC -> NWB:
nwbfile = convert_h5_to_nwb(
    'examples/config.yaml',
    'examples/m3v1mp4DLC_resnet50_openfieldAug20shuffle1_30000.h5',
)

# Convert NWB -> DLC
df = convert_nwb_to_h5(nwbfile[0])
```

Example data to run the code is provided in the folder [examples](/examples). This data is based on a DLC project you can find on [Zenodo](https://zenodo.org/record/4008504#.YWhD7NOA4-R) and that was originally presented in [Mathis et al., Nat. Neuro](https://www.nature.com/articles/s41593-018-0209-y) as well as [Mathis et al., Neuron](https://www.sciencedirect.com/science/article/pii/S0896627320307170?via%3Dihub).

To limit space, the folder only contains the project file `config.yaml` and DLC predictions for an example video called `m3v1mp4.mp4`, which are stored in `*.h5` format. The video is available, [here](https://github.com/DeepLabCut/DeepLabCut/tree/master/examples/openfield-Pranav-2018-10-30/videos).

# NWB pose ontology

The standard is presented [here](https://github.com/rly/ndx-pose). Our code is based on this NWB class (PoseEstimationSeries, PoseEstimation).


# Installation:

- install [deeplabcut](https://github.com/DeepLabCut/DeepLabCut)
- install [ndx-pose](https://github.com/rly/ndx-pose)
- install this repository

Note: this will be simplified, and made available via pypi (see below).

## To-do:

Once we are happy with this converter, we will integrate the functionality in DLC. For this purpose:
- put this code on pypi
- put [ndx-pose](https://github.com/rly/ndx-pose) on pypi
- make it a dependency of DeepLabCut
- add an argument to [`deeplabcut.predict_videos(....,export2NWB=True)`](https://github.com/DeepLabCut/DeepLabCut/blob/master/deeplabcut/pose_estimation_tensorflow/predict_videos.py#L42), that will *also* export the video predictions in the NWB format (on top of native DLC).


# Funding and contributions:

We gratefully acknowledge the generous support from the [Kavli Foundation](https://kavlifoundation.org/) via a [Kavli Neurodata Without Borders Seed Grants
](https://www.nwb.org/nwb-seed-grants/).

We furthermore acknowledge feedback and discussions with [Ben Dichter, Ryan Li and Olviver Ruebel](https://www.nwb.org/team/).
