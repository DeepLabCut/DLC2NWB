# Welcome to the DeepLabCut 2 Neurodata Without Borders Repo

Here we provide utilities to convert [DeepLabCut (DLC)](https://github.com/DeepLabCut/DeepLabCut) output to/from [Neurodata Without Borders (NWB) format](https://www.nwb.org/nwb-neurophysiology/). This repository also elaborates a way for how pose estimation data should be represented in NWB.

Specifically, this package allows you to convert DLC's predictions on videos (*.h5 files) into NWB format. This is best explained with an example (see below).

# NWB pose ontology

The standard is presented [here](https://github.com/rly/ndx-pose). Our code is based on this NWB extension (PoseEstimationSeries, PoseEstimation) that was developed with [Ben Dichter, Ryan Ly and Oliver Ruebel](https://www.nwb.org/team/).

# Installation:

Simply do (it only depends on `ndx-pose` and `deeplabcut`):

`pip install dlc2nwb`

# Example within DeepLabCut

DeepLabCut's h5 data files can be readily converted to NWB format either via the GUI from the `Analyze Videos` tab or programmatically, as follows:

```python
import deeplabcut

deeplabcut.analyze_videos_converth5_to_nwb(config_path, video_folder)
```
Note that DLC does not strictly depend on dlc2nwb just yet; if attempting to convert to NWB, a user would be asked to run `pip install dlc2nwb`.

# Example use case of this package (directly):

Here is an example for converting DLC data to NWB format (and back). Notice you can also export your data directly from DeepLabCut. This will be further documented, and is currently in this [branch](https://github.com/DeepLabCut/DeepLabCut/tree/nwb)!

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

Example data to run the code is provided in the folder [examples](/examples). The data is based on a DLC project you can find on [Zenodo](https://zenodo.org/record/4008504#.YWhD7NOA4-R) and that was originally presented in [Mathis et al., Nat. Neuro](https://www.nature.com/articles/s41593-018-0209-y) as well as [Mathis et al., Neuron](https://www.sciencedirect.com/science/article/pii/S0896627320307170?via%3Dihub). To limit space, the folder only contains the project file `config.yaml` and DLC predictions for an example video called `m3v1mp4.mp4`, which are stored in `*.h5` format. The video is available, [here](https://github.com/DeepLabCut/DeepLabCut/tree/master/examples/openfield-Pranav-2018-10-30/videos).


# Funding and contributions:

We gratefully acknowledge the generous support from the [Kavli Foundation](https://kavlifoundation.org/) via a [Kavli Neurodata Without Borders Seed Grants
](https://www.nwb.org/nwb-seed-grants/).

We also acknowledge feedback, and our collaboration with [Ben Dichter, Ryan Ly and Oliver Ruebel](https://www.nwb.org/team/).
