# How do DLC video predictions look?

The labels are stored in a multi-index Pandas array, which contains the name of the network, body part name, (x, y) label position in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient HDF format, and the filename is always a combination of the video name and the network name, which contains meta data with regard to the model (e.g. for the example below, the network is named:  DLC_resnet50_openfieldAug20shuffle1_30000).

Here are the first 5 frames for the example:
```
import pandas as pd
pd.read_hdf('m3v1mp4DLC_resnet50_openfieldAug20shuffle1_30000.h5').head()

scorer    DLC_resnet50_openfieldAug20shuffle1_30000             ...                       
bodyparts                                     snout             ...    tailbase           
coords                                            x          y  ...           y likelihood
0                                         77.297806  88.875580  ...  184.270721   0.998464
1                                         74.802086  86.629105  ...  181.814163   0.999529
2                                         75.272606  82.104889  ...  178.319321   0.999563
3                                         75.405212  79.425911  ...  174.022385   0.999148
4                                         73.988655  77.614182  ...  170.291306   0.999464

[5 rows x 12 columns]

```
