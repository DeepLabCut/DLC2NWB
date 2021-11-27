from dlc2nwb.utils import convert_h5_to_nwb, convert_nwb_to_h5


nwbfile = convert_h5_to_nwb(
    "examples/config.yaml",
    "examples/m3v1mp4DLC_resnet50_openfieldAug20shuffle1_30000.h5",
)

df = convert_nwb_to_h5(nwbfile[0])
