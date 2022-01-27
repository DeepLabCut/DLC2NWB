import os
import pandas as pd
from dlc2nwb import utils


FILEPATH = "examples/m3v1mp4DLC_resnet50_openfieldAug20shuffle1_30000.h5"
CONFIGPATH = "examples/config.yaml"


def test_round_trip_conversion():
    df_ref = pd.read_hdf(FILEPATH)
    nwbfile = utils.convert_h5_to_nwb(
        CONFIGPATH,
        FILEPATH,
    )[0]
    df = utils.convert_nwb_to_h5(nwbfile).droplevel("individuals", axis=1)
    pd.testing.assert_frame_equal(df, df_ref)


def test_multi_animal_round_trip_conversion(tmp_path):
    dfs = []
    n_animals = 3
    for i in range(1, n_animals + 1):
        temp = utils._ensure_individuals_in_header(
            pd.read_hdf(FILEPATH), f"animal_{i}",
        )
        dfs.append(temp)
    df = pd.concat(dfs, axis=1)
    path_fake_df = str(tmp_path / os.path.split(FILEPATH)[1])
    df.to_hdf(path_fake_df, key="fake")
    nwbfiles = utils.convert_h5_to_nwb(CONFIGPATH, path_fake_df)
    assert len(nwbfiles) == n_animals