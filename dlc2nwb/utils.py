import datetime
import os
import numpy as np
import pandas as pd
import warnings
from deeplabcut import __version__
from deeplabcut.utils import auxiliaryfunctions
from hdmf.build.warnings import DtypeConversionWarning
from pynwb import NWBFile, NWBHDF5IO
from ndx_pose import PoseEstimationSeries, PoseEstimation


def convert_h5_to_nwb(config, h5file, individual_name="ind1"):
    """
    Convert a DeepLabCut (DLC) video prediction, h5 data file to Neurodata Without Borders (NWB). Also
    takes project config, to store relevant metadata.

    Parameters
    ----------
    config : str
        Path to a project config.yaml file

    h5file : str
        Path to a h5 data file

    individual_name: str
        Name of the subject (whose pose is predicted) for single-animal  DLC project.
        For multi-animal projects, the names from the DLC project will be used directly.

    TODO: allow one to overwrite those names, with a mapping?

    Returns
    -------
    str
        Path to the newly created NWB data file. By default the file is stored in the same folder as the h5file.

    """
    cfg = auxiliaryfunctions.read_config(config)

    vidname, scorer = os.path.split(h5file)[-1].split("DLC")
    scorer = "DLC" + scorer.rsplit("_", 1)[0]
    video = None
    for video_path, params in cfg["video_sets"].items():
        if vidname in video_path:
            video = video_path, params["crop"]
            break
    if video is None:
        warnings.warn(f"The video file corresponding to {h5file} could not be found...")
        video = "fake_path", "0, 0, 0, 0"

    df = pd.read_hdf(h5file)
    if "individuals" not in df.columns.names:
        # Single animal project -> add individual row to the header
        # of single animal projects. The animal/individual name can be specified.
        temp = pd.concat({individual_name: df}, names=["individuals"], axis=1)
        df = temp.reorder_levels(["scorer", "individuals", "bodyparts", "coords"], axis=1)

    output_paths = []
    for animal, df_ in df.groupby(level="individuals", axis=1):
        pose_estimation_series = []
        for kpt, xyp in df_.groupby(level="bodyparts", axis=1, sort=False):
            data = xyp.to_numpy()
            timestamps = df.index.tolist()
            pes = PoseEstimationSeries(
                name=f"{animal}_{kpt}",
                description=f"Keypoint {kpt} from individual {animal}.",
                data=data[:, :2],
                unit="pixels",
                reference_frame="(0,0) corresponds to the bottom left corner of the video.",
                timestamps=timestamps,
                confidence=data[:, 2],
                confidence_definition="Softmax output of the deep neural network.",
            )
            pose_estimation_series.append(pes)

        pe = PoseEstimation(
            pose_estimation_series=pose_estimation_series,
            description="2D keypoint coordinates estimated using DeepLabCut.",
            original_videos=[video[0]],
            dimensions=[list(map(int, video[1].split(",")))[1::2]],
            scorer=scorer,
            source_software="DeepLabCut",
            source_software_version=__version__,
            nodes=[pes.name for pes in pose_estimation_series],
        )

        nwbfile = NWBFile(
            session_description=cfg["Task"],
            experimenter=cfg["scorer"],
            identifier=scorer,
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
        )

        # TODO Store the test_pose_config as well?
        behavior_pm = nwbfile.create_processing_module(
            name="behavior",
            description="processed behavioral data"
        )
        behavior_pm.add(pe)
        output_path = h5file.replace(".h5", f"_{animal}.nwb")
        with warnings.catch_warnings(), NWBHDF5IO(output_path, mode="w") as io:
            warnings.filterwarnings("ignore", category=DtypeConversionWarning)
            io.write(nwbfile)
        output_paths.append(output_path)

    return output_paths


def convert_nwb_to_h5(nwbfile,return_df=True):
    """
    Convert a NWB data file back to DeepLabCut's h5 data format.

    Parameters
    ----------
    nwbfile : str
        Path to the newly created NWB data file

    Returns
    -------
    df : pandas.array
        Pandas multi-column array containing predictions in DLC format.

    """
    with NWBHDF5IO(nwbfile, mode="r", load_namespaces=True) as io:
        read_nwbfile = io.read()
        read_pe = read_nwbfile.processing["behavior"]["PoseEstimation"]
        scorer = read_pe.scorer or "scorer"
        dfs = []
        for node in read_pe.nodes:
            pes = read_pe.pose_estimation_series[node]
            animal, kpt = node.split("_")
            array = np.c_[pes.data, pes.confidence]
            cols = pd.MultiIndex.from_product(
                [[scorer], [animal], [kpt], ["x", "y", "likelihood"]],
            )
            dfs.append(pd.DataFrame(array, np.asarray(pes.timestamps).astype(int), cols))

    return pd.concat(dfs, axis=1)
