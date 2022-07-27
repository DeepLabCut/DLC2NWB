import cv2
import datetime
import os
import numpy as np
import pandas as pd
import pickle
import warnings
from deeplabcut import __version__
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.auxfun_videos import VideoReader
from hdmf.build.warnings import DtypeConversionWarning
from pynwb import NWBFile, NWBHDF5IO
from ndx_pose import PoseEstimationSeries, PoseEstimation


def get_movie_timestamps(movie_file, VARIABILITYBOUND=1000):
    """
    Return numpy array of the timestamps for a video.

    Parameters
    ----------
    movie_file : str
        Path to movie_file
    """
    # TODO: consider moving this to DLC, and actually extract alongside video analysis!

    reader = VideoReader(movie_file)
    timestamps = []
    for _ in range(len(reader)):
        _ = reader.read_frame()
        timestamps.append(reader.video.get(cv2.CAP_PROP_POS_MSEC))

    timestamps = np.array(timestamps) / 1000  # Convert to seconds

    if np.nanvar(np.diff(timestamps)) < 1.0 / reader.fps * 1.0 / VARIABILITYBOUND:
        warnings.warn(
            "Variability of timestamps suspiciously small. See: https://github.com/DeepLabCut/DLC2NWB/issues/1"
        )

    return timestamps


def _ensure_individuals_in_header(df, dummy_name):
    if "individuals" not in df.columns.names:
        # Single animal project -> add individual row to
        # the header of single animal projects.
        temp = pd.concat({dummy_name: df}, names=["individuals"], axis=1)
        df = temp.reorder_levels(
            ["scorer", "individuals", "bodyparts", "coords"], axis=1
        )
    return df


def _get_pes_args(config_file, h5file, individual_name):
    if "DLC" not in h5file or not h5file.endswith(".h5"):
        raise IOError("The file passed in is not a DeepLabCut h5 data file.")

    cfg = auxiliaryfunctions.read_config(config_file)

    vidname, scorer = os.path.split(h5file)[-1].split("DLC")
    scorer = "DLC" + os.path.splitext(scorer)[0]
    video = None

    df = _ensure_individuals_in_header(pd.read_hdf(h5file), individual_name)

    # Fetch the corresponding metadata pickle file
    paf_graph = []
    filename, _ = os.path.splitext(h5file)
    for i, c in enumerate(filename[::-1]):
        if c.isnumeric():
            break
    if i > 0:
        filename = filename[:-i]
    metadata_file = filename + "_meta.pickle"
    if os.path.isfile(metadata_file):
        with open(metadata_file, "rb") as file:
            metadata = pickle.load(file)
        test_cfg = metadata["data"]["DLC-model-config file"]
        paf_graph = test_cfg.get("partaffinityfield_graph", [])
        if paf_graph:
            paf_inds = test_cfg.get("paf_best")
            if paf_inds is not None:
                paf_graph = [paf_graph[i] for i in paf_inds]
    else:
        warnings.warn("Metadata not found...")

    for video_path, params in cfg["video_sets"].items():
        if vidname in video_path:
            video = video_path, params["crop"]
            break

    if video is None:
        warnings.warn(f"The video file corresponding to {h5file} could not be found...")
        video = "fake_path", "0, 0, 0, 0"

        timestamps = (
            df.index.tolist()
        )  # setting timestamps to dummy TODO: extract timestamps in DLC?
    else:
        timestamps = get_movie_timestamps(video[0])
    return scorer, df, video, paf_graph, timestamps, cfg


def _write_pes_to_nwbfile(nwbfile, animal, df_animal, scorer, video, paf_graph, timestamps):
    pose_estimation_series = []
    for kpt, xyp in df_animal.groupby(level="bodyparts", axis=1, sort=False):
        data = xyp.to_numpy()

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
        # TODO check if this is a mandatory arg in ndx-pose (can skip if video is not found_
        dimensions=[list(map(int, video[1].split(",")))[1::2]],
        scorer=scorer,
        source_software="DeepLabCut",
        source_software_version=__version__,
        nodes=[pes.name for pes in pose_estimation_series],
        edges=paf_graph,
    )
    if 'behavior' in nwbfile.processing:
        behavior_pm = nwbfile.processing["behavior"]
    else:
        behavior_pm = nwbfile.create_processing_module(
            name="behavior", description="processed behavioral data"
        )
    behavior_pm.add(pe)
    return nwbfile


def write_subject_to_nwb(nwbfile, h5file, individual_name, config_file):
    """
    Given, subject name, write h5file to an existing nwbfile.

    Parameters
    ----------
    nwbfile: pynwb.NWBFile
        nwbfile to write the subject specific pose estimation series.
    h5file : str
        Path to a h5 data file
    individual_name : str
        Name of the subject (whose pose is predicted) for single-animal DLC project.
        For multi-animal projects, the names from the DLC project will be used directly.
    config_file : str
        Path to a project config.yaml file
    config_dict : dict
        dict containing configuration options. Provide this as alternative to config.yml file.

    Returns
    -------
    nwbfile: pynwb.NWBFile
        nwbfile with pes written in the behavior module
    """
    scorer, df, video, paf_graph, timestamps, _ = _get_pes_args(config_file, h5file, individual_name)
    df_animal = df.groupby(level="individuals", axis=1).get_group(individual_name)
    return _write_pes_to_nwbfile(nwbfile, individual_name, df_animal, scorer, video, paf_graph, timestamps)


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

    individual_name : str
        Name of the subject (whose pose is predicted) for single-animal DLC project.
        For multi-animal projects, the names from the DLC project will be used directly.

    TODO: allow one to overwrite those names, with a mapping?

    Returns
    -------
    list of str
        List of paths to the newly created NWB data files.
        By default NWB files are stored in the same folder as the h5file.

    """
    scorer, df, video, paf_graph, timestamps, cfg = _get_pes_args(config, h5file, individual_name)
    output_paths = []
    for animal, df_ in df.groupby(level="individuals", axis=1):
        nwbfile = NWBFile(
            session_description=cfg["Task"],
            experimenter=cfg["scorer"],
            identifier=scorer,
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
        )

        # TODO Store the test_pose_config as well?
        nwbfile = _write_pes_to_nwbfile(nwbfile, animal, df_, scorer, video, paf_graph, timestamps)
        output_path = h5file.replace(".h5", f"_{animal}.nwb")
        with warnings.catch_warnings(), NWBHDF5IO(output_path, mode="w") as io:
            warnings.filterwarnings("ignore", category=DtypeConversionWarning)
            io.write(nwbfile)
        output_paths.append(output_path)

    return output_paths


def convert_nwb_to_h5(nwbfile):
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
                names=["scorer", "individuals", "bodyparts", "coords"],
            )
            dfs.append(
                pd.DataFrame(array, np.asarray(pes.timestamps).astype(int), cols)
            )
    df = pd.concat(dfs, axis=1)
    df.to_hdf(nwbfile.replace(".nwb", ".h5"), key="poses")
    return df
