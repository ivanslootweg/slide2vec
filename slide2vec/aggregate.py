import gc
import os
import numpy as np
import tqdm
import torch
import argparse
import traceback
import torchvision
import pandas as pd
import multiprocessing as mp
import wholeslidedata as wsd

from pathlib import Path
from contextlib import nullcontext

import slide2vec.distributed as distributed

from slide2vec.utils import fix_random_seeds
from slide2vec.utils.config import get_cfg_from_file
from slide2vec.models import ModelFactory
from slide2vec.utils.signal_utils import ExitHandler, PathUnlinker

torchvision.disable_beta_transforms_warning()


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("slide2vec", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Name of output subdirectory",
    )
    return parser


def scale_coordinates(wsi_fp, coordinates, spacing, backend):
    """
    Scale coordinates based on the target spacing.
    """
    wsi = wsd.WholeSlideImage(wsi_fp, backend=backend)
    min_spacing = wsi.spacings[0]
    scale = min_spacing / spacing
    scaled_coordinates = (coordinates * scale).astype(int)
    return scaled_coordinates


def main(args):
    # setup configuration
    cfg = get_cfg_from_file(args.config_file)
    output_dir = Path(cfg.output_dir, args.run_id)
    cfg.output_dir = str(output_dir)

    coordinates_dir = Path(cfg.output_dir, "coordinates")
    fix_random_seeds(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers_embedding)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    process_list = Path(cfg.output_dir, "process_list.csv")
    assert (
        process_list.is_file()
    ), "Process list CSV not found. Ensure tiling has been run."
    process_df = pd.read_csv(process_list)
    skip_feature_aggregation = process_df["aggregation_status"].str.contains("success").all()

    if skip_feature_aggregation and distributed.is_main_process():
        print("Feature aggregation already completed.")
        return

    model = ModelFactory(cfg.model).get_model()

    # select slides where tile-level feature extraction was successfull
    tiled_df = process_df[process_df.tiling_status == "success"]
    tiled_and_features_df = tiled_df[tiled_df.feature_status == "success"]
    mask = tiled_and_features_df["aggregation_status"] != "success"
    process_stack = tiled_and_features_df[mask]
    wsi_paths_to_process = [Path(x) for x in process_stack.wsi_path.values.tolist()]

    features_dir = Path(cfg.output_dir, "features_slide")
    wsi_paths_to_process = [ w for w in wsi_paths_to_process if not os.path.exists(features_dir / f"{w.stem.replace(' ','_')}.pt")]
    total = len(wsi_paths_to_process)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if cfg.speed.fp16
        else nullcontext()
    )
    feature_aggregation_updates = {}

    for wsi_fp in tqdm.tqdm(
        wsi_paths_to_process,
        desc="Pooling tile features",
        unit="slide",
        total=total,
        leave=True,
    ):
        try:

            name = wsi_fp.stem.replace(" ", "_")
            lock_path = Path(cfg.output_dir, "locks", f"{wsi_fp.stem}_agg.lock")
            # Only the main rank handles locking
            if lock_path.exists():
                if distributed.is_main_process():
                    print(f"[{name}] Skipping (lock exists)")
                skip_slide = True
                path_unlinker = None
            else:
                skip_slide = False
                if distributed.is_main_process():
                    lock_path.parent.mkdir(parents=True, exist_ok=True)
                    lock_path.touch()
                    path_unlinker = ExitHandler.instance().add_path_unlinker(lock_path)

            torch.distributed.barrier()  # Ensure all processes wait until lock is made or decision is made for skipping 
            if skip_slide:
                continue

            feature_path = features_dir / f"{name}.pt"
            if not Path(feature_path).exists():
                print(f"[{name}] processing")

                coordinates_file = coordinates_dir / f"{name}.npy"
                coordinates_arr = np.load(coordinates_file, allow_pickle=True)
                coordinates = (np.array([coordinates_arr["x"], coordinates_arr["y"]]).T).astype(int)


                # run forward pass with slide encoder
                if cfg.model.name == "prov-gigapath":
                    # need to scale coordinates for gigapath
                    scaled_coordinates = scale_coordinates(wsi_fp, coordinates, cfg.tiling.params.spacing, cfg.tiling.backend)
                    coordinates = torch.tensor(
                        scaled_coordinates,
                        dtype=torch.int,
                        device=model.device,
                    )
                else:
                    coordinates = torch.tensor(
                        coordinates,
                        dtype=torch.int,
                        device=model.device,
                    )

                with torch.inference_mode():
                    with autocast_context:
                        features = torch.load(feature_path).to(model.device)
                        tile_size_lv0 = coordinates_arr["tile_size_lv0"][0]
                        wsi_feature = model.forward_slide(
                            features,
                            tile_coordinates=coordinates,
                            tile_size_lv0=tile_size_lv0,
                        )


                torch.distributed.barrier()

                if distributed.is_main_process():
                    torch.save(wsi_feature, feature_path)
                    del wsi_feature
                    torch.cuda.empty_cache()
                    gc.collect()

                torch.distributed.barrier()

            else:
                print(f"[{name}] exists")
                
            feature_aggregation_updates[str(wsi_fp)] = {"status": "success"}

        except Exception as e:
            feature_aggregation_updates[str(wsi_fp)] = {
                "status": "failed",
                "error": str(e),
                "traceback": str(traceback.format_exc()),
            }

        finally:
            torch.distributed.barrier()  # All ranks wait before releasing the lock
            if distributed.is_main_process():
                if path_unlinker:
                    lock_path.unlink()
                    ExitHandler.instance().remove(path_unlinker)
            if distributed.is_enabled_and_multiple_gpus():
                torch.distributed.barrier()

        # update process_df
        # process_df_lock_path = Path(cfg.output_dir, "locks", "process_df.lock")
        # # Wait until no one else is updating process_df
        # while process_df_lock_path.exists():
        #     torch.distributed.barrier()
        # # Acquire lock
        # if distributed.is_main_process():
        #     process_df_lock_path.parent.mkdir(parents=True, exist_ok=True)
        #     process_df_lock_path.touch()
        #     process_df_path_unlinker = ExitHandler.instance().add_path_unlinker(process_df_lock_path)
        # torch.distributed.barrier()

        # try:
        #     if distributed.is_main_process():
        #         status_info = feature_aggregation_updates[str(wsi_fp)]
        #         process_df.loc[
        #             process_df["wsi_path"] == str(wsi_fp), "aggregation_status"
        #         ] = status_info["status"]
        #         if "error" in status_info:
        #             process_df.loc[
        #                 process_df["wsi_path"] == str(wsi_fp), "error"
        #             ] = status_info["error"]
        #             process_df.loc[
        #                 process_df["wsi_path"] == str(wsi_fp), "traceback"
        #             ] = status_info["traceback"]
        #         process_df.to_csv(process_list, index=False)

        # finally:
        #     torch.distributed.barrier()  # wait until all ranks finish reading/writing
        #     if distributed.is_main_process() and 'process_df_path_unlinker' in locals():
        #         process_df_lock_path.unlink()
        #         ExitHandler.instance().remove(process_df_path_unlinker)
        #     torch.distributed.barrier()
                
        status_info = feature_aggregation_updates[str(wsi_fp)]
        process_df.loc[
            process_df["wsi_path"] == str(wsi_fp), "aggregation_status"
        ] = status_info["status"]
        if "error" in status_info:
            process_df.loc[
                process_df["wsi_path"] == str(wsi_fp), "error"
            ] = status_info["error"]
            process_df.loc[
                process_df["wsi_path"] == str(wsi_fp), "traceback"
            ] = status_info["traceback"]
        process_df.to_csv(process_list, index=False)


    if distributed.is_enabled_and_multiple_gpus():
        torch.distributed.barrier()

    if distributed.is_main_process():

        # summary logging
        slides_with_tile_features = len(tiled_and_features_df)
        total_slides = len(process_df)
        failed_feature_aggregation = process_df[
            ~(process_df["aggregation_status"] == "success")
        ]
        print("=+=" * 10)
        print(f"Total number of slides with tile-level features: {slides_with_tile_features}/{total_slides}")
        print(f"Failed slide-level feature aggregation: {len(failed_feature_aggregation)}")
        print(
            f"Completed slide-level feature aggregation: {total_slides - len(failed_feature_aggregation)}"
        )
        print("=+=" * 10)

    if distributed.is_enabled():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
