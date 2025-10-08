import gc
import os
import h5py
import tqdm
import torch
import argparse
import traceback
import torchvision
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from contextlib import nullcontext

import slide2vec.distributed as distributed

from slide2vec.utils import fix_random_seeds
from slide2vec.utils.config import get_cfg_from_file, setup_distributed
from slide2vec.models import ModelFactory
from slide2vec.data import TileDataset, RegionUnfolding
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


def create_transforms(cfg, model):
    if cfg.model.level in ["tile", "slide"]:
        return model.get_transforms()
    elif cfg.model.level == "region":
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                RegionUnfolding(model.tile_size),
                model.get_transforms(),
            ]
        )
    else:
        raise ValueError(f"Unknown model level: {cfg.model.level}")


def create_dataset(wsi_fp, coordinates_dir, spacing, backend, transforms):
    return TileDataset(
        wsi_fp,
        coordinates_dir,
        spacing,
        backend=backend,
        transforms=transforms,
    )


def run_inference(dataloader, model, device, autocast_context, unit, batch_size, feature_path, feature_dim, dtype):
    with h5py.File(feature_path, "w") as f:
        features = f.create_dataset("features", shape=(0, *feature_dim), maxshape=(None, *feature_dim), dtype=dtype, chunks=(batch_size, *feature_dim))
        indices = f.create_dataset("indices", shape=(0,), maxshape=(None,), dtype='int64', chunks=(batch_size,))
        with torch.inference_mode(), autocast_context:
            for batch in tqdm.tqdm(
                dataloader,
                desc=f"Inference on GPU {distributed.get_global_rank()}",
                unit=unit,
                unit_scale=batch_size,
                leave=False,
                position=2 + distributed.get_global_rank(),
                ncols=200
            ):
                idx, image = batch
                image = image.to(device, non_blocking=True)
                feature = model(image).cpu().numpy()
                features.resize(features.shape[0] + feature.shape[0], axis=0)
                features[-feature.shape[0]:] = feature
                indices.resize(indices.shape[0] + idx.shape[0], axis=0)
                indices[-idx.shape[0]:] = idx.cpu().numpy()

                # cleanup
                del image, feature

    # cleanup
    torch.cuda.empty_cache()
    gc.collect()


def load_and_sort_features(tmp_dir, name, expected_len=None):
    features_list, indices_list = [], []
    for rank in range(distributed.get_global_size()):
        fp = tmp_dir / f"{name}-rank_{rank}.h5"
        with h5py.File(fp, "r") as f:
            features_list.append(torch.from_numpy(f["features"][:]))
            indices_list.append(torch.from_numpy(f["indices"][:]))
        os.remove(fp)
    features = torch.cat(features_list, dim=0)
    indices = torch.cat(indices_list, dim=0)
    order = torch.argsort(indices)
    indices = indices[order]
    features = features[order]

    # deduplicate
    keep = torch.ones_like(indices, dtype=torch.bool)
    keep[1:] = indices[1:] != indices[:-1]
    indices = indices[keep]
    features = features[keep]
    if expected_len is not None:
        assert len(indices) == expected_len, f"Got {len(indices)} items, expected {expected_len}"
    
    
    return features

def update_process_list_with_features(process_df, features_dir, process_list_path):
    """
    Check for existing feature files and update the process_df accordingly.
    Saves the updated process_df back to CSV.

    Args:
        process_df (pd.DataFrame): The process list dataframe.
        features_dir (Path): Path where extracted feature files are stored.
        process_list_path (Path): Path to the CSV process list.
    Returns:
        pd.DataFrame: Updated process_df with 'feature_status' set to 'success'
                      where feature files exist.
    """
    features_dir.mkdir(exist_ok=True, parents=True)

    for idx, row in process_df.iterrows():
        if row["tiling_status"] == "success":
            feat_path = features_dir / f"{Path(row['wsi_path']).stem.replace(' ', '_')}.pt"
            if feat_path.exists():
                process_df.at[idx, "feature_status"] = "success"

    # Save immediately
    process_df.to_csv(process_list_path, index=False)
    return process_df


def main(args):
    # setup configuration
    cfg = get_cfg_from_file(args.config_file)
    output_dir = Path(cfg.output_dir, args.run_id)
    cfg.output_dir = str(output_dir)
    setup_distributed()
    coordinates_dir = Path(cfg.output_dir, "coordinates")
    fix_random_seeds(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    unit = "tile" if cfg.model.level != "region" else "region"

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers_embedding)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    # --- Load process list ---
    process_list = Path(cfg.output_dir, f"{cfg.process_list}.csv")
    assert process_list.is_file(), "Process list CSV not found. Ensure tiling has been run."
    process_df = pd.read_csv(process_list)

    # --- Update with existing feature files ---
    features_dir = Path(cfg.output_dir, f"features_{cfg.model.name}")
    process_df = update_process_list_with_features(process_df, features_dir, process_list)

    # --- Check if all features are done ---
    skip_feature_extraction = process_df.query("tiling_status == 'success'")["feature_status"].eq("success").all()

    if skip_feature_extraction:
        if distributed.is_main_process():
            print("=+=" * 10)
            print(f"All slides have been embedded. Skipping {unit}-level feature extraction step.")
            print("=+=" * 10)
        if distributed.is_enabled():
            torch.distributed.destroy_process_group()
        return
        ## END OF PROGRAM ##

    # --- Otherwise continue with extraction ---
    model = ModelFactory(cfg.model).get_model()
    if distributed.is_main_process():
        print(f"Starting {unit}-level feature extraction...")
    torch.distributed.barrier()

    # Select slides that still need features
    tiled_df = process_df[process_df.tiling_status == "success"]
    process_stack = tiled_df[tiled_df.feature_status != "success"]
    wsi_paths_to_process = [Path(x) for x in process_stack.wsi_path.values.tolist()]
    total = len(wsi_paths_to_process)

    tmp_dir = Path("/tmp")
    if distributed.is_main_process():
        tmp_dir.mkdir(exist_ok=True, parents=True)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if cfg.speed.fp16
        else nullcontext()
    )
    feature_extraction_updates = {}

    transforms = create_transforms(cfg, model)
    print(f"transforms: {transforms}")

    for wsi_fp in tqdm.tqdm(
        wsi_paths_to_process,
        desc="Inference",
        unit="slide",
        total=total,
        leave=True,
        disable=not distributed.is_main_process(),
        position=1,
        ncols=100
    ):
        try:
            name = wsi_fp.stem.replace(" ", "_")
            lock_path = Path(cfg.output_dir, "locks", f"{wsi_fp.stem}.lock")
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
                dataset = create_dataset(wsi_fp, coordinates_dir, cfg.tiling.params.spacing, cfg.tiling.backend, transforms)
                if distributed.is_enabled_and_multiple_gpus():
                    sampler = torch.utils.data.DistributedSampler(
                        dataset,
                        shuffle=False,
                        drop_last=False,
                    )
                else:
                    sampler = None
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=cfg.model.batch_size,
                    sampler=sampler,
                    num_workers=num_workers,
                    pin_memory=True,
                )

                name = wsi_fp.stem.replace(" ", "_")
                feature_path = features_dir / f"{name}.pt"
                tmp_feature_path = tmp_dir / f"{name}-rank_{distributed.get_global_rank()}.h5"

                # get feature dimension and dtype using a dry run
                with torch.inference_mode(), autocast_context:
                    sample_batch = next(iter(dataloader))
                    sample_image = sample_batch[1].to(model.device)
                    sample_feature = model(sample_image).cpu().numpy()
                    feature_dim = sample_feature.shape[1:]
                    dtype = sample_feature.dtype

                run_inference(
                    dataloader,
                    model,
                    model.device,
                    autocast_context,
                    unit,
                    cfg.model.batch_size,
                    tmp_feature_path,
                    feature_dim,
                    dtype,
                )

                torch.distributed.barrier()

                if distributed.is_main_process():
                    wsi_feature = load_and_sort_features(tmp_dir, name)
                    torch.save(wsi_feature, feature_path)

                    # cleanup
                    del wsi_feature
                    torch.cuda.empty_cache()
                    gc.collect()

                torch.distributed.barrier()
            else:
                print(f"[{name}] exists")
                
            feature_extraction_updates[str(wsi_fp)] = {"status": "success"}

        except Exception as e:
            print(e)
            feature_extraction_updates[str(wsi_fp)] = {
                "status": "failed",
                "error": str(e),
                "traceback": str(traceback.format_exc()),
            }
            
        finally: 
            if distributed.is_main_process():
                if path_unlinker and (not os.path.exists(lock_path)):
                    print("Not removing lock. Does not exists ", lock_path)
                elif path_unlinker:
                    lock_path.unlink()
                    ExitHandler.instance().remove(path_unlinker)
            if distributed.is_enabled_and_multiple_gpus():
                torch.distributed.barrier()



                
        if distributed.is_main_process():
            status_info = feature_extraction_updates[str(wsi_fp)]
            process_df.loc[
                process_df["wsi_path"] == str(wsi_fp), "feature_status"
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
        slides_with_tiles = len(tiled_df)   
        total_slides = len(process_df)
        failed_feature_extraction = process_df[
            ~(process_df["feature_status"] == "success")
        ]
        print("=+=" * 10)
        print(f"Total number of slides with {unit}s: {slides_with_tiles}/{total_slides}")
        print(f"Failed {unit}-level feature extraction: {len(failed_feature_extraction)}")
        print(
            f"Completed {unit}-level feature extraction: {total_slides - len(failed_feature_extraction)}"
        )
        print("=+=" * 10)

    if distributed.is_enabled():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
