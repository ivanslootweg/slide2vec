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
import traceback
import shutil
from pathlib import Path
from contextlib import nullcontext

import slide2vec.distributed as distributed
from slide2vec.utils import fix_random_seeds
from slide2vec.utils.config import get_cfg_from_file, setup_distributed
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


def deduplicate_features(features):
    # deduplicate. where feature is identical to next feature
    # assumes that features were sorted by index in embed.py::load_and_sort_features    
    N = features.size(0)
    constructed_indices = torch.arange(N)
    keep = torch.ones(N, dtype=torch.bool)
    keep[1:] = (features[1:] != features[:-1]).any(dim=1)
    constructed_indices = constructed_indices[keep]
    dedup_features = features[keep]
    # print("deduplicated features: ", dedup_features.size(),"\n")
    # print("indices: ", constructed_indices.size(),"\n")
    return dedup_features

def scale_coordinates(wsi_fp, coordinates, spacing, backend):
    """
    Scale coordinates based on the target spacing.
    """
    wsi = wsd.WholeSlideImage(wsi_fp, backend=backend)
    min_spacing = wsi.spacings[0]
    scale = min_spacing / spacing
    scaled_coordinates = (coordinates * scale).astype(int)
    return scaled_coordinates

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
        if row["feature_status"] == "success":
            feat_path = features_dir / f"{Path(row['wsi_path']).stem.replace(' ', '_')}.pt"
            if feat_path.exists():
                process_df.at[idx, "aggregation_status"] = "success"

    # Save immediately
    process_df.to_csv(process_list_path, index=False)
    return process_df



def main(args):
    # setup configuration
    
    cfg = get_cfg_from_file(args.config_file)
    output_dir = Path(cfg.output_dir, args.run_id)
    cfg.output_dir = str(output_dir)
    coordinates_dir = Path(cfg.output_dir, "coordinates")
    setup_distributed()
    fix_random_seeds(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ExitHandler.instance()
    num_workers = min(mp.cpu_count(), cfg.speed.num_workers_embedding)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    features_dir = Path(cfg.output_dir, "features_slide")
    patch_features_dir = Path(cfg.output_dir, f"features_{cfg.model.name}")
    dedup_features_dir = Path(str(patch_features_dir )+ "_dedup")
    
    process_list = Path(cfg.output_dir, "process_list.csv")
    assert (
        process_list.is_file()
    ), "Process list CSV not found. Ensure tiling has been run."
    process_df = pd.read_csv(process_list)

    process_df = update_process_list_with_features(process_df,features_dir,process_list)
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
    print("wsi's to process: ", len(wsi_paths_to_process))

    os.makedirs(dedup_features_dir,exist_ok=True)
    wsi_paths_to_process = [ w for w in wsi_paths_to_process if not os.path.exists(features_dir / f"{w.stem.replace(' ','_')}.pt")]
    total = len(wsi_paths_to_process)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if cfg.speed.fp16
        else nullcontext()
    )
    feature_aggregation_updates = {}

    ## slide2vec repo
    for wsi_fp in tqdm.tqdm(
        wsi_paths_to_process,
        desc="Pooling tile features",
        unit="slide",
        total=total,
        leave=True,
    ):
        try:
            name = wsi_fp.stem.replace(" ", "_")
            coordinates_file = coordinates_dir / f"{name}.npy"
            coordinates_arr = np.load(coordinates_file, allow_pickle=True)
            coordinates = (np.array([coordinates_arr["x"], coordinates_arr["y"]]).T).astype(int)

            feature_path = features_dir / f"{name}.pt"
            patch_feature_path = patch_features_dir / f"{name}.pt"
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
            # coordinates = coordinates.long()
            with autocast_context:
                features = torch.load(patch_feature_path).to(model.device)
                tile_size_lv0 = coordinates_arr["tile_size_lv0"][0]

                # --- Deduplication ---
                if features.size(0) != (coordinates_arr.shape[0]):
                    dedup_feature = deduplicate_features(features)
                    if dedup_feature.size(0) == coordinates_arr.shape[0]:
                        features = dedup_feature
                    else:
                        os.remove(patch_feature_path)
                        raise AssertionError(
                            f"deduplication unsuccesful: produced {dedup_feature.size()} instead of {coordinates_arr.shape}"
                        )
                # elif features.size(0) != coordinates_arr.shape[0]:
                #     # os.remove(patch_feature_path)
                #     raise AssertionError(
                #         f"mismatch: features {features.size()} vs coordinates {coordinates_arr.shape}"
                #     )
                    
                    print("deduplicated features")
                    torch.save(features, dedup_features_dir / f"{name}.pt")

            with torch.inference_mode():
                    wsi_feature = model.forward_slide(
                        features,
                        tile_coordinates=coordinates,
                        tile_size_lv0=tile_size_lv0,
                    )
                

            torch.save(wsi_feature, feature_path)
            del wsi_feature
            del features
            torch.cuda.empty_cache()
            gc.collect()

            feature_aggregation_updates[str(wsi_fp)] = {"status": "success"}

        except Exception as e:
            feature_aggregation_updates[str(wsi_fp)] = {
                "status": "failed",
                "error": str(e),
                "traceback": str(traceback.format_exc()),
            }
            print(e)

        # update process_df
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

    # summary logging
    slides_with_tile_features = len(tiled_and_features_df)
    total_slides = len(process_df)
    failed_feature_aggregation = process_df[
        ~(process_df["aggregation_status"] == "success")
    ]
    print("=+=" * 10)
    print(f"Total number of slides with tile-level features: {slides_with_tile_features}/{total_slides}")
    print(f"Failed slide-level feature aggregation: {len(failed_feature_aggregation)}/{total_slides}")
    print(
        f"Completed slide-level feature aggregation: {total_slides - len(failed_feature_aggregation)}/{total_slides}"
    )
    print("=+=" * 10)


    ##### chatgpt

    # for wsi_fp in tqdm.tqdm(
    #         wsi_paths_to_process,
    #         desc="Pooling tile features",
    #         unit="slide",
    #         total=total,
    #         leave=True,
    #         ncols=20
    #     ):
    #     coordinates = None
    #     features = None
    #     wsi_feature = None
    #     path_unlinker = None
    #     skip_slide = False

    #     name = wsi_fp.stem.replace(" ", "_")
    #     lock_path = Path(cfg.output_dir, "locks", f"{wsi_fp.stem}_agg.lock")

    #     try:
    #         # --- Lock handling ---
    #         if lock_path.exists():
    #             if distributed.is_main_process():
    #                 print(f"[{name}] Skipping (lock exists)")
    #             skip_slide = True
    #             feature_aggregation_updates[str(wsi_fp)] = {"status": "skipped"}
    #         else:
    #             if distributed.is_main_process():
    #                 lock_path.parent.mkdir(parents=True, exist_ok=True)
    #                 lock_path.touch()
    #                 path_unlinker = ExitHandler.instance().add_path_unlinker(lock_path)

    #         # --- Barrier after lock decision ---
    #         if distributed.is_enabled_and_multiple_gpus():
    #             torch.distributed.barrier()

    #         if skip_slide:
    #             # still hit all later barriers in finally
    #             continue

    #         # --- Feature / coordinate loading ---
    #         feature_path = features_dir / f"{name}.pt"
    #         patch_feature_path = patch_features_dir / f"{name}.pt"

    #         if not feature_path.exists():
    #             print(f"[{name}] processing")

    #             coordinates_file = coordinates_dir / f"{name}.npy"
    #             coordinates_arr = np.load(coordinates_file, allow_pickle=True)
    #             coordinates = (np.array([coordinates_arr["x"], coordinates_arr["y"]]).T).astype(int)

    #             if cfg.model.name == "prov-gigapath":
    #                 scaled_coordinates = scale_coordinates(
    #                     wsi_fp, coordinates, cfg.tiling.params.spacing, cfg.tiling.backend
    #                 )
    #                 coordinates = torch.tensor(scaled_coordinates, dtype=torch.int, device=model.device)
    #             else:
    #                 coordinates = torch.tensor(coordinates, dtype=torch.int, device=model.device)

    #             with torch.inference_mode():
    #                 coordinates = coordinates.long()
    #                 with autocast_context:
    #                     features = torch.load(patch_feature_path).to(model.device)

    #                     # --- Deduplication ---
    #                     if features.size(0) == (coordinates_arr.shape[0] + 1):
    #                         dedup_feature = deduplicate_features(features)
    #                         if dedup_feature.size(0) == coordinates_arr.shape[0]:
    #                             features = dedup_feature
    #                             print("deduplicated features")
    #                         else:
    #                             os.remove(patch_feature_path)
    #                             raise AssertionError(
    #                                 f"deduplication unsuccesful: produced {dedup_feature.size()} instead of {features.size()}"
    #                             )
    #                     elif features.size(0) != coordinates_arr.shape[0]:
    #                         raise AssertionError(
    #                             f"mismatch: features {features.size()} vs coordinates {coordinates_arr.shape}"
    #                         )

    #                     print("deduplicated features")
    #                     torch.save(features, dedup_features_dir / f"{name}.pt")
    #                     # --- Slide feature aggregation ---
    #                     tile_size_lv0 = coordinates_arr["tile_size_lv0"][0]
    #                     wsi_feature = model.forward_slide(
    #                         features,
    #                         tile_coordinates=coordinates,
    #                         tile_size_lv0=tile_size_lv0,
    #                     )

    #         # --- Barrier before saving ---
    #         if distributed.is_enabled_and_multiple_gpus():
    #             torch.distributed.barrier()

    #         # --- Save features if main process ---
    #         if distributed.is_main_process() and wsi_feature is not None:
    #             torch.save(wsi_feature, feature_path)
    #             del wsi_feature
    #             torch.cuda.empty_cache()
    #             gc.collect()
    #         elif distributed.is_main_process() and not feature_path.exists():
    #             print(f"[{name}] no wsi_feature available and feature file missing!")

    #         # --- Extra barrier after saving ---
    #         if distributed.is_enabled_and_multiple_gpus():
    #             torch.distributed.barrier()
    #         else:
    #             print(f"[{name}] exists")

    #         # --- Mark success ---
    #         feature_aggregation_updates[str(wsi_fp)] = {"status": "success"}

    #     except IndexError as e:
    #         print(f"[WARNING] IndexError processing {wsi_fp}: {e}")
    #         if coordinates is not None:
    #             try: print("coordinates:", coordinates.size())
    #             except: pass
    #         if features is not None:
    #             try: print("features:", features.size())
    #             except: pass
    #         tb = traceback.format_exc()
    #         print(tb)
    #         feature_aggregation_updates[str(wsi_fp)] = {
    #             "status": "failed",
    #             "error": "IndexError",
    #             "traceback": tb,
    #         }

    #     except Exception as e:
    #         print(f"[ERROR] {e}")
    #         if coordinates is not None:
    #             try: print("coordinates:", coordinates.size())
    #             except: pass
    #         if features is not None:
    #             try: print("features:", features.size())
    #             except: pass
    #         tb = traceback.format_exc()
    #         print(tb)
    #         feature_aggregation_updates[str(wsi_fp)] = {
    #             "status": "failed",
    #             "error": str(e),
    #             "traceback": tb,
    #         }

    #     finally:
    #         # --- Lock cleanup ---
    #         if distributed.is_main_process() and path_unlinker:
    #             try:
    #                 if os.path.exists(lock_path):
    #                     lock_path.unlink()
    #                 ExitHandler.instance().remove(path_unlinker)
    #             except Exception as e:
    #                 print(f"Failed to remove lock {lock_path}: {e}")

    #         # --- Free memory ---
    #         if features is not None:
    #             del features
    #             torch.cuda.empty_cache()

    #         # --- Final barrier (symmetric) ---
    #         if distributed.is_enabled_and_multiple_gpus():
    #             torch.distributed.barrier()

    #     # --- Update process_df ---
    #     if str(wsi_fp) in feature_aggregation_updates:
    #         status_info = feature_aggregation_updates[str(wsi_fp)]
    #         process_df.loc[process_df["wsi_path"] == str(wsi_fp), "aggregation_status"] = status_info["status"]
    #         if "error" in status_info:
    #             process_df.loc[process_df["wsi_path"] == str(wsi_fp), "error"] = status_info["error"]
    #             process_df.loc[process_df["wsi_path"] == str(wsi_fp), "traceback"] = status_info["traceback"]
    #         process_df.to_csv(process_list, index=False)
    #     else:
    #         print(f"[BUG] No status recorded for {wsi_fp}; skipping df update.")

    ### old

    # for wsi_fp in tqdm.tqdm(
    #     wsi_paths_to_process,
    #     desc="Pooling tile features",
    #     unit="slide",
    #     total=total,
    #     leave=True,
    #     ncols = 20
    # ):
    #     try:

    #         name = wsi_fp.stem.replace(" ", "_")
    #         lock_path = Path(cfg.output_dir, "locks", f"{wsi_fp.stem}_agg.lock")
    #         # Only the main rank handles locking
    #         if lock_path.exists():
    #             if distributed.is_main_process():
    #                 print(f"[{name}] Skipping (lock exists)")
    #             skip_slide = True
    #             path_unlinker = None
    #         else:
    #             skip_slide = False
    #             if distributed.is_main_process():
    #                 lock_path.parent.mkdir(parents=True, exist_ok=True)
    #                 lock_path.touch()
    #                 path_unlinker = ExitHandler.instance().add_path_unlinker(lock_path)

    #         if distributed.is_enabled_and_multiple_gpus():
    #             torch.distributed.barrier()  # Ensure all processes wait until lock is made or decision is made for skipping 
    #         if skip_slide:
    #             continue

    #         feature_path = features_dir / f"{name}.pt"
    #         patch_feature_path = patch_features_dir / f"{name}.pt"

    #         if not Path(feature_path).exists():
    #             print(f"[{name}] processing")

    #             coordinates_file = coordinates_dir / f"{name}.npy"
    #             coordinates_arr = np.load(coordinates_file, allow_pickle=True)
    #             coordinates = (np.array([coordinates_arr["x"], coordinates_arr["y"]]).T).astype(int)


    #             # run forward pass with slide encoder
    #             if cfg.model.name == "prov-gigapath":
    #                 # need to scale coordinates for gigapath
    #                 scaled_coordinates = scale_coordinates(wsi_fp, coordinates, cfg.tiling.params.spacing, cfg.tiling.backend)
    #                 coordinates = torch.tensor(
    #                     scaled_coordinates,
    #                     dtype=torch.int,
    #                     device=model.device,
    #                 )
    #             else:
    #                 coordinates = torch.tensor(
    #                     coordinates,
    #                     dtype=torch.int,
    #                     device=model.device,
    #                 )

    #             with torch.inference_mode():
    #                 coordinates = coordinates.long()
    #                 with autocast_context:
    #                     features = torch.load(patch_feature_path).to(model.device)
    #                     if features.size()[0] == (coordinates_arr.shape[0] + 1):
    #                         dedup_feature = deduplicate_features(features)
    #                         if dedup_feature.size()[0] == coordinates_arr.shape[0] :
    #                             features  = dedup_feature
    #                             del dedup_feature
    #                             print("deduplicated features")
    #                             torch.save(features, dedup_features_dir / f"{name}.pt" )

    #                         else:
    #                             # delete patch features and re-process with embed.py
    #                             os.remove(patch_feature_path)
    #                             raise AssertionError(f"deduplication unsuccesful: produced {dedup_feature.size()} instead of {features.size()}")
                            
    #                     elif features.size()[0]  != coordinates_arr.shape[0]:
    #                         raise AssertionError(f"mismatch : features {features.size()} . coordinates  {coordinates_arr.shape}")
                            
    #                     tile_size_lv0 = coordinates_arr["tile_size_lv0"][0]
    #                     wsi_feature = model.forward_slide(
    #                         features,
    #                         tile_coordinates=coordinates,
    #                         tile_size_lv0=tile_size_lv0,
    #                     )

    #         if distributed.is_enabled_and_multiple_gpus():
    #             torch.distributed.barrier()

    #             if distributed.is_main_process():
    #                 torch.save(wsi_feature, feature_path)
    #                 del wsi_feature
    #                 torch.cuda.empty_cache()
    #                 gc.collect()

    #         if distributed.is_enabled_and_multiple_gpus():
    #             torch.distributed.barrier()

    #         else:
    #             print(f"[{name}] exists")
                
    #         feature_aggregation_updates[str(wsi_fp)] = {"status": "success"}

    #     except IndexError : 
    #         print("coordinates: ", coordinates.size())
    #         print("features: ", features.size())
    #         print(traceback.format_exc())
    #         feature_aggregation_updates[str(wsi_fp)] = {
    #             "status": "failed",
    #             "error": "error",
    #             "traceback": str(traceback.format_exc()),
    #         }
    #         continue
            
    #     except Exception as e:
    #         print(e)
    #         print("coordinates: ", coordinates.size())
    #         print("features: ", features.size())
    #         print(traceback.format_exc())
    #         feature_aggregation_updates[str(wsi_fp)] = {
    #             "status": "failed",
    #             "error": str(e),
    #             "traceback": str(traceback.format_exc()),
    #         }

    #     finally:
    #         # torch.distributed.barrier()  # All ranks wait before releasing the lock
    #         if distributed.is_main_process():
    #             if path_unlinker and (not os.path.exists(lock_path)):
    #                 print("Not removing lock. Does not exists ", lock_path)
    #             elif path_unlinker:
    #                 lock_path.unlink()
    #                 ExitHandler.instance().remove(path_unlinker)
    #         if distributed.is_enabled_and_multiple_gpus():
    #             torch.distributed.barrier()

    #     status_info = feature_aggregation_updates[str(wsi_fp)]
    #     process_df.loc[
    #         process_df["wsi_path"] == str(wsi_fp), "aggregation_status"
    #     ] = status_info["status"]
    #     if "error" in status_info:
    #         process_df.loc[
    #             process_df["wsi_path"] == str(wsi_fp), "error"
    #         ] = status_info["error"]
    #         process_df.loc[
    #             process_df["wsi_path"] == str(wsi_fp), "traceback"
    #         ] = status_info["traceback"]
    #     process_df.to_csv(process_list, index=False)


    # if distributed.is_enabled_and_multiple_gpus():
    #     torch.distributed.barrier()

    # if distributed.is_main_process():

    #     # summary logging
    #     slides_with_tile_features = len(tiled_and_features_df)
    #     total_slides = len(process_df)
    #     failed_feature_aggregation = process_df[
    #         ~(process_df["aggregation_status"] == "success")
    #     ]
    #     print("=+=" * 10)
    #     print(f"Total number of slides with tile-level features: {slides_with_tile_features}/{total_slides}")
    #     print(f"Failed slide-level feature aggregation: {len(failed_feature_aggregation)}")
    #     print(
    #         f"Completed slide-level feature aggregation: {total_slides - len(failed_feature_aggregation)}"
    #     )
    #     print("=+=" * 10)

    # if distributed.is_enabled():
    #     torch.distributed.destroy_process_group()

if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
