"""CPU regression for excluding a target pose from LLFF support views."""

import argparse
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import torch

from run_bayesgs_diff import LLFFDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llff-root", type=Path, required=True)
    args = parser.parse_args()
    scene_dir = args.llff_root / "flower"
    dataset = LLFFDataset.__new__(LLFFDataset)
    dataset.scene_dir = scene_dir
    dataset.input_data_dir = scene_dir / "images_8"
    dataset.images_dir = dataset.input_data_dir
    dataset.image_files = dataset._collect_image_files_from_dir(dataset.images_dir)
    dataset.sparse_image_files = dataset._collect_image_files_from_dir(scene_dir / "images")
    pose_count = int(np.load(scene_dir / "poses_bounds.npy", mmap_mode="r").shape[0])
    assert pose_count == len(dataset.image_files) == len(dataset.sparse_image_files)
    dataset.poses = torch.eye(4).repeat(pose_count, 1, 1)
    dataset.poses[:, 0, 3] = torch.arange(pose_count, dtype=torch.float32)
    dataset.pose_indices_for_images = dataset._build_pose_index_mapping(dataset.image_files)
    dataset.pose_indices_for_sparse_images = dataset._build_pose_index_mapping(dataset.sparse_image_files)
    assert dataset.image_files[0].name == "image000.png"
    assert dataset.sparse_image_files[0].name == "IMG_2962.JPG"
    assert int(dataset.pose_indices_for_images[0]) == int(dataset.pose_indices_for_sparse_images[0]) == 0

    nearest = dataset._build_nearest_sparse_indices_by_target()
    assert 0 not in nearest[0]
    assert len(nearest[0]) == pose_count - 1
    dataset.effective_num_sparse_views = 3
    global_indices = dataset._materialize_scene_global_support_template_indices(
        target_pose_index=0, ordered_candidate_sparse_indices=[0, 1, 2, 3]
    )
    fallback_indices = dataset._materialize_scene_global_support_template_indices(
        target_pose_index=0, ordered_candidate_sparse_indices=[0]
    )
    assert global_indices.tolist() == [1, 2, 3]
    assert fallback_indices.tolist() == [1, 2, 3]

    dataset.indices = [0]
    dataset.use_phase1_3dgs_conditions = False
    dataset.is_train = False
    dataset.strict_geometry_contract = True
    dataset.image_size = (8, 8)
    dataset.camera_intrinsics = torch.eye(3).repeat(pose_count, 1, 1)
    dataset.condition_mode = "corrupt"
    dataset.corruption_ratio = 0.0
    dataset.support_template_stage_tag = "cpu_regression"
    loaded_paths = []

    def load_image(path: Path) -> torch.Tensor:
        loaded_paths.append(path)
        return torch.zeros(3, 8, 8)

    dataset._load_image = load_image
    dataset._get_cached_depth_map_for_image_file = lambda *args, **kwargs: torch.ones(1, 8, 8)
    dataset._get_cached_depth_map_for_image_index = lambda *args: torch.ones(1, 8, 8)
    dataset._add_corruption = lambda image, ratio: (image.clone(), torch.zeros(1, 8, 8))
    dataset._degrade_online_conditions = lambda rgb, depth: (rgb, depth)
    dataset._depth_to_normal = lambda depth: torch.zeros(3, 8, 8)
    sample = dataset[0]
    assert sample["sparse_images"].shape[0] == 3
    assert loaded_paths == [dataset.image_files[0]] + dataset.sparse_image_files[1:4]

    dataset.support_template_stage_tag = "stage1_pretrain"
    dataset.support_template_scope = "target_local"
    dataset.verbose_dataset_logs = False
    invalid_templates = {0: [{"sparse_indices": [0, 1, 2]}]}
    for cache_kind in ("template", "episode"):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "cache.pt"
            cache_payload = {
                "cache_version": 4 if cache_kind == "episode" else 2,
                "support_template_stage_tag": dataset.support_template_stage_tag,
                "support_template_scope": dataset.support_template_scope,
                "effective_num_sparse_views": dataset.effective_num_sparse_views,
                "target_image_names": [path.name for path in dataset.image_files],
                "sparse_image_names": [path.name for path in dataset.sparse_image_files],
                "support_templates_by_target": invalid_templates,
            }
            torch.save(cache_payload, cache_path)
            original_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()
            if cache_kind == "episode":
                dataset._phase1_support_template_bank = {0: [{"sparse_indices": [1, 2, 3]}]}
                dataset._resolve_phase1_episode_bank_metadata_file = lambda: cache_path
                load_cache = dataset._load_phase1_episode_bank_metadata
            else:
                load_cache = lambda: dataset._load_support_template_bank(cache_path)
            try:
                load_cache()
            except RuntimeError as error:
                assert "self-view pose" in str(error), error
            else:
                raise AssertionError(f"{cache_kind} self-view cache was accepted")
            assert hashlib.sha256(cache_path.read_bytes()).hexdigest() == original_hash

    valid_templates = {0: [{"sparse_indices": [1, 2, 3]}]}
    with tempfile.TemporaryDirectory() as temporary_directory:
        cache_path = Path(temporary_directory) / "clean.pt"
        clean_payload = {
            "cache_version": 2,
            "support_template_stage_tag": dataset.support_template_stage_tag,
            "support_template_scope": dataset.support_template_scope,
            "effective_num_sparse_views": dataset.effective_num_sparse_views,
            "target_image_names": [path.name for path in dataset.image_files],
            "sparse_image_names": [path.name for path in dataset.sparse_image_files],
            "support_templates_by_target": valid_templates,
        }
        torch.save(clean_payload, cache_path)
        original_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        assert dataset._load_support_template_bank(cache_path) == valid_templates
        assert hashlib.sha256(cache_path.read_bytes()).hexdigest() == original_hash

    for invalid_templates in (
        {-1: [{"sparse_indices": [1, 2, 3]}]},
        {0: [{"sparse_indices": [-1, 2, 3]}]},
        {0: [{"sparse_indices": [pose_count, 2, 3]}]},
    ):
        try:
            dataset._validate_support_template_poses(invalid_templates, "CPU regression")
        except RuntimeError as error:
            assert "out of range" in str(error), error
        else:
            raise AssertionError(f"Invalid cache index was accepted: {invalid_templates}")

    assert not torch.cuda.is_initialized()
    print("PASS: flower PNG/JPG same pose excluded in nearest, global, fallback, stubbed __getitem__; invalid caches rejected unchanged, clean cache loaded, bounds checked; CPU only")


if __name__ == "__main__":
    main()
