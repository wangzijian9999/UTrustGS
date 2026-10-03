"""DTU 开发表面诊断；不是官方 MATLAB 结果的等价复现。

输入必须是三角网格在 DTU 标定参考系中的毫米坐标，不接受高斯中心冒充表面。
按面积随机采样、0.2mm 邻域降采样；随机数和 MeshSupSamp 与官方不同。
距离分块、ObsMask、平面和严格小于20mm的统计依据本地官方 MATLAB 源码。
同时保留未截断统计与被排除比例；本工具不判定基线资格或几何非劣。
CLI 仅接受显式 development/scan1 清单，且校验每个输入 SHA256。
坐标来源、表面提取独立性和开发集角色仍需外部审计，哈希不能证明这些声明。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.io import loadmat
from scipy.spatial import cKDTree

from evaluate_repair_reliability import sha256_file


def require(condition, message):
    if not condition:
        raise ValueError(message)


def points_array(points):
    points = np.asarray(points, dtype=np.float64)
    require(points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all(),
            "点必须是有限的 N×3 数组")
    return points


def sample_triangle_surface(vertices, faces, count, seed):
    vertices = points_array(vertices)
    faces = np.asarray(faces)
    require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
            and np.issubdtype(faces.dtype, np.integer), "必须提供非空三角面，不能用高斯中心代替表面")
    require(faces.min() >= 0 and faces.max() < len(vertices), "三角面索引越界")
    require(type(count) is int and 1 <= count <= 2_000_000, "采样数必须为1至2000000的整数")
    require(type(seed) is int and seed >= 0, "seed必须为非负整数")
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    require(np.isfinite(area).all() and area.sum() > 0, "表面必须具有有限的正面积")
    random = np.random.default_rng(seed)
    selected = triangles[random.choice(len(triangles), count, p=area / area.sum())]
    coordinates = random.random((count, 2))
    square_root = np.sqrt(coordinates[:, :1])
    return ((1 - square_root) * selected[:, 0]
            + square_root * (1 - coordinates[:, 1:]) * selected[:, 1]
            + square_root * coordinates[:, 1:] * selected[:, 2])


def reduce_surface_samples(points, radius, seed):
    points = points_array(points)
    require(np.isfinite(radius) and radius > 0, "邻域半径必须为正")
    keep = np.ones(len(points), dtype=bool)
    tree = cKDTree(points)
    for index in np.random.default_rng(seed).permutation(len(points)):
        if keep[index]:
            keep[tree.query_ball_point(points[index], radius)] = False
            keep[index] = True
    return points[keep]


def bounded_distances(target, source, bounding_box, cell_size=60.0):
    """对应 MaxDistCP 的分块及缺失值语义；有邻居的距离不强截到60mm。"""
    target, source = points_array(target), points_array(source)
    bounding_box = np.asarray(bounding_box, dtype=np.float64)
    require(bounding_box.shape == (2, 3) and np.isfinite(bounding_box).all()
            and (bounding_box[1] >= bounding_box[0]).all(), "BB必须为有序2×3数组")
    require(np.isfinite(cell_size) and cell_size > 0, "cell_size必须为正")
    distances = np.full(len(source), cell_size, dtype=np.float64)
    cells = np.floor((source - bounding_box[0]) / cell_size)
    cell_range = np.floor((bounding_box[1] - bounding_box[0]) / cell_size)
    active = np.all((cells >= 0) & (cells <= cell_range), axis=1)
    for cell in np.unique(cells[active], axis=0):
        selected_source = active & np.all(cells == cell, axis=1)
        low = bounding_box[0] + cell * cell_size
        selected_target = np.all((target >= low - cell_size) & (target < low + 2 * cell_size), axis=1)
        if selected_target.any():
            distances[selected_source] = cKDTree(target[selected_target]).query(source[selected_source], k=1)[0]
    return distances


def observation_membership(points, bounding_box, resolution, observation_mask):
    points = points_array(points)
    bounding_box = np.asarray(bounding_box, dtype=np.float64)
    require(bounding_box.shape == (2, 3) and np.isfinite(bounding_box).all(), "无效BB")
    require(np.isfinite(resolution) and resolution > 0, "Res必须为正")
    observation_mask = np.asarray(observation_mask)
    require(observation_mask.ndim == 3 and np.isfinite(observation_mask).all(), "ObsMask必须为有限三维数组")
    # 先加1，再使用MATLAB半整数远离零规则；不能直接使用numpy.rint。
    coordinates = (points - bounding_box[0]) / resolution + 1
    rounded = np.sign(coordinates) * np.floor(np.abs(coordinates) + 0.5)
    inside = np.all((rounded >= 1) & (rounded <= observation_mask.shape), axis=1)
    indices = rounded[inside].astype(np.int64) - 1
    result = np.zeros(len(points), dtype=bool)
    result[inside] = observation_mask[tuple(indices.T)].astype(bool)
    return result


def distance_statistics(distances, domain):
    distances, domain = np.asarray(distances), np.asarray(domain)
    require(distances.ndim == 1 and np.isfinite(distances).all() and (distances >= 0).all(), "无效距离")
    require(domain.dtype == bool and domain.shape == distances.shape, "评价域必须为同形布尔数组")
    selected = distances[domain]
    retained = selected[selected < 20.0]
    return {"all_count": len(distances), "domain_count": len(selected), "retained_count": len(retained),
            "outside_domain_count": int((~domain).sum()), "outlier_count": len(selected) - len(retained),
            "outlier_fraction": float(np.mean(selected >= 20)) if len(selected) else None,
            "untrimmed_mean_mm": float(selected.mean()) if len(selected) else None,
            "trimmed_mean_mm": float(retained.mean()) if len(retained) else None,
            "trimmed_median_mm": float(np.median(retained)) if len(retained) else None,
            "trimmed_sample_variance_mm2": float(retained.var(ddof=1)) if len(retained) > 1 else (0.0 if len(retained) else None)}


def evaluate_surface(vertices, faces, reference, bounding_box, resolution, observation_mask,
                     ground_plane, sample_count, seed):
    surface = reduce_surface_samples(sample_triangle_surface(vertices, faces, sample_count, seed), 0.2, seed)
    reference = points_array(reference)
    require(len(reference) > 0, "扫描参考不能为空")
    ground_plane = np.asarray(ground_plane, dtype=np.float64).reshape(-1)
    require(ground_plane.shape == (4,) and np.isfinite(ground_plane).all()
            and np.linalg.norm(ground_plane[:3]) > 0, "无效地面平面")
    accuracy_domain = observation_membership(surface, bounding_box, resolution, observation_mask)
    completeness_domain = reference @ ground_plane[:3] + ground_plane[3] > 0
    return {"status": "DIAGNOSTIC_ONLY_NOT_OFFICIAL_EQUIVALENCE", "sampled_count": sample_count,
            "reduced_surface_count": len(surface),
            "accuracy": distance_statistics(bounded_distances(reference, surface, bounding_box), accuracy_domain),
            "completeness": distance_statistics(bounded_distances(surface, reference, bounding_box), completeness_domain),
            "sampling": "面积加权三角形随机采样+0.2mm随机贪心降采样；NumPy PCG64；非官方MeshSupSamp",
            "limitations": ["未与原生MATLAB数值对照", "没有自动配准或ICP；毫米坐标系由上游保证",
                            "扫描不用于表面提取；输入来源声明需独立审计", "截断均值必须与域大小、异常比例同时解释"]}


def run_manifest(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema_version") == 1 and manifest.get("data_role") == "development"
            and manifest.get("scan") == 1, "仅允许当前已开放的DTU开发scan1")
    require(manifest.get("representation") == "triangle_surface"
            and manifest.get("coordinate_frame") == "dtu_cal18_reference_mm", "必须声明三角表面及毫米参考系")
    for name in ("extraction_provenance", "coordinate_provenance"):
        require(isinstance(manifest.get(name), str) and bool(manifest[name].strip()), name + "缺失")
    paths = {}
    for name in ("mesh", "reference", "observation_mask", "ground_plane"):
        entry = manifest["inputs"][name]
        path = (manifest_path.parent / entry["path"]).resolve()
        require(sha256_file(path) == entry["sha256"], name + "哈希不匹配")
        paths[name] = path
    mesh = PlyData.read(str(paths["mesh"]))
    require("face" in mesh and "vertex_indices" in mesh["face"].data.dtype.names, "输入缺少三角面")
    faces = mesh["face"].data["vertex_indices"]
    require(len(faces) > 0 and all(len(face) == 3 for face in faces), "仅接受非空三角网格")
    vertices = np.column_stack([mesh["vertex"].data[name] for name in ("x", "y", "z")])
    reference_ply = PlyData.read(str(paths["reference"]))
    reference = np.column_stack([reference_ply["vertex"].data[name] for name in ("x", "y", "z")])
    masks = loadmat(paths["observation_mask"])
    result = evaluate_surface(vertices, np.stack(faces), reference, masks["BB"], float(masks["Res"].item()),
                              masks["ObsMask"], loadmat(paths["ground_plane"])["P"],
                              manifest["sample_count"], manifest["seed"])
    result.update(manifest_sha256=sha256_file(manifest_path), evaluator_sha256=sha256_file(Path(__file__)),
                  inputs=manifest["inputs"], declared_extraction_provenance=manifest["extraction_provenance"],
                  declared_coordinate_provenance=manifest["coordinate_provenance"],
                  data_role=manifest["data_role"], scan=manifest["scan"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    require(not arguments.output.exists(), "禁止覆盖已有报告")
    result = run_manifest(arguments.manifest)
    with arguments.output.open("x") as output:
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
