import os
import json
import pdal
from datetime import datetime


def prepare_las_file_by_icp(args):
    filter_by_icp(args.input)


def filter_by_icp(in_filename):
    """
    Фільтрація LAS/LAZ файлу через SMRF, виділення ground та coplanar planes, збереження результатів.

    :param in_filename: шлях до вхідного LAS/LAZ файлу
    :return: шляхи до smrf та фінального ground+planes LAS файлів
    """
    # Підготовка імен файлів та директорій
    name, ext = os.path.splitext(os.path.basename(in_filename))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join("data", "prepare_las_file", name)
    os.makedirs(work_dir, exist_ok=True)

    smrf_file = os.path.join(work_dir, f"{name}_smrf_{ts}{ext}")
    out_file = os.path.join(work_dir, f"{name}_GroundAndPlanes_{ts}{ext}")

    # 1-й пайплайн: ground filter and sample
    pipeline1 = [
        {"type": "readers.las", "filename": in_filename},
        {"type": "filters.assign", "assignment": "Classification[0:255]=0"},
        {"type": "filters.smrf"},
        {"type": "filters.sample", "radius": 0.5},
        {"type": "writers.las", "filename": smrf_file}
    ]
    p1 = pdal.Pipeline(json.dumps(pipeline1))
    p1.execute()

    # 2-й пайплайн: extract planes, merge with ground
    pipeline2 = [
        {"type": "readers.las", "filename": smrf_file},
        {"type": "filters.hag_dem","raster": "autzen_dem.tif"},
        {"type": "filters.range", "limits": "HeightAboveGround[3:15]"},
        {"type": "filters.approximatecoplanar", "knn": 16, "thresh1": 50, "thresh2": 6},
        {"type": "filters.range", "limits": "Coplanar[1:1]"},
        {"type": "filters.outlier", "mean_k": 32},
        {"type": "filters.range", "limits": "Classification![7:7]"},
        {"type": "filters.assign", "assignment": "Classification[1:1]=6", "tag": "PLANES"},
        {"type": "readers.las", "filename": smrf_file, "tag": "GROUND_INPUT"},
        {"type": "filters.range", "limits": "Classification[2:2]", "tag": "GROUND"},
        {"type": "filters.merge", "inputs": ["GROUND", "PLANES"]},
        {"type": "writers.las", "filename": out_file}
    ]
    p2 = pdal.Pipeline(json.dumps(pipeline2))
    p2.execute()

    print(f"SMRF-файл: {smrf_file}")
    print(f"Фінальний файл: {out_file}")

    return smrf_file, out_file
