from typing import List, Tuple


def get_las_bounds(filepath: str):
    """
    Отримує мінімальні та максимальні X, Y координати з метаданих LAS-файлу.
    Працює із різними форматами метаданих PDAL.
    """
    import json
    import pdal

    pipeline_json = json.dumps([{"type": "readers.las", "filename": filepath}])
    pipeline = pdal.Pipeline(pipeline_json)
    pipeline.execute()
    metadata = pipeline.metadata

    las_meta = metadata['metadata']['readers.las']

    # Перевірка наявності bbox (на майбутнє)
    if 'bbox' in las_meta and 'native' in las_meta['bbox']:
        bounds = las_meta['bbox']['native']['bbox']
        return bounds['minx'], bounds['miny'], bounds['maxx'], bounds['maxy']

    # Якщо bbox немає, беремо напряму
    return (
        las_meta['minx'],
        las_meta['miny'],
        las_meta['maxx'],
        las_meta['maxy']
    )


def split_area_into_tiles(minx, miny, maxx, maxy, tile_size: int) -> List[Tuple[float, float, float, float]]:
    """
    Розбиває загальну прямокутну область на менші квадрати (тайли).
    """
    tiles = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            # Координати одного тайла
            tiles.append((x, y, x + tile_size, y + tile_size))
            y += tile_size
        x += tile_size
    return tiles
