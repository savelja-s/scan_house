import json
import pdal

def get_las_bounds(filepath: str):
    """
    Returns the bounding box of the LAS/LAZ file.
    """
    pipeline_json = json.dumps([
        {"type": "readers.las", "filename": filepath}
    ])
    pipeline = pdal.Pipeline(pipeline_json)
    pipeline.execute()
    meta = pipeline.metadata['metadata']['readers.las']

    # If spatial index exists
    if 'bbox' in meta.get('bbox', {}):
        bounds = meta['bbox']['native']['bbox']
        return bounds['minx'], bounds['miny'], bounds['maxx'], bounds['maxy']

    # Fallback to direct min/max
    return meta['minx'], meta['miny'], meta['maxx'], meta['maxy']


def split_area_into_tiles(minx, miny, maxx, maxy, tile_size: float):
    """
    Splits the overall bounding box into square tiles of given size.
    """
    tiles = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tiles.append((
                x,
                y,
                min(x + tile_size, maxx),
                min(y + tile_size, maxy)
            ))
            y += tile_size
        x += tile_size
    return tiles
