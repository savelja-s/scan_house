import json
import pdal

def get_las_bounds(filepath: str):
    """
    Повертає bbox у форматі (minx, miny, maxx, maxy).
    """
    pipeline = pdal.Pipeline(json.dumps([
        {"type": "readers.las", "filename": filepath}
    ]))
    pipeline.execute()
    meta = pipeline.metadata['metadata']['readers.las']
    if 'native' in meta.get('bbox', {}):
        b = meta['bbox']['native']['bbox']
        return b['minx'], b['miny'], b['maxx'], b['maxy']
    return meta['minx'], meta['miny'], meta['maxx'], meta['maxy']

def split_area_into_tiles(minx, miny, maxx, maxy, tile_size: float):
    """
    Розбиває bbox на квадратні тайли.
    """
    tiles = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tiles.append((x, y, min(x+tile_size, maxx), min(y+tile_size, maxy)))
            y += tile_size
        x += tile_size
    return tiles
