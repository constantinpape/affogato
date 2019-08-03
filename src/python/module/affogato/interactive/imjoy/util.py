import numpy as np


def parse_geojson(geojson, shape):
    """ Parse input dicts in geo json format
        and return input for ``InteractiveMWS.update_seeds``
    """
    # we always expect to get a 'FeatureCollection'
    if geojson.get("type", "") != "FeatureCollection":
        raise ValueError("Expect to get a geojson with FeatureCollection")

    seed_coordinates = {}
    annotations = geojson['features']

    # iterate over the annotations and get the coords
    for annotation in annotations:
        # get the seed id from the properties.
        seed_name = annotation['properties']['name']
        coords = annotation['geometry']['coordinates']
        geo_type = annotation['geometry']['type']
        if geo_type == 'Point':
            coords = [coords]
        if seed_name in seed_coordinates:
            seed_coordinates[seed_name].extend(coords)
        else:
            seed_coordinates[seed_name] = coords

    seed_coordinates = {int(seed_id): tuple(np.array([coord[1-i] if i == 1 else shape[1 - i] - coord[1 - i]
                                                      for coord in coords], dtype='uint64')
                        for i in range(2))
                        for seed_id, coords in seed_coordinates.items()}

    return seed_coordinates
