from geopy.distance import geodesic

def lat_long2meter(coord_1, coord_2, unit = 'm'):
    unit = unit.lower()

    if unit == 'm':
        return geodesic(coord_1, coord_2).m
    elif unit == 'km':
        return geodesic(coord_1, coord_2).km
    else:
        raise ValueError(unit)

def legal_coord(coord):
    latitude = coord[0]
    longitude = coord[1]

    if (not len(str(latitude)) == 7) or not (len(str(longitude)) == 8):
        return False
    
    try:
        latitude = int(latitude)
        longitude = int(longitude)
    except:
        return False

    latitude /= 1e5
    longitude /= 1e5

    if (not latitude >= -90.0) or (not latitude <= 90.0) or 
        (not longitude >= -180.0) or (not longitude <= 180.0):
        return False
    
    return True
