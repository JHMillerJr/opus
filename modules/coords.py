def ra_to_degrees(hours, minutes, seconds):
    """
    Convert Right Ascension (RA) from hh mm ss to degrees.
    """
    return 15 * (hours + minutes / 60 + seconds / 3600)


def dec_to_degrees(degrees, arcminutes, arcseconds):
    """
    Convert Declination (Dec) from dd mm ss to decimal degrees.
    Handles positive and negative declinations.
    """
    sign = -1 if degrees < 0 else 1
    degrees = abs(degrees)

    return sign * (
        degrees + arcminutes / 60 + arcseconds / 3600
    )


# example
ra_deg = ra_to_degrees(11, 38, 3.70)
dec_deg = dec_to_degrees(3, 14, 58.3)

print("RA:", ra_deg)
print("Dec:", dec_deg)