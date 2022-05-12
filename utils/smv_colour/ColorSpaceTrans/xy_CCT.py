import numpy as np

def xy2CCT_mccamy1992(xy):
    """[Returns the correlated colour temperature]

    Args:
        xy ([type]): [*xy* chromaticity coordinates.]

    Returns:
        [type]: [Correlated colour temperature]
    """
    x, y = xy[..., 0], xy[...,1]
    n = (x - 0.3320) / (y - 0.1858)
    cct = -449 * np.power(n, 3) + 3525 * np.power(n, 2) - 6823.3 * n + 5520.33
    
    return cct