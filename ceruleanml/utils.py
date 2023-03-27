# Single Source of Truth for categorical classes
# and their respective Human Readable (hr) text and Color Codes (cc) photopea designations
class_dict = {
    "background": {"hr": "Background", "cc": (0, 255, 255)},
    "infra_slick": {"hr": "Infrastructure", "cc": (0, 0, 255)},
    "natural_seep": {"hr": "Natural Seep", "cc": (0, 255, 0)},
    "coincident_vessel": {"hr": "Coincident Vessel", "cc": (255, 0, 0)},
    "recent_vessel": {"hr": "Recent Vessel", "cc": (255, 255, 0)},
    "old_vessel": {"hr": "Old Vessel", "cc": (255, 0, 255)},
    "ambiguous": {"hr": "Ambiguous", "cc": (255, 255, 255)},
}
class_list = list(class_dict.keys())
