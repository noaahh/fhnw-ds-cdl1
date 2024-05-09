# This file defines the label mappings from the naming schema present in the file names to the actual labels.
from enum import Enum


class MeasurementGroup(Enum):
    NO_GROUP = 0
    GRUPPE1 = 1
    GRUPPE2 = 2
    GRUPPE3 = 3


class Labels(Enum):
    WALKING = "walking"
    SITTING = "sitting"
    STANDING = "standing"
    RUNNING = "running"
    CLIMBING = "climbing"


MAPPING = {
    "_laufen": Labels.WALKING,
    "treppe": Labels.CLIMBING,
    "stehen": Labels.STANDING,

    "walking": Labels.WALKING,
    "sitting": Labels.SITTING,
    "standing": Labels.STANDING,
    "running": Labels.RUNNING,
    "climbing": Labels.CLIMBING
}


def extract_label_from_file_name(file_name: str) -> str | None:
    for key, value in MAPPING.items():
        if key.lower() in file_name.lower():
            return value.value

    return None
