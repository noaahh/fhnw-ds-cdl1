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


LABEL_MAPPING = {
    "walking": 0,
    "running": 1,
    "sitting": 2,
    "standing": 3,
    "climbing": 4
}

LABEL_COLUMN = 'label'
NUM_CLASSES = len(LABEL_MAPPING)

MAPPING = {
    "_laufen": Labels.WALKING,
    "treppe": Labels.CLIMBING,
    "stehen": Labels.STANDING,
    "rennen": Labels.RUNNING,
    "sitzen": Labels.SITTING,
    "treppenlaufen": Labels.CLIMBING,

    "walking": Labels.WALKING,
    "sitting": Labels.SITTING,
    "standing": Labels.STANDING,
    "running": Labels.RUNNING,
    "climbing": Labels.CLIMBING,
    "jogging": Labels.RUNNING
}


def extract_label_from_file_name(file_name: str) -> str | None:
    for key, value in MAPPING.items():
        if key.lower() in file_name.lower():
            return value.value

    return None
