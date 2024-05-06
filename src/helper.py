import os


def get_env_variable(variable_name):
    value = os.getenv(variable_name)
    if value is None:
        raise ValueError(f"Environment variable {variable_name} must be set")

    if value.isdigit():
        return int(value)

    return value
