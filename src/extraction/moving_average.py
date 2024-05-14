from src.utils import get_env_variable


def calc_window_size(window_size_seconds, sampling_rate_hz=get_env_variable('RESAMPLE_RATE_HZ')):
    return int(window_size_seconds * sampling_rate_hz)


def calculate_moving_average(df, column, window_size):
    moving_average = df[column].rolling(window=window_size).mean()
    return moving_average

    # moving_average = df.loc[segment_mask, column].rolling(window=window_size).mean()
    # segment_features[f'{column}_moving_avg'] = moving_average.tolist()
