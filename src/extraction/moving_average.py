def calc_window_size(window_size_seconds, sampling_rate_hz):
    return int(window_size_seconds * sampling_rate_hz)


def calculate_moving_average(df, column, window_size):
    moving_average = df[column].rolling(window=window_size).mean()
    return moving_average
