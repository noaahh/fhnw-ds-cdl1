import pandas as pd


def _assert_index_is_datetime(data):
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index is not a DateTimeIndex")


def crop_signal(data, start_seconds, end_seconds):
    _assert_index_is_datetime(data)

    total_duration = (data.index.max() - data.index.min()).total_seconds()

    if total_duration < start_seconds + end_seconds:
        raise ValueError("Signal is too short to crop")

    start_time = data.index.min() + pd.Timedelta(seconds=start_seconds)
    end_time = data.index.max() - pd.Timedelta(seconds=end_seconds)

    return data.loc[start_time:end_time]


def resample_signal(data, rate_hz):
    _assert_index_is_datetime(data)

    current_sampling_rate = 1 / (data.index[1] - data.index[0]).total_seconds()
    if current_sampling_rate == rate_hz:
        return data

    if current_sampling_rate < rate_hz:
        raise ValueError(f"Cannot upsample from {current_sampling_rate}Hz to {rate_hz}Hz")

    rate = f"{int(1E6 / rate_hz)}us"
    data = data.resample(rate, origin="start").mean()

    if data.isnull().values.any() or data.isna().values.any():
        data = data.bfill()  # Backfill NaNs

    return data


def create_segments(file_data, segment_size_seconds, overlap_seconds):
    _assert_index_is_datetime(file_data)

    segment_length = pd.Timedelta(seconds=segment_size_seconds)
    overlap_length = pd.Timedelta(seconds=overlap_seconds)
    segments = []

    start_time = file_data.index.min()
    end_time = file_data.index.max()

    min_segment_length = max(segment_length, overlap_length)
    if end_time - start_time < min_segment_length:
        raise ValueError(f"Signal is too short to segment: {end_time - start_time} < {min_segment_length}")

    while start_time + segment_length <= end_time:  # TODO: check ways to use overlap in a better way in case the we are running out of data at the end of the signal
        segment_end = start_time + segment_length
        segment = file_data.loc[start_time:segment_end]
        segments.append(segment)

        start_time = segment_end - overlap_length

    segment_lengths = set([len(segment) for segment in segments])
    if len(segment_lengths) != 1:
        raise ValueError(
            f"Segments are not of equal length: {segment_lengths}. Number of segments: {len(segments)}. Start time: {start_time}. End time: {end_time}")

    return segments
