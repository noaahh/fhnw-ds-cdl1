import logging

import pandas as pd

from src.helper import get_env_variable

logger = logging.getLogger(__name__)


class BaseProcessor:
    def __init__(self, measurement_file):
        self.measurement_file = measurement_file

    def preprocess(self, data):
        data = data.drop(columns=['seconds_elapsed'])
        data['time'] = pd.to_datetime(data['time'], unit='ns')
        data = data.set_index('time', drop=True)

        return data

    def crop(self, data):
        total_duration = (data.index.max() - data.index.min()).total_seconds()

        start_seconds, end_seconds = get_env_variable("START_CROP_SECONDS"), get_env_variable("END_CROP_SECONDS")

        if total_duration < start_seconds + end_seconds:
            raise ValueError("Signal is too short to crop")

        start_time = data.index.min() + pd.Timedelta(seconds=start_seconds)
        end_time = data.index.max() - pd.Timedelta(seconds=end_seconds)

        return data.loc[start_time:end_time]

    def resample(self, data):
        rate_hz = get_env_variable("RESAMPLE_RATE_HZ")

        rate = f"{int(1E6 / rate_hz)}us"
        data = data.resample(rate, origin="start").mean()  # TODO: check if mean is the best way to interpolate the data

        if data.isnull().values.any() or data.isna().values.any():
            logger.warning(f"Warning: NaNs found in resampled data for {self.measurement_file}")
            data = data.bfill()  # Backfill NaNs

        return data

    def segment(self, data):
        segment_size_seconds = get_env_variable("SEGMENT_SIZE_SECONDS")
        overlap_seconds = get_env_variable("OVERLAP_SECONDS")

        segment_length = pd.Timedelta(seconds=segment_size_seconds)
        overlap_length = pd.Timedelta(seconds=overlap_seconds)
        segments = []

        start_time = data.index.min()
        end_time = data.index.max()

        while start_time + segment_length <= end_time:  # TODO: check ways to use overlap in a better way in case the we are running out of data at the end of the signal
            segment_end = start_time + segment_length
            segment = data.loc[start_time:segment_end]
            segments.append(segment)

            start_time = segment_end - overlap_length

        segment_lengths = set([len(segment) for segment in segments])
        if len(segment_lengths) != 1:
            raise ValueError("Segments are not of equal length")

        return segments

    def extract(self, segment):

        return segment

    def scale(self, segment):

        return segment

    def process(self, data):
        preprocessed_data = self.preprocess(data)
        cropped_data = self.crop(preprocessed_data)
        resampled_data = self.resample(cropped_data)

        segments = self.segment(resampled_data)
        segments = [self.extract(segment) for segment in segments]
        segments = [self.scale(segment) for segment in segments]

        if len(segments) == 0:
            raise ValueError("No segments found")

        return segments
