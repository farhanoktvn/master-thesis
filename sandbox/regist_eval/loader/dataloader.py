import pandas as pd
from pathlib import Path

from loader.sample import Sample


class DataLoader:
    def __init__(self, root_dir, metadata_dir):
        self.root_dir = Path(root_dir)
        self.metadata = pd.read_hdf(metadata_dir)  # read h5 file
        self.samples = self._init_samples(self.root_dir)

    def _init_samples(self, root_dir):
        sample_ids = self.metadata["sample_id"].unique()
        samples = list()
        for sample_id in sample_ids:
            samples.append(
                Sample(
                    sample_id,
                    self.root_dir,
                    self.metadata.loc[self.metadata["sample_id"] == sample_id],
                )
            )
        return samples
