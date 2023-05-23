from loader.run import Run


class Sample:
    def __init__(self, sample_id, root_dir, sample_df):
        self.sampl = sample_id
        self.root_dir = root_dir
        self.runs = self._init_runs(sample_df)

    def _init_runs(self, sample_df):
        run_ids = sample_df["run_id"].unique()
        runs = list()
        for run_id in run_ids:
            # remove run_id with run_group equal to orient white or dark
            if sample_df.loc[sample_df["run_id"] == run_id]["run_group"].tolist()[0] in [
                "orient",
                "white",
                "dark",
            ]:
                continue
            runs.append(
                Run(
                    run_id,
                    self.root_dir,
                    sample_df.loc[sample_df["run_id"] == run_id],
                )
            )
        return runs

    def __repr__(self) -> str:
        return f"Sample({self.sampl})"
