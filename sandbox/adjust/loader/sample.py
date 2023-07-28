from loader.run import Run


class Sample:
    def __init__(self, sample_id, root_dir, sample_df):
        self.sample_id = sample_id
        self.root_dir = root_dir
        self._init_data(sample_df)

    def _init_data(self, sample_df):
        run_ids = sample_df["run_id"].unique()

        runs = list()
        orients = list()
        whites = list()
        darks = list()

        for run_id in run_ids:
            # remove run_id with run_group equal to orient white or dark
            run_group = sample_df.loc[sample_df["run_id"] == run_id]["run_group"].tolist()[0]
            if run_group == "orient":
                orients.append(
                    Run(
                        self.sample_id,
                        run_id,
                        self.root_dir,
                        sample_df.loc[sample_df["run_id"] == run_id],
                    )
                )
            elif run_group == "white":
                whites.append(
                    Run(
                        self.sample_id,
                        run_id,
                        self.root_dir,
                        sample_df.loc[sample_df["run_id"] == run_id],
                    )
                )
            elif run_group == "dark":
                darks.append(
                    Run(
                        self.sample_id,
                        run_id,
                        self.root_dir,
                        sample_df.loc[sample_df["run_id"] == run_id],
                    )
                )
            else:
                runs.append(
                    Run(
                        self.sample_id,
                        run_id,
                        self.root_dir,
                        sample_df.loc[sample_df["run_id"] == run_id],
                    )
                )
        self.runs = runs
        self.orients = orients
        self.whites = whites
        self.darks = darks

    def __repr__(self) -> str:
        return f"Sample({self.sampl})"
