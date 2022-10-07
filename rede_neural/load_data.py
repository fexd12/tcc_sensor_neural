import mne
import pandas as pd
from glob import glob


class DataLoader():
    def __init__(self, filename: str, sfreq: int) -> None:
        self.filename = filename
        self.sfreq = sfreq
        self.raw: mne.io.RawArray = self.load_data()

    def load_data(self) -> mne.io.RawArray:
        return self._load(plot_sensors=False, plot_raw=False,
                          plot_raw_psd=False)

    def _read_files(self):
        # merging two csv files
        # nao esta fazendo o merge dos csvs
        df = pd.concat(
            map(pd.read_csv, glob(self.filename), ";"))

        return df

    def _load(self, plot_sensors=True, plot_raw=True,
              plot_raw_psd=True) -> mne.io.RawArray:
        """Load in recorder data files."""

        # explica isso depois

        csv = self._read_files()

        ch_names = csv.columns.tolist()

        # verificar o valor da frequencia
        info = mne.create_info(ch_names=ch_names, sfreq=self.sfreq, ch_types="stim")

        raw = mne.io.RawArray(csv.transpose(), info)

        # load channel locations
        print('Loading Channel Locations')
        if plot_sensors:
            raw.plot_sensors(show_names='True')

        # Plot raw data
        if plot_raw:
            raw.plot(n_channels=len(ch_names), block=True)

            # plot raw psd
        if plot_raw_psd:
            raw.plot_psd(fmin=.1, fmax=100)

        return raw
