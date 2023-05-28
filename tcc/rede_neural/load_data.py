import numpy as np
import tensorflow as tf
import mne
from tcc.utils.feats import Feats


class DataLoader():
    def __init__(self, pathname: str, feats: Feats) -> None:
        self.pathname = pathname
        self._feats = feats

    def _emwa(self, data: np.ndarray, alpha=0.1):
        """
        Fast emwa mostly taken from:
        https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
        :param data: Assuming at least three axes, with the first definitely the batch axis.
        :param axis:
        :return:
        """
        alpha_rev = 1 - alpha
        n = data.shape[-1]
        pows = alpha_rev ** (np.arange(n + 1))

        scale_arr = 1 / pows[:-1]
        offset = data[..., 0, np.newaxis] * pows[1:]
        pw0 = alpha * alpha_rev ** (n - 1)

        mult = data * pw0 * scale_arr
        cumsums = mult.cumsum(axis=-1)
        out = offset + cumsums * scale_arr[::-1]
        return out

    def _exp_moving_whiten(self, data: np.ndarray, factor_new=0.01):
        """
        Exponentially whitening
        Some code in this function taken from:
        https://github.com/robintibor/braindecode/blob/master/braindecode/datautil/signalproc.py
        :param data:
        :param factor_new:
        :return:
        """
        meaned = self._emwa(data, alpha=factor_new).mean()
        demeaned = data - meaned
        squared = demeaned * demeaned
        sq_mean = self._emwa(squared, alpha=factor_new).mean()
        return demeaned / np.maximum(1e-4, np.sqrt(sq_mean))

    def _arrays(self, path):
        print("reading file: " + path)
        raw = mne.io.read_raw_fif(str(path), preload=True)
        # raw.plot_psd()
        picks = mne.pick_types(raw.info, eog=False, stim=False, meg=False, eeg=True)
        events = mne.find_events(raw)
        epochs = mne.Epochs(raw, events, tmin=self._feats.t_min,
                            tmax=self._feats.t_min + self._feats.t_len - 1 / raw.info['sfreq'],
                            preload=True, picks=picks, baseline=None)
        x = epochs.get_data()
        # epochs.plot_psd()
        return x.transpose((0, 2, 1)), tf.keras.utils.to_categorical(epochs.events[:, -1] - 1, num_classes=self._feats.num_classes)
        # return self._exp_moving_whiten(x).transpose((0, 2, 1)), \
        #     tf.keras.utils.to_categorical(epochs.events[:, -1] - 1, self._feats.num_classes)

    def _read_files(self):
        print('reading files...')

        try:
            # data = np.array(*tuple(map(self._arrays,
            #                            [self.pathname + 'A0{}{}.raw.fif'.format(j, i)
            #                             for j in range(1, 10, 1) for i in 'TE']))

            train = list(map(self._arrays, [self.pathname + 'A0{}{}.raw.fif'.format(j, i)
                                            for j in range(1, 9, 1) for i in 'T']))
            #    for j in range(1, 6, 1) for i in 'TE']))
            # for j in range(1, 10, 1) for i in 'TE']))

            test = list(map(self._arrays, [self.pathname + 'A0{}{}.raw.fif'.format(j, i)
                                           for j in range(9, 10, 1) for i in 'E']))
            #    for j in range(1, 6, 1) for i in 'TE']))
            # for j in range(1, 10, 1) for i in 'TE']))
            # x_data = np.array([])
            # y_data = np.array([])

            return train, test
            # return x_data, y_data
            # return [self._arrays(self.pathname + 'A0{}{}.raw.fif'.format(j, i)) for j in range(9) for i in 'TE']
        except Exception as e:
            print('Error processing :', e)
            pass
