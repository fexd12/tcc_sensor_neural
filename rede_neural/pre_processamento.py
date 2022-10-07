import numpy as np
from keras.utils import to_categorical
from mne import (Epochs, concatenate_epochs, find_events, pick_channels,
                 pick_types, viz)
from mne.time_frequency import tfr_morlet
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from feats import Feats
from load_data import DataLoader


class PreProcessamento(DataLoader):
    def __init__(self, feats: Feats, filename: str, sfreq: int) -> None:
        DataLoader.__init__(self, filename, sfreq)
        self._feats = feats
        self.event_id = {'Target': 1, 'Standard': 2}

        epochs = concatenate_epochs(self.pre_process(
            filter_data=False
        ))
        self.feature_engineer(epochs,
                              frequency_domain=True
                              )

    def __mastoidReref(self):
        ref_idx = pick_channels(self.raw.info['ch_names'])
        eeg_idx = pick_types(self.raw.info, eeg=True)
        self.raw._data[eeg_idx, :] = self.raw._data[eeg_idx, :] - self.raw._data[ref_idx, :] * .5

    def __grattonEmcpRaw(self):
        raw_eeg = self.raw.copy().pick_types(eeg=True)[:][0]
        raw_eog = self.raw.copy().pick_types(eog=True)[:][0]
        b = np.linalg.solve(np.dot(raw_eog, raw_eog.T), np.dot(raw_eog, raw_eeg.T))
        eeg_corrected = (raw_eeg.T - np.dot(raw_eog.T, b)).T
        raw_new = self.raw.copy()
        raw_new._data[pick_types(self.raw.info, eeg=True), :] = eeg_corrected
        return raw_new

    def __grattonEmcpEpochs(self, epochs: Epochs):
        '''
        # Correct EEG data for EOG artifacts with regression
        # INPUT - MNE epochs object (with eeg and eog channels)
        # OUTPUT - MNE epochs object (with eeg corrected)
        # After: Gratton,Coles,Donchin, 1983
        # -compute the ERP in each condition
        # -subtract ERP from each trial
        # -subtract baseline (mean over all epoch)
        # -predict eye channel remainder from eeg remainder
        # -use coefficients to subtract eog from eeg
        '''

        event_names = ['A_error', 'B_error']
        i = 0
        for key, value in sorted(epochs.event_id.items(), key=lambda x: (x[1], x[0])):
            event_names[i] = key
            i += 1

        # select the correct channels and data
        eeg_chans = pick_types(epochs.info, eeg=True, eog=False)
        eog_chans = pick_types(epochs.info, eeg=False, eog=True)
        original_data = epochs._data

        # subtract the average over trials from each trial
        rem = {}
        for event in event_names:
            data = epochs[event]._data
            avg = np.mean(epochs[event]._data, axis=0)
            rem[event] = data-avg

        # concatenate trials together of different types
        # then put them all back together in X (regression on all at once)
        allrem = np.concatenate([rem[event] for event in event_names])

        # separate eog and eeg
        X = allrem[:, eeg_chans, :]
        Y = allrem[:, eog_chans, :]

        # subtract mean over time from every trial/channel
        X = (X.T - np.mean(X, 2).T).T
        Y = (Y.T - np.mean(Y, 2).T).T

        # move electrodes first
        X = np.moveaxis(X, 0, 1)
        Y = np.moveaxis(Y, 0, 1)

        # make 2d and compute regression
        X = np.reshape(X, (X.shape[0], np.prod(X.shape[1:])))
        Y = np.reshape(Y, (Y.shape[0], np.prod(Y.shape[1:])))
        b = np.linalg.solve(np.dot(Y, Y.T), np.dot(Y, X.T))

        # get original data and electrodes first for matrix math
        raw_eeg = np.moveaxis(original_data[:, eeg_chans, :], 0, 1)
        raw_eog = np.moveaxis(original_data[:, eog_chans, :], 0, 1)

        # subtract weighted eye channels from eeg channels
        eeg_corrected = (raw_eeg.T - np.dot(raw_eog.T, b)).T

        # move back to match epochs
        eeg_corrected = np.moveaxis(eeg_corrected, 0, 1)

        # copy original epochs and replace with corrected data
        epochs_new = epochs.copy()
        epochs_new._data[:, eeg_chans, :] = eeg_corrected

        return epochs_new

    def pre_process(self, plot_psd=False, filter_data=True,
                    filter_range=(1, 30), plot_events=False, epoch_time=(-.2, 1),
                    baseline=(-.2, 0), rej_thresh_uV=200, rereference=False,
                    emcp_raw=False, emcp_epochs=False, epoch_decim=1, plot_electrodes=False,
                    plot_erp=False):

        sfreq = self.raw.info['sfreq']
        # create new output freq for after epoch or wavelet decim
        nsfreq = sfreq/epoch_decim
        tmin = epoch_time[0]
        tmax = epoch_time[1]
        if filter_range[1] > nsfreq:
            filter_range[1] = nsfreq/2.5  # lower than 2 to avoid aliasing from decim??

        # pull event names in order of trigger number
        event_names = ['A_error', 'B_error']
        i = 0
        for key, value in sorted(self.event_id.items(), key=lambda x: (x[1], x[0])):
            event_names[i] = key
            i += 1

        # Filtering
        if rereference:
            print('Rerefering to average mastoid')
            self.raw = self.__mastoidReref()

        if filter_data:
            print('Filtering Data Between ' + str(filter_range[0]) +
                  ' and ' + str(filter_range[1]) + ' Hz.')
            self.raw.filter(filter_range[0], filter_range[1],
                            method='iir', verbose='WARNING')

        if plot_psd:
            self.raw.plot_psd(fmin=filter_range[0], fmax=nsfreq/2)

        # Eye Correction
        if emcp_raw:
            print('Raw Eye Movement Correction')
            self.raw = self.__grattonEmcpRaw()

        # Epoching

        events = find_events(self.raw, shortest_event=1, initial_event=True)
        print(events)
        color = {1: 'red', 2: 'black'}
        # artifact rejection
        rej_thresh = rej_thresh_uV*1e-6

        # plot event timing
        if plot_events:
            viz.plot_events(events, sfreq, self.raw.first_samp, color=color,
                            event_id=self.event_id)

        # Construct events - Main function from MNE
        epochs = Epochs(self.raw, events=events, event_id=self.event_id,
                        tmin=tmin, tmax=tmax, baseline=baseline,
                        preload=True, reject={'eeg': rej_thresh},
                        verbose=False, decim=epoch_decim,)
        print('Remaining Trials: ' + str(len(epochs)))

        # Gratton eye movement correction procedure on epochs
        if emcp_epochs:
            print('Epochs Eye Movement Correct')
            epochs = self.__grattonEmcpEpochs(epochs)

        # plot ERP at each electrode
        evoked_dict = {event_names[0]: epochs[event_names[0]].average(),
                       event_names[1]: epochs[event_names[1]].average()}

        # butterfly plot
        if plot_electrodes:
            picks = pick_types(evoked_dict[event_names[0]].info, meg=False, eeg=True, eog=False)
            # fig_zero = evoked_dict[event_names[0]].plot(spatial_colors=True, picks=picks)
            # fig_zero = evoked_dict[event_names[1]].plot(spatial_colors=True, picks=picks)

        # plot ERP in each condition on same plot
        if plot_erp:
            # find the electrode most miximal on the head (highest in z)
            picks = np.argmax([evoked_dict[event_names[0]].info['chs'][i]['loc'][2]
                               for i in range(len(evoked_dict[event_names[0]].info['chs']))])
            colors = {event_names[0]: "Red", event_names[1]: "Blue"}
            viz.plot_compare_evokeds(evoked_dict, colors=colors,
                                     picks=picks, split_legend=True)

        return epochs

    def feature_engineer(self, epochs: Epochs, frequency_domain=False,
                         normalization=False, electrode_median=False,
                         wavelet_decim=1, flims=(3, 30), include_phase=False,
                         f_bins=20, wave_cycles=3,
                         wavelet_electrodes=[11, 12, 13, 14, 15],
                         spect_baseline=[-1, -.5],
                         watermark=False):
        """
        Takes epochs object as

        input and settings,
        outputs  feats(training, test and val data option to use frequency or time domain)

        TODO: take tfr? or autoencoder encoded object?

        FeatureEngineer(epochs, model_type='NN',
                        frequency_domain=False,
                        normalization=False, electrode_median=False,
                        wavelet_decim=1, flims=(3,30), include_phase=False,
                        f_bins=20, wave_cycles=3,
                        wavelet_electrodes = [11,12,13,14,15],
                        spect_baseline=[-1,-.5],
                        test_split = 0.2, val_split = 0.2,
                        random_seed=1017, watermark = False):
        """
        np.random.seed(self._feats.random_seed)

        # pull event names in order of trigger number
        epochs.event_id = {'cond0': 1, 'cond1': 2}
        event_names = ['cond0', 'cond1']
        i = 0
        for key, value in sorted(epochs.event_id.items(),
                                 key=lambda item: (item[1], item[0])):
            event_names[i] = key
            i += 1

        # Create feats object for output
        self._feats.num_classes = len(epochs.event_id)

        if frequency_domain:
            print('Constructing Frequency Domain Features')

            # list of frequencies to output
            f_low = flims[0]
            f_high = flims[1]
            frequencies = np.linspace(f_low, f_high, f_bins, endpoint=True)

            # option to select all electrodes for fft
            if wavelet_electrodes == 'all':
                wavelet_electrodes = pick_types(epochs.info, eeg=True, eog=False)

            # type of output from wavelet analysis
            if include_phase:
                tfr_output_type = 'complex'
            else:
                tfr_output_type = 'power'

            tfr_dict = {}
            for event in event_names:
                print('Computing Morlet Wavelets on ' + event)
                tfr_temp = tfr_morlet(epochs[event], freqs=frequencies,
                                      n_cycles=wave_cycles, return_itc=False,
                                      picks=wavelet_electrodes, average=False,
                                      decim=wavelet_decim, output=tfr_output_type)

                # Apply spectral baseline and find stim onset time
                tfr_temp = tfr_temp.apply_baseline(spect_baseline, mode='mean')
                stim_onset = np.argmax(tfr_temp.times > 0)

                # Reshape power output and save to tfr dict
                power_out_temp = np.moveaxis(tfr_temp.data[:, :, :, stim_onset:], 1, 3)
                power_out_temp = np.moveaxis(power_out_temp, 1, 2)
                print(event + ' trials: ' + str(len(power_out_temp)))
                tfr_dict[event] = power_out_temp

            # reshape times (sloppy but just use the last temp tfr)
            self._feats.new_times = tfr_temp.times[stim_onset:]

            for event in event_names:
                print(event + ' Time Points: ' + str(len(self._feats.new_times)))
                print(event + ' Frequencies: ' + str(len(tfr_temp.freqs)))

            # Construct X and Y
            for ievent, event in enumerate(event_names):
                if ievent == 0:
                    X = tfr_dict[event]
                    Y_class = np.zeros(len(tfr_dict[event]))
                else:
                    X = np.append(X, tfr_dict[event], 0)
                    Y_class = np.append(Y_class, np.ones(len(tfr_dict[event]))*ievent, 0)

            # concatenate real and imaginary data
            if include_phase:
                print('Concatenating the real and imaginary components')
                X = np.append(np.real(X), np.imag(X), 2)

            # compute median over electrodes to decrease features
            if electrode_median:
                print('Computing Median over electrodes')
                X = np.expand_dims(np.median(X, axis=len(X.shape)-1), 2)

            # reshape for various models
            if self._feats.model_type == 'NN':
                X = np.reshape(X, (X.shape[0], X.shape[1], np.prod(X.shape[2:])))

        if not frequency_domain:
            print('Constructing Time Domain Features')

            # if using muse aux port as eeg must label it as such
            eeg_chans = pick_types(epochs.info, eeg=True, eog=False)

            # put channels last, remove eye and stim
            X = np.moveaxis(epochs._data[:, eeg_chans, :], 1, 2)

            # take post baseline only
            stim_onset = np.argmax(epochs.times > 0)
            self._feats.new_times = epochs.times[stim_onset:]
            X = X[:, stim_onset:, :]

            # convert markers to class
            # requires markers to be 1 and 2 in data file?
            # This probably is not robust to other marker numbers
            Y_class = epochs.events[:, 2]-1  # subtract 1 to make 0 and 1

            # median over electrodes to reduce features
            if electrode_median:
                print('Computing Median over electrodes')
                X = np.expand_dims(np.median(X, axis=len(X.shape)-1), 2)

        # Normalize X - TODO: need to save mean and std for future test + val
        if normalization:
            print('Normalizing X')
            X = (X - np.mean(X)) / np.std(X)

        # convert class vectors to one hot Y and recast X
        Y = to_categorical(Y_class, self._feats.num_classes)
        X = X.astype('float32')

        # add watermark for testing models
        if watermark:
            X[Y[:, 0] == 0, 0:2, ] = 0
            X[Y[:, 0] == 1, 0:2, ] = 1

        # Compute model input shape
        self._feats.input_shape = X.shape[1:]

        # Split training test and validation data
        val_prop = self._feats.val_split / (1-self._feats.test_split)
        (self._feats.x_train,
         self._feats.x_test,
         self._feats.y_train,
         self._feats.y_test) = train_test_split(X, Y,
                                                test_size=self._feats.test_split,
                                                random_state=self._feats.random_seed)
        (self._feats.x_train,
         self._feats.x_val,
         self._feats.y_train,
         self._feats.y_val) = train_test_split(self._feats.x_train, self._feats.y_train,
                                               test_size=val_prop,
                                               random_state=self._feats.random_seed)

        # compute class weights for uneven classes
        y_ints = [y.argmax() for y in self._feats.y_train]
        self._feats.class_weights = class_weight.compute_class_weight('balanced',
                                                                      np.unique(y_ints),
                                                                      y_ints)

        # Print some outputs
        print('Combined X Shape: ' + str(X.shape))
        print('Combined Y Shape: ' + str(Y_class.shape))
        print('Y Example (should be 1s & 0s): ' + str(Y_class[0:10]))
        print('X Range: ' + str(np.min(X)) + ':' + str(np.max(X)))
        print('Input Shape: ' + str(self._feats.input_shape))
        print('x_train shape:', self._feats.x_train.shape)
        print(self._feats.x_train.shape[0], 'train samples')
        print(self._feats.x_test.shape[0], 'test samples')
        print(self._feats.x_val.shape[0], 'validation samples')
        print('Class Weights: ' + str(self._feats.class_weights))
