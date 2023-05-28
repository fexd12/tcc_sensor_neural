import mne
import numpy as np
import tensorflow as tf
from tcc.utils.feats import Feats

ch_names = ['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7',
            'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16',
            'EOG-left', 'EOG-central', 'EOG-right']


def _arrays(path):
    print("reading file: " + path)
    raw = mne.io.read_raw_gdf(str(path), preload=True, verbose=True,
                              stim_channel=ch_names)

    print(raw.info)
    raw.info['lowpass'] = 100.
    raw.info['highpass'] = 0.5

    # raw.plot_psd()
    picks = mne.pick_types(raw.info, eog=True, stim=True, meg=False, eeg=True,
                           include=ch_names)
    # events = mne.find_events(raw, )
    events = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events[0], tmin=_feats.t_min,
                        tmax=_feats.t_min + _feats.t_len - 1 / raw.info['sfreq'],
                        preload=True, picks=picks, baseline=None,
                        event_repeated='merge')

    x = epochs.get_data()
    # teste = tf.keras.backend.(x, axis=0)
    original_array = np.transpose(x, (0, 2, 1))
    # data.reshape( 490, 1125, 26)
    # new_array = np.zeros((490, 1125, 26))

    new_input_data[:, :, :] = np.tile(original_array[:, np.newaxis, :], (1, 1125, 1))

    # new_array[:, :, :6] = original_array[:, 0, :]
    # new_array[:, :, 6:12] = original_array[:, 1, :]
    # new_array[:, :, 12:18] = original_array[:, 2, :]
    # new_array[:, :, 18:24] = original_array[:, 3, :]
    # new_array[:, :, 24:] = original_array[:, 4, :]

    print(new_input_data.shape)

    # data.reshape((data.shape[0],data.shape[1], 26))

    # epochs.plot_psd()
    return x.transpose((0, 2, 1)), tf.keras.utils.to_categorical(epochs.events[:, -1] - 1)
    # return _exp_moving_whiten(x).transpose((0, 2, 1)), \
    #     tf.keras.utils.to_categorical(epochs.events[:, -1] - 1, _feats.num_classes)


# def _read_files():
#     print('reading files...')

#     try:
#         # data = np.array(*tuple(map(_arrays,
#         #                            [pathname + 'A0{}{}.raw.fif'.format(j, i)
#         #                             for j in range(1, 10, 1) for i in 'TE'])),

#         data = list(map(_arrays, [pathname + 'A0{}{}.raw.fif'.format(j, i)
#                                   for j in range(6, 10, 1) for i in 'TE']))
#         #    for j in range(1, 6, 1) for i in 'TE']))
#         # for j in range(1, 10, 1) for i in 'TE']))

#         # x_data = np.array([])
#         # y_data = np.array([])

#         for x, y in data:
#             yield x, y

#         # return x_data, y_data
#         # return [_arrays(pathname + 'A0{}{}.raw.fif'.format(j, i)) for j in range(9) for i in 'TE']
#     except Exception as e:
#         print('Error processing :', e)
#         pass


_feats = Feats()

x, y = _arrays('/Users/felipe/Desktop/projetos/tcc/dataset/validation/B0804E.gdf')

model = tf.keras.saving.load_model('./model/model_v1.h5')


predictions = model.predict(tf.convert_to_tensor(x),
                            batch_size=_feats.batch_size,
                            verbose=True)

print(predictions)
