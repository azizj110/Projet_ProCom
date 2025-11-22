import os
import pandas as pd
import mne

relative_path = 'data\ds003825'
subject = 'sub-01'

vhdr_path = os.path.join( relative_path,subject, 'eeg', f'{subject}_task-rsvp_eeg.vhdr')
raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)

tsv_path = os.path.join(relative_path, subject, 'eeg', f'{subject}_task-rsvp_events.tsv')
events_df = pd.read_csv(tsv_path, sep='\t')
print(events_df[["onset","object"]])

events, event_id = mne.events_from_annotations(raw)
print('5 premiers évènements:', events[:5])
print('nb events', len(events))
print('3 évènement possibles:', event_id, '(image, écran, changement de séquence)')
a=events[:, 2]==10003
indices = [i for i, val in enumerate(a) if val]
print("nb séquences;:", len(indices))


##### Essai 1 ######
def create_windows_from_events(raw,events_df,events, trial_start_offset_samples=0,trial_stop_offset_samples=500,event_code=10001):
    starts = events[:, 0] + trial_start_offset_samples
    stops = events[:, 0] + trial_stop_offset_samples

    img_mask=events[:, 2]==event_code
    starts = starts[img_mask]
    stops = stops[img_mask]

    windows = []
    for start, stop in zip(starts, stops):
      data, times = raw[:, start:stop]
      windows.append(data)

    data={'eeg':windows,'label':  events_df["object"]}

    return data

data=create_windows_from_events(raw,events_df,events,0,500,10001)
print(len(data['eeg']))
print(data['eeg'][0].shape)
print(data['label'])

##### Essai 2 ######

first_label, first_code = next(iter(event_id.items()))
epochs = mne.Epochs(
    raw,
    events,
    event_id={first_label: first_code},
    tmin=0,
    tmax=0.5,
    baseline=None,
    preload=True
)

labels=events_df[["object"]]

data={'eeg':epochs.get_data(),'label': labels}
print(len(data['eeg']))
print(data['eeg'][0].shape)
print(data['label'])