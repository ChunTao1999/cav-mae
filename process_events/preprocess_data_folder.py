import json
import numpy as np
import pandas as pd
import os
import subprocess
import matplotlib.pyplot as plt
import pdb # for debug


def preprocess(data_path, wheelAccel_conf, download=True, plot_wheelAccel=True):
    """
    Preprocess the data folder for one drive
    """
    # extract session ids from the data folder
    dataFolderNames = [s for s in os.listdir(data_path) \
                       if (os.path.isdir(os.path.join(data_path, s)) and s.split('_')[-1].isnumeric())]
    sessionIdStrs = ' '.join(s.split('_')[-1] for s in dataFolderNames)
    # prepare folder to save downloaded csvs
    csv_save_path = os.path.join('/'.join(data_path.split('/')[:-1]), 'session_csvs')
    if not os.path.exists(csv_save_path):
        os.makedirs(csv_save_path)
    wheelAccel_save_path = os.path.join(data_path, 'wheelAccels')
    if not os.path.exists(wheelAccel_save_path):
        os.makedirs(wheelAccel_save_path)
    
    # download all relevant session data csvs according to the extracted session ids, if needed
    if download:
        # 100Hz session data, 500Hz session data, session offset data
        print(subprocess.run(["chmod", "+x", "/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadSessionCsv.sh"]))
        subprocess.call("/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadSessionCsv.sh {}".format(sessionIdStrs), shell=True)
        print(subprocess.run(["chomd", "+x", "/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadSessionuhfdCsv.sh"]))
        subprocess.call("/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadSessionuhfdCsv.sh {}".format(sessionIdStrs), shell=True)
        print(subprocess.run(["chmod", "+x", "/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadUnpackedSession.sh"]))
        subprocess.call("/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadUnpackedSession.sh {}".format(sessionIdStrs), shell=True)
        print(subprocess.run(["chomd", "+x", "/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadEventList.sh"]))
        subprocess.call("/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadEventList.sh {}".format(sessionIdStrs), shell=True)
    
    # get the vehicle's rear WheelAccel and speed sensor data, and event timestamps from session csv data for all sessions
    num_samples = wheelAccel_conf['timespan'] * wheelAccel_conf['sampling_freq']
    for sessionFolderName in dataFolderNames:
        session_id = int(sessionFolderName.split('_')[-1]) # cast to type int
        # read the session csvs
        sessionData_100 = pd.read_csv(os.path.join(csv_save_path, 'session_{}.csv'.format(session_id)))
        sessionData_500 = pd.read_csv(os.path.join(csv_save_path, 'uhfdsession_{}.csv'.format(session_id)))
        cols_100 = sessionData_100.columns.values
        cols_500 = sessionData_500.columns.values
        sessionData_events = json.load(open(os.path.join(csv_save_path, 'eventList_{}.json'.format(session_id))))
        session_timeShift = json.load(open(os.path.join(csv_save_path, 'unpackedSession_{}.json'.format(session_id))))[0]['timeOffsetShift']
        session_timeShift = float(format(session_timeShift, '.3f'))
        session_eventDict = {}
        for event in sessionData_events:
            session_eventDict[float(format(event['timeOffset'], '.3f'))] = [event['type'], event['leftIntensity'], event['rightIntensity']]
        # read the json data recorded online during the session
        json_data = json.load(open(os.path.join(data_path, sessionFolderName, '{}.json'.format(sessionFolderName))))
        assert (json_data['session_id']==session_id), "session id mismatch detected between session folder name and sesion json data!"
        del json_data['session_id'] # so that it only contains events
        # crop the wheelAccel segments around event timestamps
        for key, values in json_data.items(): # keys are event timestamps, values are frame timestamps and distances
            event_timestamp = float(key)
            # allow overlaps across events
            event_start_idx = int((event_timestamp - session_timeShift - wheelAccel_conf['timespan']/2)*wheelAccel_conf['sampling_freq'])
            event_end_idx = int(event_start_idx + num_samples)
            event_left, event_right = session_eventDict[float(format(event_timestamp-session_timeShift, '.3f'))][1:3]
            # crop the wheelAccel data into segments around event timestamps
            if event_left and not event_right:
                wheelAccel_seg = np.float32(sessionData_500['rlWheelAccel'][event_start_idx:event_end_idx]) # (512,)
            elif event_right and not event_left:
                wheelAccel_seg = np.float32(sessionData_500['rrWheelAccel'][event_start_idx:event_end_idx])
            else: # the event is detected on both wheels
                # concat both wheels' wheelAccels
                wheelAccel_left_seg = np.float32(sessionData_500['rlWheelAccel'][event_start_idx:event_end_idx])
                wheelAccel_left_seg = np.expand_dims(wheelAccel_left_seg, axis=0) # (1, 512)
                wheelAccel_right_seg = np.float32(sessionData_500['rrWheelAccel'][event_start_idx:event_end_idx])
                wheelAccel_right_seg = np.expand_dims(wheelAccel_right_seg, axis=0) # (1, 512)
                wheelAccel_seg = np.float32(np.concatenate([wheelAccel_left_seg, wheelAccel_right_seg], axis=0)) # (2, 512)
            # save the wheelAccel segments
            with open(os.path.join(wheelAccel_save_path, 'wheelAccel_session_{:d}_event_{:.3f}.npy'.format(session_id, event_timestamp)), 'wb') as f:
                np.save(f, wheelAccel_seg)
            # if visualize=True, plot and save the left and right wheelAccels
            if plot_wheelAccel:
                fig, axes = plt.subplots(nrows=1, ncols=2)
                fig.suptitle('WheelAccel, session {:d}, event {:.3f}, left {}, right {}'.format(session_id, event_timestamp, event_left, event_right))
                axes[0].plot(np.array(sessionData_500.index/500),
                             sessionData_500['rlWheelAccel'][event_start_idx:event_end_idx])
                axes[0].set_title('rlWheelAccel')
                axes[0].set_xlabel('time')
                axes[1].plot(np.array(sessionData_500.index/500),
                             sessionData_500['rrWheelAccel'][event_start_idx:event_end_idx])
                axes[1].set_title('rrWheelAccel')
                axes[1].set_xlabel('time')
                plt.savefig(os.path.join(wheelAccel_save_path, 'wheelAccel_session_{:d}_event_{:.3f}.png'.format(session_id, event_timestamp)))
        print('WheelAccel segments saved for all events in session {}...'.format(session_id))
    print('All wheelAccels saved!')
    return 