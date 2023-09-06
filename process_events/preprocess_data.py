import cv2
import json
import numpy as np
import glob
import pandas as pd
import os
import subprocess
import matplotlib.pyplot as plt
from utils import compute_event_loc_dist_curves, compute_event_loc_dist_curves_2, add_bbox_to_frame, add_accum_boxcenter, boox_coords_to_bbox_label, normalize_wheel_accel
import pdb # for debug


def preprocess(cal_data_path, 
               data_path, 
               save_path,
               json_save_path,
               wheelAccel_conf, 
               eventmarking_conf, 
               eventType_json_path,
               download=True, 
               plot_veh_speed_yawrate=False,
               plot_wheelAccel=True,
               plot_processedFrames=True):
    """
    Preprocess the data folder for one drive
    """
    # length for each wheelAccel segment
    num_samples = wheelAccel_conf['timespan'] * wheelAccel_conf['sampling_freq']
    # load camera matrices, and PM and IPM transform matrices
    cal_npz = np.load(os.path.join(cal_data_path, 'caldata.npz'))
    mtx, dist = cal_npz['x'], cal_npz['y']
    resmatrix, resmatrix_inv = np.load(os.path.join(cal_data_path, 'resmatrix.npy')), \
                               np.load(os.path.join(cal_data_path, 'resmatrix_inv.npy'))
    # extract session ids from the data folder
    dataFolderNames = [s for s in os.listdir(data_path) \
                       if (os.path.isdir(os.path.join(data_path, s)) and s.split('_')[-1].isnumeric())]
    sessionIdStrs = ' '.join(s.split('_')[-1] for s in dataFolderNames)
    # prepare folder paths
    csv_save_path = os.path.join('/'.join(data_path.split('/')[:-1]), 'session_csvs')
    if not os.path.exists(csv_save_path):
        os.makedirs(csv_save_path)
    wheelAccel_imsave_path = os.path.join(data_path, 'wheelAccels')
    if not os.path.exists(wheelAccel_imsave_path):
        os.makedirs(wheelAccel_imsave_path)
    origFrame_imsave_rv_path = os.path.join(save_path, 'frames_rv')
    if not os.path.exists(origFrame_imsave_rv_path):
        os.makedirs(origFrame_imsave_rv_path)
    sessionInfo_save_path = os.path.join(data_path, 'results', 'session_info')
    if not os.path.exists(sessionInfo_save_path):
        os.makedirs(sessionInfo_save_path)
    processedFrame_imsave_rv_path = os.path.join(data_path, 'results', 'frames_rv_annotated')
    if not os.path.exists(processedFrame_imsave_rv_path):
        os.makedirs(processedFrame_imsave_rv_path)
    processedFrame_imsave_bev_path = os.path.join(data_path, 'results', 'frames_bev_annotated')
    if not os.path.exists(processedFrame_imsave_bev_path):
        os.makedirs(processedFrame_imsave_bev_path) 
    wheelAccel_save_path = os.path.join(save_path, 'wheelAccels')
    if not os.path.exists(wheelAccel_save_path):
        os.makedirs(wheelAccel_save_path)
    wheelAccel_spec_save_path = os.path.join(save_path, 'wheelAccel_specs')
    if not os.path.exists(wheelAccel_spec_save_path):
        os.makedirs(wheelAccel_spec_save_path)
    # download all relevant session data csvs according to the extracted session ids, if needed
    if download:
        # 100Hz session data, 500Hz session data, session offset data
        print(subprocess.run(["chmod", "+x", "/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadSessionCsv.sh"]))
        subprocess.call("/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadSessionCsv.sh {}".format(sessionIdStrs), shell=True)
        print(subprocess.run(["chmod", "+x", "/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadSessionuhfdCsv.sh"]))
        subprocess.call("/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadSessionuhfdCsv.sh {}".format(sessionIdStrs), shell=True)
        print(subprocess.run(["chmod", "+x", "/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadUnpackedSession.sh"]))
        subprocess.call("/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadUnpackedSession.sh {}".format(sessionIdStrs), shell=True)
        print(subprocess.run(["chmod", "+x", "/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadEventList.sh"]))
        subprocess.call("/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadEventList.sh {}".format(sessionIdStrs), shell=True)
    # initiate road event data dict
    road_event_dict = {}
    road_event_dict['data'] = {}

    # get the vehicle's rear WheelAccel and speed sensor data, and event timestamps from session csv data for all sessions
    for sessionFolderName in dataFolderNames: # for each recorded drive session
        session_id = int(sessionFolderName.split('_')[-1]) # cast to type int
        # read the session csvs
        sessionData_100 = pd.read_csv(os.path.join(csv_save_path, 'session_{}.csv'.format(session_id)))
        sessionData_500 = pd.read_csv(os.path.join(csv_save_path, 'uhfdsession_{}.csv'.format(session_id)))
        cols_100 = sessionData_100.columns.values
        cols_500 = sessionData_500.columns.values
        sessionData_events = json.load(open(os.path.join(csv_save_path, 'eventList_{}.json'.format(session_id))))
        session_timeShift = json.load(open(os.path.join(csv_save_path, 'unpackedSession_{}.json'.format(session_id))))[0]['timeOffsetShift']
        session_timeShift = float(format(session_timeShift, '.3f'))
        # extract speed and yawrate columns from 100Hz sessiond data
        veh_speed, veh_yawrate = sessionData_100.loc[:, ['time_offset', 'speed']], \
                                 sessionData_100.loc[:, ['time_offset', 'yaw_rate']]
        if plot_veh_speed_yawrate:
            for idx, session_col in enumerate([veh_speed, veh_yawrate, sessionData_500['rlWheelAccel'], sessionData_500['rrWheelAccel']]):
                plt.figure()
                if idx == 0 or idx == 1:
                    plt.plot(session_col.iloc[:,0], session_col.iloc[:,1])
                else:
                    plt.plot(np.arange(session_col.shape[0])*0.002, session_col)
                if idx == 0: 
                    plt.title(f'session {session_id} speed')
                    plt.savefig(os.path.join(sessionInfo_save_path, f'session_{session_id}_speed.png'))
                elif idx == 1: 
                    plt.title(f'session {session_id} yawrate')
                    plt.savefig(os.path.join(sessionInfo_save_path, f'session_{session_id}_yawrate.png'))
                elif idx == 2:
                    plt.title(f'session {session_id} rlWheelAccel')
                    plt.savefig(os.path.join(sessionInfo_save_path, f'session_{session_id}_rlWheelAccel.png'))
                else:
                    plt.title(f'session {session_id} rrWheelAccel')
                    plt.savefig(os.path.join(sessionInfo_save_path, f'session_{session_id}_rrWheelAccel.png'))
                plt.close()

        veh_speed.iloc[:,0] += float(session_timeShift)
        veh_yawrate.iloc[:,0] += float(session_timeShift)
        session_eventDict = {}
        for event in sessionData_events:
            session_eventDict[float(format(event['timeOffset']+session_timeShift, '.3f'))] = [event['type'], event['leftIntensity'], event['rightIntensity']]
        # read the json data recorded online during the session
        json_data = json.load(open(os.path.join(data_path, sessionFolderName, '{}.json'.format(sessionFolderName))))
        assert (json_data['session_id']==session_id), "session id mismatch detected between session folder name and sesion json data!"
        del json_data['session_id'] # so that it only contains events
        # read the json data containing event type label and description
        eventType_json_data = json.load(open(eventType_json_path))

        for key, values in json_data.items(): # for each event, keys are event timestamps, values are frame timestamps and distances
            event_timestamp = float(key)
            event_id = f'session_{session_id:d}_event_{event_timestamp:.3f}'
            road_event_dict['data'][event_id] = {'event_id': event_id} # write to dict
            
            # 1. Process wheelAccel segments and their specs
            # allow overlaps across events
            event_start_idx = int((event_timestamp - session_timeShift - wheelAccel_conf['timespan']/2)*wheelAccel_conf['sampling_freq'])
            event_end_idx = int(event_start_idx + num_samples)
            event_start_idx_100, event_end_idx_100 = int((event_timestamp - session_timeShift - wheelAccel_conf['timespan']/2)*100), \
                                                     int((event_timestamp - session_timeShift + wheelAccel_conf['timespan']/2)*100)
            print('Event:', str(session_id), str(format(event_timestamp, '.3f')))
            event_label = session_eventDict[float(format(event_timestamp, '.3f'))][0]
            road_event_dict['data'][event_id]['event_label'] = str(event_label)
            road_event_dict['data'][event_id]['event_type'] = eventType_json_data[str(event_label)]
            event_left, event_right = session_eventDict[float(format(event_timestamp, '.3f'))][1:3]
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
            # normalize the wheelAccel_seg based on peak speed and nominal speed
            wheelAccel_seg = normalize_wheel_accel(wheelAccel_seg=wheelAccel_seg,
                                                   v_peak=np.max(sessionData_100['speed'][event_start_idx_100:event_end_idx_100]),
                                                   v_nom=wheelAccel_conf['nominal_speed'])
            # save the wheelAccel segments
            with open(os.path.join(wheelAccel_save_path, 'wheelAccel_session_{:d}_event_{:.3f}.npy'.format(session_id, event_timestamp)), 'wb') as f:
                np.save(f, wheelAccel_seg)
            road_event_dict['data'][event_id]['wheelAccel_path'] = os.path.join(wheelAccel_save_path, f'wheelAccel_session_{session_id:d}_event_{event_timestamp:.3f}.npy')
            # process the wheelAccel segs to transform into spectrograms
            plt.figure()
            # 8.22: need to adjust the spectrogram for road events that happen on both left and right sides
            spectrum, freqs, t_bins, im = plt.specgram(x=wheelAccel_seg, 
                                                       NFFT=wheelAccel_conf['N_windows_fft'], 
                                                       noverlap=wheelAccel_conf['noverlap'], # can add some overlap
                                                       Fs=wheelAccel_conf['sampling_freq'], 
                                                       Fc=0,
                                                       mode='default',
                                                       scale='default',
                                                       scale_by_freq=True) # (17,16) or (33,8)
            # save the spectrogram
            with open(os.path.join(wheelAccel_spec_save_path, 'wheelAccel_session_{:d}_event_{:.3f}_spec.npy'.format(session_id, event_timestamp)), 'wb') as f:
                np.save(f, spectrum)
            road_event_dict['data'][event_id]['wheelAccel_spec_path'] = os.path.join(wheelAccel_spec_save_path, f'wheelAccel_session_{session_id:d}_event_{event_timestamp:.3f}_spec.npy')
            if plot_wheelAccel:
                plt.savefig(os.path.join(wheelAccel_imsave_path, 'wheelAccel_session_{:d}_event_{:.3f}_spec.png'.format(session_id, event_timestamp)))
            plt.close()
            # if visualize=True, plot and save the left and right wheelAccels
            if plot_wheelAccel:
                fig, axes = plt.subplots(nrows=1, ncols=2)
                fig.suptitle('WheelAccel, session {:d}, event {:.3f}, left {}, right {}'.format(session_id, event_timestamp, event_left, event_right))
                axes[0].plot(np.array(sessionData_500.index[event_start_idx:event_end_idx]/500),
                             sessionData_500['rlWheelAccel'][event_start_idx:event_end_idx])
                axes[0].set_title('rlWheelAccel')
                axes[0].set_xlabel('time')
                axes[1].plot(np.array(sessionData_500.index[event_start_idx:event_end_idx]/500),
                             sessionData_500['rrWheelAccel'][event_start_idx:event_end_idx])
                axes[1].set_title('rrWheelAccel')
                axes[1].set_xlabel('time')
                plt.savefig(os.path.join(wheelAccel_imsave_path, 'wheelAccel_session_{:d}_event_{:.3f}.png'.format(session_id, event_timestamp)))
                plt.close()

            # 2. Process one/all of the frames relevant to the current event
            road_event_dict['data'][event_id]['frame_paths'] = []
            road_event_dict['data'][event_id]['bbox_coords'] = []
            for frame_idx, [frame_timestamp, frame_dist] in enumerate(values): # for each frame
                frame_name = f'session_{session_id:d}_event_{event_timestamp:.3f}_frame_{frame_idx:d}_at_time_{frame_timestamp:.3f}_dist_{frame_dist:.3f}.png'
                frame = cv2.imread(os.path.join(data_path, sessionFolderName, '_'.join(frame_name.split('_')[2:])))  
                frame = cv2.undistort(frame, mtx, dist)
                road_event_dict['data'][event_id]['frame_paths'].append(os.path.join(origFrame_imsave_rv_path, frame_name))
                cv2.imwrite(os.path.join(origFrame_imsave_rv_path, frame_name), frame)
                pts_bev, pts_inv, accum_boxcenter = compute_event_loc_dist_curves_2(event_timeoffset=event_timestamp+eventmarking_conf['event_timestamp_shift'],
                                                                                    event_left=session_eventDict[float(format(event_timestamp, '.3f'))][1],
                                                                                    event_right=session_eventDict[float(format(event_timestamp, '.3f'))][2],
                                                                                    frame_image=frame,
                                                                                    frame_dim=eventmarking_conf['bev_frame_dim'],
                                                                                    frame_timeoffset=frame_timestamp+eventmarking_conf['frame_timestamp_shift'],
                                                                                    frame_dist=frame_dist,
                                                                                    wheel_to_base_dist=eventmarking_conf['wheel_to_base_dist'],
                                                                                    base_pixel=eventmarking_conf['base_pixel'],
                                                                                    wheel_width=eventmarking_conf['wheel_width'],
                                                                                    veh_speed=veh_speed,
                                                                                    veh_yawrate=veh_yawrate,
                                                                                    xm_per_pix=eventmarking_conf['xm_per_pix'],
                                                                                    ym_per_pix=eventmarking_conf['ym_per_pix'],
                                                                                    event_len_pix=eventmarking_conf['event_len_pix'],
                                                                                    resmatrix_inv=resmatrix_inv,
                                                                                    resmatrix=resmatrix)
                frame_rv = add_bbox_to_frame(image=frame,
                                             pts_inv=pts_inv)
                frame_bev = cv2.warpPerspective(np.float32(frame),
                                                resmatrix,
                                                eventmarking_conf['bev_frame_dim'])
                frame_bev = add_accum_boxcenter(image=frame_bev,
                                                accum_boxcenter=accum_boxcenter)
                frame_bev = add_bbox_to_frame(image=frame_bev,
                                              pts_inv=pts_bev)
                road_event_dict['data'][event_id]['bbox_coords'].append(boox_coords_to_bbox_label(pts_inv))
                if plot_processedFrames:
                    cv2.imwrite(os.path.join(processedFrame_imsave_rv_path, f'session_{session_id:d}_event_{event_timestamp:.3f}_frame_{frame_idx:d}_at_time_{frame_timestamp:.3f}_dist_{frame_dist:.3f}_rv.png'), frame_rv) 
                    cv2.imwrite(os.path.join(processedFrame_imsave_bev_path, f'session_{session_id:d}_event_{event_timestamp:.3f}_frame_{frame_idx:d}_at_time_{frame_timestamp:.3f}_dist_{frame_dist:.3f}_bev.png'), frame_bev) 
                print('\tsession_{:d}_event_{:.3f}_frame_{}_at_time_{:.3f}_dist_{:.3f}'.format(session_id,
                                                                                               event_timestamp,
                                                                                               frame_idx, 
                                                                                               frame_timestamp,
                                                                                               frame_dist))
        print(f'Session {session_id} processed......')
    
    # 3. Write event data path into to the event dataset dictionary and save to the jsonfile path
    with open(json_save_path, 'w') as outfile:
        json.dump(road_event_dict, outfile)
    print('All processed frames and wheelAccels saved!')
    return 


def preprocess_internal(session_list,
                        date_list,
                        cal_data_path, 
                        data_path, 
                        save_path,
                        json_save_path,
                        wheelAccel_conf, 
                        eventmarking_conf, 
                        eventType_json_path,
                        download=True, 
                        plot_processedFrames=True):
    """
    Preprocess the data folder for one drive
    """
    # length for each wheelAccel segment
    num_samples = wheelAccel_conf['timespan'] * wheelAccel_conf['sampling_freq']
    # load camera matrices, and PM and IPM transform matrices
    cal_npz = np.load(os.path.join(cal_data_path, 'caldata.npz'))
    mtx, dist = cal_npz['x'], cal_npz['y']
    resmatrix, resmatrix_inv = np.load(os.path.join(cal_data_path, 'resmatrix.npy')), \
                               np.load(os.path.join(cal_data_path, 'resmatrix_inv.npy'))
    # prepare folder paths
    csv_save_path = os.path.join(data_path, 'session_csvs')
    if not os.path.exists(csv_save_path):
        os.makedirs(csv_save_path)
    # download all relevant session data csvs according to the extracted session ids, if needed
    if download:
        # 100Hz session data, 500Hz session data, session offset data
        print(subprocess.run(["chmod", "+x", "/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadSessionCsv.sh"]))
        subprocess.call("/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadSessionCsv.sh {}".format(session_list), shell=True)
        print(subprocess.run(["chmod", "+x", "/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadSessionuhfdCsv.sh"]))
        subprocess.call("/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadSessionuhfdCsv.sh {}".format(session_list), shell=True)
        print(subprocess.run(["chmod", "+x", "/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadUnpackedSession.sh"]))
        subprocess.call("/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadUnpackedSession.sh {}".format(session_list), shell=True)
        print(subprocess.run(["chmod", "+x", "/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadEventList.sh"]))
        subprocess.call("/home/nano01/a/tao88/cav-mae/process_events/scripts/downloadEventList.sh {}".format(session_list), shell=True)
    # initiate road event data dict
    road_event_dict = {}
    # get dataFolder_list from session_list and date_list
    dataFolder_list = []
    for idx, session_id in enumerate(session_list):
        dataFolder_list.append(f"events_{date_list[idx]}/session_{session_id}")

    # get the vehicle's rear WheelAccel and speed sensor data, and event timestamps from session csv data for all sessions
    for sessionFolderName in dataFolder_list: # for each recorded drive session
        session_id = int(sessionFolderName.split('_')[-1]) # cast to type int
        # read the session csvs
        sessionData_100 = pd.read_csv(os.path.join(csv_save_path, 'session_{}.csv'.format(session_id)))
        sessionData_500 = pd.read_csv(os.path.join(csv_save_path, 'uhfdsession_{}.csv'.format(session_id)))

        sessionData_events = json.load(open(os.path.join(csv_save_path, 'eventList_{}.json'.format(session_id))))
        session_timeShift = json.load(open(os.path.join(csv_save_path, 'unpackedSession_{}.json'.format(session_id))))[0]['timeOffsetShift']
        session_timeShift = float(session_timeShift)
        # extract speed and yawrate columns from 100Hz sessiond data
        veh_speed, veh_yawrate = sessionData_100.loc[:, ['time_offset', 'speed']], \
                                 sessionData_100.loc[:, ['time_offset', 'yaw_rate']]
        # veh_speed.iloc[:,0] += float(session_timeShift)
        # veh_yawrate.iloc[:,0] += float(session_timeShift)
        session_eventDict = {}
        for event in sessionData_events:
            session_eventDict[float(event['timeOffset'])] = [event['type'], event['leftIntensity'], event['rightIntensity']]
        # read the json data recorded online during the session
        json_data_name = glob.glob(os.path.join(data_path, sessionFolderName, '*.json'))[0]
        json_data = json.load(open(json_data_name))
        assert (json_data['session_id']==session_id), "session id mismatch detected between session folder name and sesion json data!"
        del json_data['session_id'] # so that it only contains events
        # read the json data containing event type label and description
        eventType_json_data = json.load(open(eventType_json_path))

        for key, values in json_data.items(): # for each event, keys are event timestamps, values are frame timestamps and distances
            event_timestamp = float(key) - session_timeShift
            # change the event_id
            event_id = f"s_{session_id:d}_{event_timestamp:.3f}".replace(".", "")
            road_event_dict[event_id] = {}
            # 1. Process wheelAccel segments and their specs
            # allow overlaps across events
            event_start_idx = int((event_timestamp - wheelAccel_conf['timespan']/2)*wheelAccel_conf['sampling_freq'])
            event_end_idx = int(event_start_idx + num_samples)
            event_start_idx_100, event_end_idx_100 = int((event_timestamp - wheelAccel_conf['timespan']/2)*100), \
                                                     int((event_timestamp + wheelAccel_conf['timespan']/2)*100)
            print('Event:', str(session_id), str(event_timestamp))
            event_label = session_eventDict[float(event_timestamp)][0]
            road_event_dict[event_id]['label'] = str(event_label)
            road_event_dict[event_id]['type'] = eventType_json_data[str(event_label)]
            event_left, event_right = session_eventDict[float(event_timestamp)][1:3]
            if event_left and not event_right:
                wheel_number = 3
            elif event_right and not event_left:
                wheel_number = 4
            else:
                wheel_number = 34
            road_event_dict[event_id]['wheel'] = str(wheel_number)

            # # crop the wheelAccel data into segments around event timestamps
            # if event_left and not event_right:
            #     wheelAccel_seg = np.float32(sessionData_500['rlWheelAccel'][event_start_idx:event_end_idx]) # (512,)
            # elif event_right and not event_left:
            #     wheelAccel_seg = np.float32(sessionData_500['rrWheelAccel'][event_start_idx:event_end_idx])
            # else: # the event is detected on both wheels
            #     # concat both wheels' wheelAccels
            #     wheelAccel_left_seg = np.float32(sessionData_500['rlWheelAccel'][event_start_idx:event_end_idx])
            #     wheelAccel_left_seg = np.expand_dims(wheelAccel_left_seg, axis=0) # (1, 512)
            #     wheelAccel_right_seg = np.float32(sessionData_500['rrWheelAccel'][event_start_idx:event_end_idx])
            #     wheelAccel_right_seg = np.expand_dims(wheelAccel_right_seg, axis=0) # (1, 512)
            #     wheelAccel_seg = np.float32(np.concatenate([wheelAccel_left_seg, wheelAccel_right_seg], axis=0)) # (2, 512)
            # # normalize the wheelAccel_seg based on peak speed and nominal speed
            # wheelAccel_seg = normalize_wheel_accel(wheelAccel_seg=wheelAccel_seg,
            #                                        v_peak=np.max(sessionData_100['speed'][event_start_idx_100:event_end_idx_100]),
            #                                        v_nom=wheelAccel_conf['nominal_speed'])
            # # save the wheelAccel segments
            # with open(os.path.join(wheelAccel_save_path, 'wheelAccel_session_{:d}_event_{:.3f}.npy'.format(session_id, event_timestamp)), 'wb') as f:
            #     np.save(f, wheelAccel_seg)
            # road_event_dict[event_id]['wheelAccel_path'] = os.path.join(wheelAccel_save_path, f'wheelAccel_session_{session_id:d}_event_{event_timestamp:.3f}.npy')
            # # process the wheelAccel segs to transform into spectrograms
            # plt.figure()
            # # 8.22: need to adjust the spectrogram for road events that happen on both left and right sides
            # spectrum, freqs, t_bins, im = plt.specgram(x=wheelAccel_seg, 
            #                                            NFFT=wheelAccel_conf['N_windows_fft'], 
            #                                            noverlap=wheelAccel_conf['noverlap'], # can add some overlap
            #                                            Fs=wheelAccel_conf['sampling_freq'], 
            #                                            Fc=0,
            #                                            mode='default',
            #                                            scale='default',
            #                                            scale_by_freq=True) # (17,16) or (33,8)
            # # save the spectrogram
            # with open(os.path.join(wheelAccel_spec_save_path, 'wheelAccel_session_{:d}_event_{:.3f}_spec.npy'.format(session_id, event_timestamp)), 'wb') as f:
            #     np.save(f, spectrum)
            # road_event_dict[event_id]['wheelAccel_spec_path'] = os.path.join(wheelAccel_spec_save_path, f'wheelAccel_session_{session_id:d}_event_{event_timestamp:.3f}_spec.npy')


            # 2. Process one/all of the frames relevant to the current event
            road_event_dict[event_id]['frames'] = []
            road_event_dict[event_id]['polygon_box'] = []
            for frame_idx, [frame_timestamp, frame_dist] in enumerate(values): # for each frame
                frame_name = f'session_{session_id:d}_event_{(event_timestamp+session_timeShift):.3f}_frame_{frame_idx:d}_at_time_{frame_timestamp:.3f}_dist_{frame_dist:.3f}.png'
                frame = cv2.imread(os.path.join(data_path, sessionFolderName, '_'.join(frame_name.split('_')[2:])))  
                frame = cv2.undistort(frame, mtx, dist)
                frame_id = event_id + f"_{frame_idx:d}_{(frame_timestamp-session_timeShift):.3f}_{wheel_number:d}".replace(".", "")
                road_event_dict[event_id]['frames'].append(frame_id)
                cv2.imwrite(os.path.join(save_path ,"undistorted_rv", frame_id+".png"), frame)
                pts_bev, pts_inv, accum_boxcenter = compute_event_loc_dist_curves_2(event_timeoffset=event_timestamp+eventmarking_conf['event_timestamp_shift'],
                                                                                    event_left=session_eventDict[event_timestamp][1],
                                                                                    event_right=session_eventDict[event_timestamp][2],
                                                                                    frame_image=frame,
                                                                                    frame_dim=eventmarking_conf['bev_frame_dim'],
                                                                                    frame_timeoffset=frame_timestamp-session_timeShift+eventmarking_conf['frame_timestamp_shift'],
                                                                                    frame_dist=frame_dist,
                                                                                    wheel_to_base_dist=eventmarking_conf['wheel_to_base_dist'],
                                                                                    base_pixel=eventmarking_conf['base_pixel'],
                                                                                    wheel_width=eventmarking_conf['wheel_width'],
                                                                                    veh_speed=veh_speed,
                                                                                    veh_yawrate=veh_yawrate,
                                                                                    xm_per_pix=eventmarking_conf['xm_per_pix'],
                                                                                    ym_per_pix=eventmarking_conf['ym_per_pix'],
                                                                                    event_len_pix=eventmarking_conf['event_len_pix'],
                                                                                    resmatrix_inv=resmatrix_inv,
                                                                                    resmatrix=resmatrix)
                frame_rv = add_bbox_to_frame(image=frame,
                                             pts_inv=pts_inv)
                frame_bev = cv2.warpPerspective(np.float32(frame),
                                                resmatrix,
                                                eventmarking_conf['bev_frame_dim'])
                frame_bev = add_accum_boxcenter(image=frame_bev,
                                                accum_boxcenter=accum_boxcenter)
                frame_bev = add_bbox_to_frame(image=frame_bev,
                                              pts_inv=pts_bev)
                road_event_dict[event_id]['polygon_box'].append(boox_coords_to_bbox_label(pts_inv))
                if plot_processedFrames:
                    cv2.imwrite(os.path.join(save_path ,"undistorted_rv_annotated", frame_id+".png"), frame_rv) 
                    cv2.imwrite(os.path.join(save_path ,"undistorted_bev_annotated", frame_id+".png"), frame_bev) 
                print('\tsession_{:d}_event_{:.3f}_frame_{}_at_time_{:.3f}_dist_{:.3f}'.format(session_id,
                                                                                               event_timestamp,
                                                                                               frame_idx, 
                                                                                               frame_timestamp-session_timeShift,
                                                                                               frame_dist))
        print(f'Session {session_id} processed......')
    
    # 3. Write event data path into to the event dataset dictionary and save to the jsonfile path
    with open(json_save_path, 'w') as outfile:
        json.dump(road_event_dict, outfile)
    print('All processed frames and wheelAccels saved!')
    return 