# -*- coding: <utf-8> -*-
#!/usr/bin/env python
# Automatic picking of seismic waves using Generalized Phase Detection
# See http://scedc.caltech.edu/research-tools/deeplearning.html for more info
#
# Ross et al. (2018), Generalized Seismic Phase Detection with Deep Learning,
#                     Bull. Seismol. Soc. Am., doi:10.1785/0120180080
#
# Author: Zachary E. Ross (2018)
# Contact: zross@gps.caltech.edu
# Website: http://www.seismolab.caltech.edu/ross_z.html
# 2.Ming Zhao (mzhao@cea-igp.ac.cn) modified and applied it to P,S phase picking from continious waveform,real-time stream shared by redis,etc. 
#Usage:
#./gpd_predict_liss_from_fujian_single_station.py --V true --para_path scedc_model --plot true --receive_interval 30  --output_path gpd_model_predicts --save_mseed true
#################################################################

import string
import time
import argparse as ap
import sys,csv
import os
from datetime import datetime
import numpy as np
import obspy.core as oc
import tf.keras
from obspy.core.utcdatetime import UTCDateTime
import pandas as pd
from tf.keras.models import Sequential
from tf.keras.layers import Dense, Dropout, Activation, Flatten
from tf.keras.layers import Conv1D, MaxPooling1D
from tf.keras import losses
from tf.keras.models import model_from_json
from rediss import redis2app as rediss
from liss import liss
from obspy.signal.trigger import trigger_onset
import tensorflow as tf
import matplotlib as mpl
import json
mpl.use('Agg')
import matplotlib.pyplot as plt
#import pylab as plt
mpl.rcParams['pdf.fonttype'] = 42
#-------------------------------------------------------------
flags = tf.flags
flags.DEFINE_string('para_path',
                    None, 'path to the records containing the windows.')
flags.DEFINE_string('output_path',
                    None, 'path to save the checkpoints and summaries.')
flags.DEFINE_bool("plot", False,
                  "If we want the event traces to be plotted")
flags.DEFINE_bool("V", False,
                  "If we want the event traces to be plotted")
flags.DEFINE_integer(
    'receive_interval', 900, 'sleep time between receive (in seconds)')
flags.DEFINE_integer(
    'account_number', 1, 'sleep time between receive (in seconds)')
flags.DEFINE_bool("save_mseed",False,
                     "save the windows in mseed format")
FLAGS = flags.FLAGS

def write_json(metadata,output_metadata):
    with open(output_metadata, 'w') as outfile:
        json.dump(metadata, outfile)

def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    """
    Calculate a sliding window over a signal
    Parameters
    ----------
    data : numpy array
        The array to be slided over.
    size : int
        The sliding window size
    stepsize : int
        The sliding window stepsize. Defaults to 1.
    axis : int
        The axis to slide over. Defaults to the last axis.
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.
    Returns
    -------
    data : numpy array
        A matrix where row in last dimension consists of one instance
        of the sliding window.
    Notes
    -----
    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occurr.
    Examples
    --------
    >>> a = numpy.array([1, 2, 3, 4, 5])
    >>> sliding_window(a, size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> sliding_window(a, size=3, stepsize=2)
    array([[1, 2, 3],
           [3, 4, 5]])
    See Also
    --------
    pieces : Calculate number of pieces available by sliding
    """
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )

    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )

    if copy:
        return strided.copy()
    else:
        return strided

def main(_):
    #####################
    # Hyperparameters
    min_proba_s = 0.95  # Minimum softmax probability for phase detection
    min_proba_p = 0.85  # Minimum softmax probability for phase detection
    freq_min = 3.0
    freq_max = 20.0
    filter_data = False
    decimate_data = False  # If false, assumes data is already 100 Hz samprate
    n_shift = 10  # Number of samples to shift the sliding window at a time
    n_gpu = 1  # Number of GPUs to use (if any)
    #####################
    batch_size = 1000 * 3

    half_dur = 2.00
    only_dt = 0.01
    n_win = int(half_dur / only_dt)
    n_feat = 2 * n_win

    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)
    # Reading in liss stream
    wave = rediss.waveform('ip of the server')
    bufferLength = wave.GetBufferLen()
    bufferLength = 60
    wave.ReadStnPara()
    ##############################
    #first model
    ##############################
    # load json and model weight
    json_model_psn= os.path.join(FLAGS.para_path, 'model_first_trained.json')
    json_file_psn = open(json_model_psn, 'r')
    loaded_psn_model_json = json_file_psn.read()
    json_file_psn.close()
    model_psn = model_from_json(loaded_psn_model_json, custom_objects={'tf': tf})
    print ("model_psn:",model_psn)

    # load weights into new model
    try:
        best_model_psn= os.path.join(FLAGS.para_path, 'model_first_trained_best.hdf5')
        model_psn.load_weights(best_model_psn)
        #print("summary_psn::",model_psn.summary())
    except:
        print(model_psn.summary())
    print("Loaded model from disk")

    #writer = DataWriter(output_path)
    if n_gpu > 1:
        from tf.keras.utils import multi_gpu_model
        model_psn = multi_gpu_model(model_psn, gpus=n_gpu)
    p_picks_catalog = os.path.join(FLAGS.output_path, 'p_picks_by_CNN.csv')
    s_picks_catalog = os.path.join(FLAGS.output_path, 's_picks_by_CNN.csv')
    data_transfer_error = os.path.join(FLAGS.output_path, 'data_transfer_error_log.csv')
    #     df.to_csv(output_catalog)
    #     df1 = pd.DataFrame.from_dict(times_s_csv)
    #     output_catalog = os.path.join(FLAGS.output_path, 's_picks_by_CNN.csv')
    ofile_p = open(p_picks_catalog, 'w')
    ofile_s = open(s_picks_catalog, 'w')
    ofile_d = open(data_transfer_error, 'w')
    ofile_d.write("%s %s %s %s %s\n" % ("net","sta",  "timestamp",
                                                "starttime", "[errorSecSum, errorSecSum1, errorSecSum2]"))
    ofile_p.write("%s %s P %s\n" % ("net", "sta", "time"))
    ofile_s.write("%s %s S %s\n" % ("net", "sta", "time"))
    ofile_p.close()
    ofile_s.close()
    ofile_d.close()
    while True:
        if FLAGS.receive_interval < 0:
            print ('receive_interval cannot be negative.')
            break
        try:
            ofile_p = open(p_picks_catalog, 'a')
            ofile_s = open(s_picks_catalog, 'a')
            ofile_d = open(data_transfer_error, 'a')

            StnNo, ChnNo = wave.GetStnNoChnNo("FJ", "HAJF", "BHE", "00")
            #StnNo, ChnNo = wave.GetStnNoChnNo("FJ", "JLNK", "BHE", "00")
            startT, endT = wave.GetWaveTime(StnNo, ChnNo)
            print(startT, endT)
            d = datetime.utcfromtimestamp(startT)
            # print(d)
            startTime = UTCDateTime(d.year, d.month, d.day, d.hour, d.minute, d.second)

            print('--------------')
            print('startTime:', startTime)
            print(d.year, d.month, d.day, d.hour, d.minute, d.second, startTime.timestamp)
            print('--------------')
            tr, errorSecSum = wave.GetWaveDataUtcTime(StnNo, ChnNo, startTime, bufferLength)


            StnNo1, ChnNo1 = wave.GetStnNoChnNo("FJ", "HAJF", "BHN", "00")
            startT1, endT1 = wave.GetWaveTime(StnNo1, ChnNo1)
            # print(startT1, endT1)
            d1 = datetime.utcfromtimestamp(startT1)
            # print(d1)
            startTime1 = UTCDateTime(d1.year, d1.month, d1.day, d1.hour, d1.minute, d1.second)
            tr1, errorSecSum1 = wave.GetWaveDataUtcTime(StnNo1, ChnNo1, startTime1, bufferLength)


            StnNo2, ChnNo2 = wave.GetStnNoChnNo("FJ", "HAJF", "BHZ", "00")
            startT2, endT2 = wave.GetWaveTime(StnNo2, ChnNo2)
            # print(startT2, endT2)

            d2 = datetime.utcfromtimestamp(startT2)
            # print(d2)
            startTime2 = UTCDateTime(d2.year, d2.month, d2.day, d2.hour, d2.minute, d2.second)
            tr2, errorSecSum2 = wave.GetWaveDataUtcTime(StnNo2, ChnNo2, startTime2, bufferLength)

            print ("errorSecSum, errorSecSum1, errorSecSum2:", errorSecSum, errorSecSum1, errorSecSum2)

        #if (lastFile is not None):
        #    lastFile.close()

            net = tr.stats.network
            sta = tr.stats.station

            if max(errorSecSum, errorSecSum1, errorSecSum2) > 0:
                maxstart = max(startTime + errorSecSum, startTime1 + errorSecSum1, startTime2 + errorSecSum2) + 4
                ofile_d.write("%s %s %s %s  %s %s %s \n" % (net, sta, str(UTCDateTime(maxstart).timestamp),
                                                maxstart.isoformat(),str(errorSecSum), str(errorSecSum1), str(errorSecSum2)))
            else:
                maxstart = max(startTime, startTime1, startTime2)


            minend = min(tr1.stats.endtime, tr.stats.endtime, tr2.stats.endtime)
            tr.data = tr.data.astype(float)
            tr1.data = tr1.data.astype(float)
            tr2.data = tr2.data.astype(float)
            tr = tr.slice(maxstart, minend)
            tr1 = tr1.slice(maxstart, minend)
            tr2 = tr2.slice(maxstart, minend)

            st = oc.Stream()
            st += tr1
            st += tr
            st += tr2
            if FLAGS.save_mseed:
                outpath1 = os.path.join(FLAGS.output_path, "mseed")
                if not os.path.exists(outpath1):
                    os.makedirs(outpath1)
                seedname = os.path.join(outpath1, str(tr.stats.starttime).replace(':', '_') + '.mseed')
                st.write(seedname, format="MSEED")
            # latest_start = np.max([x.stats.starttime for x in st])
            ##earliest_stop = np.min([x.stats.endtime for x in st])
            # st.trim(latest_start, earliest_stop)

            st.detrend(type='linear')
            if filter_data:
                st.filter(type='bandpass', freqmin=freq_min, freqmax=freq_max)
            if decimate_data:
                st.interpolate(100.0)
            chan = st[0].stats.channel
            sr = st[0].stats.sampling_rate

            dt = st[0].stats.delta

            if FLAGS.V:
                print("Reshaping data matrix for sliding window")
            tt = (np.arange(0, st[0].data.size, n_shift) + n_win) * dt
            tt_i = np.arange(0, st[0].data.size, n_shift) + n_feat
            # tr_win = np.zeros((tt.size, n_feat, 3))
            sliding_N = sliding_window(st[0].data, n_feat, stepsize=n_shift)
            sliding_E = sliding_window(st[1].data, n_feat, stepsize=n_shift)
            sliding_Z = sliding_window(st[2].data, n_feat, stepsize=n_shift)
            tr_win = np.zeros((sliding_N.shape[0], n_feat, 3))
            # print ("tr_win",tr_win.shape)
            tr_win[:, :, 0] = sliding_N
            tr_win[:, :, 1] = sliding_E
            tr_win[:, :, 2] = sliding_Z
            # 将三通道400采样点数据的最大值取出，并扩展为（863961, 1, 1）数组形式，每个滑窗内一个最大值，并做归一化
            tr_win = tr_win / np.max(np.abs(tr_win), axis=(1, 2))[:, None, None]
            # print ("tr_win",tr_win.shape)
            tt = tt[:tr_win.shape[0]]
            tt_i = tt_i[:tr_win.shape[0]]

            if FLAGS.V:
                ts = model_psn.predict(tr_win, verbose=True, batch_size=batch_size)
            else:
                ts = model_psn.predict(tr_win, verbose=False, batch_size=batch_size)

            prob_S = ts[:, 1]
            prob_P = ts[:, 0]
            prob_N = ts[:, 2]

            trigs = trigger_onset(prob_P, min_proba_p, 0.1)
            p_picks = []
            s_picks = []
            for trig in trigs:
                if trig[1] == trig[0]:
                    continue
                pick = np.argmax(ts[trig[0]:trig[1], 0]) + trig[0]
                stamp_pick = st[0].stats.starttime + tt[pick]
                p_picks.append(stamp_pick)
                ofile_p.write("%s %s P %s\n" % (net, sta, stamp_pick.isoformat()))

            trigs = trigger_onset(prob_S, min_proba_s, 0.1)
            for trig in trigs:
                if trig[1] == trig[0]:
                    continue
                pick = np.argmax(ts[trig[0]:trig[1], 1]) + trig[0]
                stamp_pick = st[0].stats.starttime + tt[pick]
                s_picks.append(stamp_pick)
                ofile_s.write("%s %s S %s\n" % (net,sta, stamp_pick.isoformat()))

            if FLAGS.plot:
                fig = plt.figure(figsize=(8, 12))
                ax = []
                ax.append(fig.add_subplot(4, 1, 1))
                ax.append(fig.add_subplot(4, 1, 2, sharex=ax[0], sharey=ax[0]))
                ax.append(fig.add_subplot(4, 1, 3, sharex=ax[0], sharey=ax[0]))
                ax.append(fig.add_subplot(4, 1, 4, sharex=ax[0]))
                for i in range(3):
                    ax[i].plot(np.arange(st[i].data.size) * dt, st[i].data, c='k', \
                               lw=0.5)
                ax[3].plot(tt, ts[:, 0], c='r', lw=0.5)
                ax[3].plot(tt, ts[:, 1], c='b', lw=0.5)
                for p_pick in p_picks:
                    for i in range(3):
                        ax[i].axvline(p_pick - st[0].stats.starttime, c='r', lw=0.5)
                    #ax[2].axhline(s_pick - st[0].stats.starttime, c='b', lw=0.5)
                ax[3].axhline(min_proba_p, c='r', linestyle='--',lw=0.5)
                for s_pick in s_picks:
                    for i in range(3):
                        ax[i].axvline(s_pick - st[0].stats.starttime, c='b', lw=0.5)
                ax[3].axhline(min_proba_s, c='b', linestyle='--', lw=0.5)
                plt.tight_layout()
                plt.show()
                outpath = os.path.join(FLAGS.output_path, "figs")
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                filename = os.path.join(outpath, net+sta+'.'+str(st[0].stats.starttime).replace(':', '_') + '.png')
                plt.savefig(filename)
                plt.close()
            ofile_p.close()
            ofile_s.close()
            ofile_d.close()
        except KeyboardInterrupt:
            print ('stop receive at: ' + str(startTime))
            break
        except:
            print ('skip receive at: ' + str(startTime))
            ofile_p.close()
            ofile_s.close()
            ofile_d.close()
            continue
        print  ('Waiting for ' + str(FLAGS.receive_interval) + ' seconds to receive:')


        time.sleep(FLAGS.receive_interval)

if __name__ == "__main__":
    tf.app.run()
