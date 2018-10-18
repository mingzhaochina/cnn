#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 08:33:42 2018

@author: liao
"""
import sys
from obspy.core import read
from obspy.signal.trigger import ar_pick
from obspy.core.trace import Trace
from obspy.core.trace import Stats
from obspy.core.stream import Stream
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util.attribdict import AttribDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import redis
import struct
import numpy as np
import os
import subprocess
from datetime import datetime

sys.path.append('/Users/liao/Reaps/ReapsPy/AirGunDataPro')


class waveform():
    """
    Redis4seed是用于调用C++程序对seed文件进行解析，将解析台站参数与波形信息写入redis共享内存，
    然后本类从redis共享内存中读取台站参数信息写入？？，读取波形信息写入
    """

    def __init__(self, redisHost):
        self.redisHost = redisHost
        self.r = redis.StrictRedis(host=self.redisHost, port=6379, db=0)

    def sss2redis(self):
        import requests
        session = requests.Session()
        response = session.get('#http://server_ip#;user=#username#;pass=#passwd#')
        print(response.content.decode())
        # response = session.get('http://10.35.12.13:8080/jopens-sss/sss/retr;staList=FZCM')
        # print response.content.decode()
        r = session.get('#server address#;staList=#stname#', stream=True)
        f = open("file_path", "wb")
        for chunk in r.iter_content(chunk_size=512):
            print('chunk:')
            print(chunk)
            if chunk:
                print('512')
                f.write(chunk)
        response = session.get('#server_address#')
        print(response.content.decode())

        pass

    def seed2redis(self, wave_filename, start_sec, end_sec):
        pass

    def GetStnNoChnNo(self, network, station, channel, location):
        """
        根据network,station,channel,location查找台站与通道代码，失败返回-1
        """
        staNo = -1;
        chnNo = -1;
        findFlag = False
        for staNo in range(0, self.station_number):
            if self.station_list[staNo]['network'] == network and self.station_list[staNo]['station'] == station:
                for chnNo in range(0, self.station_list[staNo]['channelNum']):
                    if self.channel_list[staNo][chnNo]["channel"] == channel and self.channel_list[staNo][chnNo][
                        "location"] == location:
                        findFlag = True
                    if findFlag == True:
                        break
            if findFlag == True:
                break
        print(staNo, chnNo)
        return staNo, chnNo

    def ReadStnPara(self):
        """
        读取redis中的台站参数信息，台站信息存入一维数组station_list，通道信息存入二维数组channel_list
        """
        stn_para_key = 'StnPara';
        self.station_number = int(self.r.hget(stn_para_key, "station_number"))
        self.trace_list = [[Trace() for col in range(3)] for row in range(self.station_number)]
        print('++++')
        print(type(self.trace_list[0][0]))
        self.channel_list = [[0 for col in range(3)] for row in range(self.station_number)]
        self.station_list = [0 for col in range(self.station_number)]
        # multilist = [[0 for col in range(5)] for row in range(3)]   #二维数组
        coordinates = AttribDict()
        channel_name = [0 for col in range(3)]
        for key in ['latitude', 'longitude', 'elevation']:
            coordinates[key] = 0
        chnNum = 0;
        for key in ['Z', 'E', 'N']:
            channel_name[chnNum] = key
            chnNum = chnNum + 1

        for staNo in range(0, self.station_number):
            stn_para_field = '{0:0>4}'.format(staNo);
            stn_para_res = self.r.hget(stn_para_key, stn_para_field)
            stn_par = stn_para_res.decode('utf-8').split()
            # print(str(staNo) + str(stn_par))

            stn_para_defaults = AttribDict()
            stn_para_defaults['coordinates'] = AttribDict()
            stn_para_defaults['coordinates'] = coordinates
            stn_para_defaults['network'] = stn_par[1]
            stn_para_defaults['station'] = stn_par[2]
            stn_para_defaults['channel'] = stn_par[4]
            stn_para_defaults['location'] = stn_par[5]
            stn_para_defaults['latitude'] = float(stn_par[6])
            stn_para_defaults['longitude'] = float(stn_par[7])
            stn_para_defaults['elevation'] = float(stn_par[8])
            stn_para_defaults['channelNum'] = int(stn_par[9])
            stn_para_defaults['sampling_rate'] = int(stn_par[10])
            self.station_list[staNo] = stn_para_defaults

            for chnNo in range(0, self.station_list[staNo]['channelNum']):
                chn_para_defaults = Stats(AttribDict())
                chn_para_defaults['sampling_rate'] = self.station_list[staNo]['sampling_rate']
                chn_para_defaults['delta'] = 1.0
                chn_para_defaults['calib'] = 1.0
                chn_para_defaults['starttime'] = UTCDateTime(0)
                chn_para_defaults['npts'] = 0
                chn_para_defaults['network'] = stn_para_defaults['network']
                chn_para_defaults['station'] = stn_para_defaults['station']
                chn_para_defaults['channel'] = stn_para_defaults['channel'] + channel_name[chnNo]
                chn_para_defaults['location'] = stn_para_defaults['location']
                chn_para_defaults['response'] = float(stn_par[11 + chnNo])
                self.channel_list[staNo][chnNo] = chn_para_defaults

    def GetWaveDataUtcTime(self, staNo, chnNo, startTimeUtc, length):
        """
        读取redis中指定台站编号、通道编号、波形开始时间、波形结束时间的波形数据，返回Trace类。读取失败数据返回极大值
        """
        startTimeInt = int(startTimeUtc.timestamp)
        endTimeInt = startTimeInt + length
        Trace = self.GetWaveData(staNo, chnNo, startTimeInt, endTimeInt)
        return Trace

    def GetWaveData(self, staNo, chnNo, startTimeInt, endTimeInt):
        """ 
        读取redis中指定台站编号、通道编号、波形开始时间、波形结束时间的波形数据，返回Trace类。读取失败数据返回极大值
        """
        list_data = list();
        errorSecSum = 0;
        print ("GetWaveData")
        for sec in range(startTimeInt, endTimeInt):
           # print ('sec:'+str(sec))
            wave_key_name = self.channel_list[staNo][chnNo]['network'] + '.' + self.channel_list[staNo][chnNo][
                'station'] + '.' + \
                            self.channel_list[staNo][chnNo]['channel'] + '.' + self.channel_list[staNo][chnNo][
                                'location'] + '.Data'
            res = self.r.hget(wave_key_name, sec)
            ddd = [np.iinfo(np.int32).max for col in range(0, self.station_list[staNo]['sampling_rate'])]
            if res is not None:
                fmt = ''
                for num in range(0, self.station_list[staNo]['sampling_rate']):
                    fmt = fmt + 'i'
                # print (struct.unpack(fmt,res))
                ddd = list(struct.unpack(fmt, res))
            else:
                print('the ' + str(sec) + ' second, data error')
                errorSecSum = errorSecSum + 1
            list_data = list_data + ddd;
        print('Total ' + str(errorSecSum) + ' seconds data error')
        tr_data = np.array(list_data,)
        tr_Stat = self.channel_list[staNo][chnNo];
        tr_Stat['sampling_rate'] = self.station_list[staNo]['sampling_rate']
        tr_Stat['starttime'] = UTCDateTime(startTimeInt)
        tr_Stat['npts'] = len(tr_data)
        return Trace(tr_data, tr_Stat)

    def GetWaveTime(self, staNo, chnNo):
        """
        读取redis中指定台站编号、通道编号的波形开始时间与结束时间，读取失败返回-1
        """
        wave_time_key = 'WaveTime';
        wave_time_field = self.channel_list[staNo][chnNo]['network'] + '.' + self.channel_list[staNo][chnNo][
            'station'] + '.' + \
                          self.channel_list[staNo][chnNo]['channel'] + '.' + self.channel_list[staNo][chnNo][
                              'location'];
        print("wave_time_field:" + wave_time_field)
        print(staNo, chnNo, wave_time_key, wave_time_field)
        res = self.r.hget(wave_time_key, wave_time_field)
        wave_time_start = -1
        wave_time_end = -1
        if res is not None:
            wave_time_list = res.split()
            wave_time_start = int(wave_time_list[0])
            wave_time_end = int(wave_time_list[1])
        return wave_time_start, wave_time_end

    def CleanOrigin(self):
        res = self.r.delete('QuakeIdList')
        res = self.r.delete('QuakeCatalog')
        return

    def WriteOrigin(self, originTime, cataPhases, keepSeconds):
        QuakeIdListString = ''
        res = self.r.hkeys('QuakeCatalog')  # 查看已经存在的地震列表
        if res is not None:
            # res存入列表
            print(len(res))
            for i in range(0, len(res)):
                originTimeInRedis = UTCDateTime(res[i].decode('utf-8'));
                if originTime - originTimeInRedis > keepSeconds:
                    self.r.hdel('QuakeCatalog', str(originTimeInRedis))
                    print("del:" + str(originTimeInRedis))
                else:
                    QuakeIdListString = QuakeIdListString + str(originTimeInRedis) + ';'
        QuakeIdListString = QuakeIdListString + str(originTime)
        res = self.r.set('QuakeIdList', QuakeIdListString)
        res = self.r.hset('QuakeCatalog', str(originTime), cataPhases)
        #      print(origins)
        #      print(picks)
        return

    def GetWaveFileName(self):
        waveFileName = ''
        res = self.r.get('WaveFileName')
        print(type(res))
        if res != None:
            waveFileName = res.decode('utf-8')
        return waveFileName

    def GetBufferLen(self):
        bufferLength = 0
        res = self.r.hget('BufferParam','BufferLength')
        print(type(res))
        if res != None:
            bufferLength = int(res)
        return bufferLength

    def read_redis(self, wave_filename, start_sec, end_sec):
        pass
#    
#
