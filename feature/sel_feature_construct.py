import re
import warnings

import joblib
import numpy as np
import pandas as pd

from feature.spell import Spell

warnings.filterwarnings("ignore")


def _getLogID(log, sn, fault_time):
    day2sec = 24 * 60 * 60
    tmp = log[(log['sn'] == sn)].copy(deep=True)
    tmp['delta'] = tmp['timestamp'].apply(lambda x: abs((fault_time - x).total_seconds()))
    tmp = tmp[tmp['delta'] <= tmp['delta'].min() + day2sec]

    tmp = tmp.sort_values('timestamp')

    LogID = tmp.index
    LogID = ' '.join(str(i) for i in LogID)
    return LogID


def getLogID(data, submit):
    df = submit.copy(deep=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    df['fault_time'] = pd.to_datetime(df['fault_time'])

    df['LogID'] = df.apply(lambda x: _getLogID(data, x['sn'], x['fault_time']), axis=1)
    return df


def LogParse(df, train=True):
    # 建立语料库
    with open('./user_data/tmp_data/log.log', "w") as f:
        for i in range(len(df)):
            string = df['sn'].values[i] + ' ' + df['time'].values[i] + ' ' + \
                     df['server_model'].values[i] + ' ' + df['msg'].values[i]
            string = string.replace('  ', ' ')
            f.write(string)
            f.write('\n')

    # 日志模版提取
    log_format = '<sn> <Date> <Time> <server_model> <Content>'
    parser = Spell(indir='./user_data/tmp_data/', outdir='./user_data/tmp_data/', log_format=log_format,
                   keep_para=False, tau=1.5, train=train)
    df, event = parser.parse('log.log')

    if train:
        EventId2num = {}
        for num, EventId in enumerate(event['EventId'].values):
            EventId2num[EventId] = num
        joblib.dump(EventId2num, './user_data/model_data/sel_logParser_EventId2num.pkl')

        SM2ID = {}
        for i, sm in enumerate(sorted(list(set(df['server_model'].values)))):
            SM2ID[sm] = i
        joblib.dump(SM2ID, './user_data/model_data/sel_logParser_SM2ID.pkl')
    else:
        EventId2num = joblib.load('./user_data/model_data/sel_logParser_EventId2num.pkl')
        SM2ID = joblib.load('./user_data/model_data/sel_logParser_SM2ID.pkl')

    # 日志向量化
    df['timestamp'] = df.apply(lambda x: x['Date'] + ' ' + x['Time'], axis=1)
    df['EventId'] = df['EventId'].apply(lambda x: EventId2num.get(x, len(EventId2num)))
    df['server_model'] = df['server_model'].apply(lambda x: SM2ID.get(x, len(SM2ID)))

    df = df[['sn', 'timestamp', 'server_model', 'EventId']].reset_index(drop=True)
    return df


def DeviceClean(string):
    string = str(string)
    string = string.lower()
    string = string.strip()
    string = re.sub(r'[^a-z0-9]', ' ', string)  # 换标点
    string = string.strip()
    string = string.split(' ')
    if len(string) > 1:
        string = string[0]
        string = re.sub(r'[0-9]\S*\b', '', string)  # 屏蔽设备序号
        return string
    string = string[0]
    string = re.sub(r'\b\S*\d\S*\b', '', string)  # 去除含数字的组合
    string = re.sub(r'\b[a-z]\b', '', string)  # 过滤单独字母
    return string


def StateClean(string):
    string = str(string)
    string = string.lower()
    string = string.strip()
    string = re.sub(r'\(\S*\)', '', string)

    string = re.sub(r'\b\S*:', '', string)
    string = re.sub(r'/\S*\b', '', string)
    string = re.sub(r'[0-9]\S*\b', '', string)
    string = re.sub(r'[^a-z0-9]', ' ', string)
    string = re.sub(r' +', ' ', string)

    string = string.strip()
    return string


def LogClean(df):
    df['msg01'] = df['msg'].apply(lambda x: x.split('|')[0] if len(x.split('|')) > 0 else '')
    df['msg02'] = df['msg'].apply(lambda x: x.split('|')[1] if len(x.split('|')) > 1 else '')
    df['msg03'] = df['msg'].apply(lambda x: x.split('|')[2] if len(x.split('|')) > 2 else '')

    df['Device'] = df['msg01'].apply(lambda x: DeviceClean(x))
    df['State'] = df['msg02'].apply(lambda x: StateClean(x))
    df['msg'] = df.apply(lambda x: x['Device'] + ' ' + x['State'], axis=1)
    return df[['sn', 'time', 'msg', 'server_model']]


def getEmbedding(log, logid, num):
    logid = list(map(int, logid.split(' ')))
    templateid = log['EventId'].values[logid]

    templateid = np.eye(num)[templateid]
    templateid = templateid.max(axis=0)
    return templateid


def getTimeFeature(log, fault_time, logid):
    logid = list(map(int, logid.split(' ')))
    df = log.iloc[logid, :][['timestamp']]
    df = df.sort_values('timestamp')

    df['delta'] = df['timestamp'].apply(lambda x: (x - fault_time).total_seconds())
    df['delta_diff'] = df['delta'].diff(1)

    min_delta = np.min(np.abs(df['delta'].values))
    max_delta = np.max(np.abs(df['delta'].values))
    mean_delta = np.mean(np.abs(df['delta'].values))
    std_delta = np.std(np.abs(df['delta'].values))

    num_log = len(logid)
    span_delta = df['delta'].values[-1] - df['delta'].values[0]
    before = df['delta'].values[np.argmin(np.abs(df['delta'].values))] / min_delta

    if num_log > 1:
        min_delta_diff = np.min(df['delta_diff'].values[1:])
        max_delta_diff = np.max(df['delta_diff'].values[1:])
        mean_delta_diff = np.mean(df['delta_diff'].values[1:])
        std_delta_diff = np.std(df['delta_diff'].values[1:])
    else:
        min_delta_diff = 0
        max_delta_diff = 0
        mean_delta_diff = 0
        std_delta_diff = 0

    return [num_log, span_delta, before,
            min_delta, max_delta, mean_delta, std_delta,
            min_delta_diff, max_delta_diff, mean_delta_diff, std_delta_diff]


def getSM(log, logid):
    logid = list(map(int, logid.split(' ')))
    SM = log.iloc[logid, :]['server_model'].values

    # 返回众数
    counts = np.bincount(SM)
    SM = np.argmax(counts)
    return SM


def SelFeatureConstruct(data, submit, train=True):
    # 日志清洗
    print('  日志清洗...')
    data = LogClean(data)

    # 日志模版提取
    print('  模板提取...')
    data = LogParse(data, train=train)

    # 日志对齐
    print('  日志对齐...')
    if train:
        logid = pd.read_csv('./user_data/tmp_data/train_sel_logid.csv')
    else:
        logid = getLogID(data, submit)

    template = joblib.load('./user_data/model_data/sel_logParser_EventId2num.pkl')
    n_template = len(template) + 1

    # 特征构造
    print('  特征构造...')
    logid['fault_time'] = pd.to_datetime(logid['fault_time'])
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    logid['EventId'] = logid['LogID'].apply(lambda x: getEmbedding(data, x, n_template))
    for i in range(n_template):
        logid['SEL_Event_Id_' + str(i)] = logid['EventId'].apply(lambda x: x[i])

    logid['TimeFeature'] = logid.apply(lambda x: getTimeFeature(data, x['fault_time'], x['LogID']), axis=1)

    logid['SEL_num_log'] = logid['TimeFeature'].apply(lambda x: x[0])
    logid['SEL_span_delta'] = logid['TimeFeature'].apply(lambda x: x[1])
    logid['SEL_before'] = logid['TimeFeature'].apply(lambda x: x[2])
    logid['SEL_min_delta'] = logid['TimeFeature'].apply(lambda x: x[3])
    logid['SEL_max_delta'] = logid['TimeFeature'].apply(lambda x: x[4])
    logid['SEL_mean_delta'] = logid['TimeFeature'].apply(lambda x: x[5])
    logid['SEL_std_delta'] = logid['TimeFeature'].apply(lambda x: x[6])
    logid['SEL_min_delta_diff'] = logid['TimeFeature'].apply(lambda x: x[7])
    logid['SEL_max_delta_diff'] = logid['TimeFeature'].apply(lambda x: x[8])
    logid['SEL_mean_delta_diff'] = logid['TimeFeature'].apply(lambda x: x[9])
    logid['SEL_std_delta_diff'] = logid['TimeFeature'].apply(lambda x: x[10])

    logid['SEL_server_model'] = logid['LogID'].apply(lambda x: getSM(data, x))

    logid.drop(['LogID', 'EventId', 'TimeFeature'], axis=1, inplace=True)
    return logid
