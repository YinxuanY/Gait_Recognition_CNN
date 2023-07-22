# -*- coding: utf-8 -*-
# @Author  : Abner
# @Time    : 2018/12/19

from time import sleep
import argparse
import os
from f import cut_pickle
from f import boolean_string
from multiprocessing import Pool
from multiprocessing import TimeoutError as MP_TimeoutError


if __name__ == '__main__':
    START = "START"
    FINISH = "FINISH"
    WARNING = "WARNING"
    FAIL = "FAIL"





    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--input_path', default='C:/Users/61705/gait/CB', type=str,
                        help='Root path of raw dataset.')
    parser.add_argument('--output_path', default='C:/Users/61705/gait/Train', type=str,
                        help='Root path for output.')
    parser.add_argument('--log_file', default='./pretreatment.log', type=str,
                        help='Log file path. Default: ./pretreatment.log')
    parser.add_argument('--log', default=False, type=boolean_string,
                        help='If set as True, all logs will be saved. '
                             'Otherwise, only warnings and errors will be saved.'
                             'Default: False')
    parser.add_argument('--worker_num', default=1, type=int,
                        help='How many subprocesses to use for data pretreatment. '
                             'Default: 1')
    opt = parser.parse_args()

    INPUT_PATH = opt.input_path
    OUTPUT_PATH = opt.output_path
    IF_LOG = opt.log
    LOG_PATH = opt.log_file
    WORKERS = opt.worker_num

    T_H = 64
    T_W = 64








    pool = Pool(WORKERS)
    results = list()
    pid = 0

    print('Pretreatment Start.\n'
          'Input path: %s\n'
          'Output path: %s\n'
          'Log file: %s\n'
          'Worker num: %d' % (
              INPUT_PATH, OUTPUT_PATH, LOG_PATH, WORKERS))

    id_list = os.listdir(INPUT_PATH)
    id_list.sort()
    # Walk the input path
    for _id in id_list:
        seq_type = os.listdir(os.path.join(INPUT_PATH, _id))
        seq_type.sort()
        for _seq_type in seq_type:
            view = os.listdir(os.path.join(INPUT_PATH, _id, _seq_type))
            view.sort()
            for _view in view:
                seq_info = [_id, _seq_type, _view]
                out_dir = os.path.join(OUTPUT_PATH, *seq_info)
                os.makedirs(out_dir)
                results.append(
                    pool.apply_async(
                        cut_pickle,
                        args=(seq_info, pid, START,INPUT_PATH,OUTPUT_PATH,WARNING,T_H,T_W,FINISH,FAIL,LOG_PATH)))
                sleep(0.02)
                pid += 1

    pool.close()
    unfinish = 1
    while unfinish > 0:
        unfinish = 0
        for i, res in enumerate(results):
            try:
                res.get(timeout=0.1)
            except Exception as e:
                if type(e) == MP_TimeoutError:
                    unfinish += 1
                    continue
                else:
                    print('\n\n\nERROR OCCUR: PID ##%d##, ERRORTYPE: %s\n\n\n',
                          i, type(e))
                    raise e
    pool.join()





