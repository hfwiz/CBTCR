#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
Name        : IJSEKE-CBTCR-code.py
Author      : wiz
Created     : 2024/1/7 22:16
Version     : 1.0
Modified    : 2024/1/7 22:16
Description : some words
"""

import gzip
import collections
import re

import gzip
import collections
import re
import itertools
import pandas as pd
import numpy as np
import random
import time
import shutil
import os
import matplotlib.pylab as plt
import scipy.stats as stats
import csv


def return_first_n_data():
    return 0


def mkdir(path):
    folder = os.path.exists(path)
    if not folder: os.makedirs(path)


def read_exam_reduce():
    filename_path = './=Result/' + base_technique + '/exam/' + str(reduce_rate) + '/'
    filename = filename_path + str(reduce_rate) + '-' + formula + '.csv'

    df = pd.read_csv(filename)
    exam_random = df.iloc[:, 5].values.tolist()
    exam_ietcr = df.iloc[:, 6].values.tolist()
    exam_ftmes = df.iloc[:, 7].values.tolist()
    exam_ctcr = df.iloc[:, 8].values.tolist()

    return exam_random, exam_ietcr, exam_ftmes, exam_ctcr


def read_exam_reduce_formula():
    filename_path = './=Result/' + base_technique + '-formula' + '/exam/' + str(reduce_rate) + '/'
    filename = filename_path + str(reduce_rate) + '-' + formula + '.csv'

    df = pd.read_csv(filename)
    # df_list = df.values.tolist()

    exam_tarantula = df.iloc[:, 5].values.tolist()
    exam_ochiai = df.iloc[:, 6].values.tolist()
    exam_dstar2 = df.iloc[:, 7].values.tolist()
    exam_gp13 = df.iloc[:, 8].values.tolist()

    exam_opt2 = df.iloc[:, 9].values.tolist()

    return exam_tarantula, exam_ochiai, exam_dstar2, exam_gp13, exam_opt2


def read_topnmap_reduce():
    filename_path = './=Result/' + base_technique + '/topn-map/' + str(reduce_rate) + '/'
    filename = filename_path + str(reduce_rate) + '-' + formula + '-topn-map.csv'

    df = pd.read_csv(filename, header=None)
    df_list = df.values.tolist()

    if RQ_index == 1: technique_list = ['random', 'ietcr', 'FTMES', 'ctcr-dstar2']
    if RQ_index == 2 or RQ_index == 3: technique_list = ['ctcr-dstar2']
    # technique_list = ['FTMES','ctcr']

    RQ_name = 'RQ' + str(RQ_index)

    _path = './==Paper_results/' + RQ_name + '/' + 'topn-map' + '-' + RQ_name + '.csv'
    mkdir('./==Paper_results/' + RQ_name + '/')

    for technique in technique_list:
        all_top1 = 0
        all_top3 = 0
        all_top5 = 0
        avg_map = 0
        tmp = 0
        for line in df_list:
            technique_name = line[2]
            if technique == technique_name:
                tmp += 1
                all_top1 += int(line[3])
                all_top3 += int(line[4])
                all_top5 += int(line[5])
                avg_map += float(line[7])

        write_csv(_path,
                  ['Title', 'Technique', 'MBFL Formula', 'Ratio', 'Reduce Strategy', 'Top-1', 'Top-3', 'Top-5', 'MAP'],
                  [' ', base_technique, formula, reduce_rate, technique, all_top1, all_top3, all_top5,
                   np.around((avg_map) / tmp, 4)])

    return 0


def read_topnmap_reduce_formula():
    filename_path = './=Result/' + base_technique + '-formula' + '/topn-map/' + str(reduce_rate) + '/'
    filename = filename_path + str(reduce_rate) + '-' + formula + '-topn-map.csv'

    df = pd.read_csv(filename, header=None)
    df_list = df.values.tolist()

    technique_list = ['ctcr-tarantula', 'ctcr-ochiai', 'ctcr-dstar2', 'ctcr-gp13', 'ctcr-opt2']


    RQ_name = 'RQ' + str(RQ_index)

    _path = './==Paper_results/' + RQ_name + '/' + 'topn-map' + '-' + RQ_name + '.csv'

    for technique in technique_list:
        all_top1 = 0
        all_top3 = 0
        all_top5 = 0
        avg_map = 0
        tmp = 0
        for line in df_list:
            technique_name = line[2]
            if technique == technique_name:
                tmp += 1
                all_top1 += int(line[3])
                all_top3 += int(line[4])
                all_top5 += int(line[5])
                avg_map += float(line[7])

        print(technique, all_top1, all_top3, all_top5, np.around((avg_map) / tmp, 4))
        write_csv(_path,
                  ['Title', 'Technique', 'MBFL Formula', 'Ratio', 'Contribution formula', 'Top-1', 'Top-3', 'Top-5',
                   'MAP'],
                  [' ', base_technique, formula, reduce_rate, technique, all_top1, all_top3, all_top5,
                   np.around((avg_map) / tmp, 4)])
    return 0


def read_exam_base_technique():
    filename = '../==Result/exam/' + formula + '.csv'

    df = pd.read_csv(filename)

    filename_sbfl = '../==Result/exam/' + 'dstar2' + '.csv'

    df_sbfl = pd.read_csv(filename_sbfl)

    exam_sbfl = df_sbfl.iloc[:, 5].values.tolist()
    exam_mbfl = df.iloc[:, 6].values.tolist()
    exam_muse = df.iloc[:, 7].values.tolist()
    exam_mcbfl = df.iloc[:, 8].values.tolist()

    return exam_sbfl, exam_mbfl, exam_muse, exam_mcbfl


def read_topnmap_base_technique():
    filename = '../==Result/topn-map/' + formula + '-topn-map.csv'

    df = pd.read_csv(filename, header=None)
    df_list = df.values.tolist()

    technique_list = ['SBFL', 'MBFL', 'MUSE', 'MCBFL']

    if RQ_index == 2: technique_list = [base_technique]
    if RQ_index == 3: technique_list = ['SBFL', 'MUSE']

    RQ_name = 'RQ' + str(RQ_index)

    _path = './==Paper_results/' + RQ_name + '/' + 'topn-map' + '-' + RQ_name + '.csv'

    for technique in technique_list:
        all_top1 = 0
        all_top3 = 0
        all_top5 = 0
        avg_map = 0
        tmp = 0
        for line in df_list:
            technique_name = line[2]
            if technique == technique_name:
                tmp += 1
                all_top1 += int(line[3])
                all_top3 += int(line[4])
                all_top5 += int(line[5])
                avg_map += float(line[7])

        write_csv(_path,
                  ['Title', 'Technique', 'MBFL Formula', 'Ratio', 'Reduce Strategy', 'Top-1', 'Top-3', 'Top-5', 'MAP'],
                  [' ', base_technique, formula, reduce_rate, technique, all_top1, all_top3, all_top5,
                   np.around((avg_map) / tmp, 4)])

    return 0


def exam(exam_fom):
    sort_data = sorted(exam_fom)

    count_per = 0

    y_axis = []
    x_axis = []
    former = 0
    fault_num = len(sort_data)
    for i in sort_data:
        if i != former:
            y_axis.append(count_per / fault_num)
            x_axis.append(former)
            former = i
        count_per += 1


    return x_axis, y_axis


def plot_box_percent():
    import matplotlib.pyplot as plt

    exam_all = [exam_random, exam_ietcr, exam_ftmes, exam_ctcr]

    fig = plt.figure(figsize=(8, 6))

    bplot = plt.boxplot(exam_all,
                        notch=False,  # notch shape
                        vert=True,  # vertical box aligmnent
                        patch_artist=True)  # fill with color
    colors = ['#A0C9F2', '#FFB485', '#D0BCFC', '#8DE4A4']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.xticks([y + 1 for y in range(len(exam_all))],
               ['SBFL', 'Metallaxis', 'MCBFL-hybrid-avg', 'ImpHMBFL'])
    # plt.xlabel('Techniques')
    plt.ylabel(u'EXAM', fontsize=y_title_size)

    plt.savefig('RQ3-' + formula + '.pdf', bbox_inches='tight')


def write_csv(csvname, header, row):
    if not os.path.exists(csvname):
        with open(csvname, 'w') as f:
            write = csv.writer(f)
            write.writerow(header)
            write.writerow(row)
    else:
        with open(csvname, 'a+') as f:
            write = csv.writer(f)
            write.writerow(row)


def pvalue():
    RQ_name = 'RQ' + str(RQ_index)

    _path = './==Paper_results/' + RQ_name + '/' + 'p-values' + '-' + RQ_name + '.csv'

    if RQ_index == 5:
        u_statistic, p1 = stats.wilcoxon(exam_dstar2, exam_gp13)
        u_statistic, p2 = stats.wilcoxon(exam_dstar2, exam_ochiai)
        u_statistic, p3 = stats.wilcoxon(exam_dstar2, exam_opt2)
        u_statistic, p4 = stats.wilcoxon(exam_dstar2, exam_tarantula)

        if p1 > 0.0001: p1 = np.around(p1, 4)
        if p2 > 0.0001: p2 = np.around(p2, 4)
        if p3 > 0.0001: p3 = np.around(p3, 4)
        if p4 > 0.0001: p4 = np.around(p4, 4)

        write_csv(_path, ['Title', 'Technique', 'MBFL Formula', 'Ratio', 'GP13', 'Ochiai', 'OP2', 'Tarantula'],
                  ['RQ5-contribution formula', base_technique, formula, reduce_rate, p1, p2, p3, p4])

    if RQ_index == 3:
        # 基准为 Dstar方法
        x_30 = (ctcr_different_reduce_rate_exam[0])
        less_num = len(x_30) - len(exam_mbfl)



        u_statistic, p1 = stats.wilcoxon(x_30, exam_sbfl)
        u_statistic, p2 = stats.wilcoxon(x_30, exam_muse)
        # u_statistic, p3 = stats.wilcoxon(exam_ctcr, exam_ftmes)
        # u_statistic, p4 = stats.wilcoxon(exam_dstar2, exam_ctcr)

        if p1 > 0.0001: p1 = np.around(p1, 4)
        if p2 > 0.0001: p2 = np.around(p2, 4)


        write_csv(_path, ['Title', 'Technique', 'MBFL Formula', 'Ratio', 'SBFL', 'MUSE'],
                  ['RQ3-different techniques', base_technique, formula, reduce_rate, p1, p2])

    if RQ_index == 1:
        u_statistic, p1 = stats.wilcoxon(exam_ctcr, exam_random)
        u_statistic, p2 = stats.wilcoxon(exam_ctcr, exam_ietcr)
        u_statistic, p3 = stats.wilcoxon(exam_ctcr, exam_ftmes)

        if p1 > 0.0001: p1 = np.around(p1, 4)
        if p2 > 0.0001: p2 = np.around(p2, 4)
        if p3 > 0.0001: p3 = np.around(p3, 4)

        write_csv(_path, ['Title', 'Technique', 'MBFL Formula', 'Ratio', 'Random', 'IETCR', 'FTMES'],
                  ['RQ1-different reduce strategy', base_technique, formula, reduce_rate, p1, p2, p3])

    if RQ_index == 2:
        x_10 = (ctcr_different_reduce_rate_exam[0])
        x_20 = (ctcr_different_reduce_rate_exam[1])
        x_30 = (ctcr_different_reduce_rate_exam[2])


        if base_technique == 'MBFL':
            u_statistic, p1 = stats.wilcoxon(exam_mbfl, x_10)
            u_statistic, p2 = stats.wilcoxon(exam_mbfl, x_20)
            u_statistic, p3 = stats.wilcoxon(exam_mbfl, x_30)

        if base_technique == 'MCBFL':
            u_statistic, p1 = stats.wilcoxon(exam_mcbfl, x_10)
            u_statistic, p2 = stats.wilcoxon(exam_mcbfl, x_20)
            u_statistic, p3 = stats.wilcoxon(exam_mcbfl, x_30)

        if p1 > 0.0001: p1 = np.around(p1, 4)
        if p2 > 0.0001: p2 = np.around(p2, 4)
        if p3 > 0.0001: p3 = np.around(p3, 4)

        write_csv(_path, ['Title', 'Technique', 'MBFL Formula', 'CBTCR-10%', 'CBTCR-20%', 'CBTCR-30%'],
                  ['RQ3-different rates and orginal', base_technique, formula, p1, p2, p3])


def MTP():
    import matplotlib.pyplot as plt
    color = "darkgrey"

    file_path = './==Paper_results/' + 'MTP-ext'

    # num_list = [1.5,0.6,7.8,6]
    fig, ax = plt.subplots(figsize=(8, 6))

    num_list = [1302686503, 1140699782, 1009323347, 1302686503, 4951970439]  # SCI
    num_list = [i / 1000000 for i in num_list]

    name_list = ['RAND', 'IETCR', 'FTMES', 'CBTCR', 'Original']
    b = ax.bar(name_list, num_list)

    colors = ['white', 'white', 'white', 'white', 'black']

    # / 、\、 | 、-、+、x、o、O、.、 *
    # hatch = ['-', '++++', 'xxxx', '\\\\\\', '////', '...', '.']
    hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    hatch = ['-', '++++', 'xxxx', '\\\\\\', '////', '...']

    # ax.set_yticks(range(0,1000,5000))
    ax.set_ylim(0, 5000)

    ax.grid(alpha=0.3, axis='y')

    plt.bar(range(len(num_list)), num_list, color=colors, tick_label=name_list)

    bars = ax.patches
    hatches = ['-', '++++', 'xxxx', '\\\\\\', '////', '...', '-', '++++', 'xxxx', '\\\\\\', '////', '...']
    hatches = ['----', '++++', 'xxxx', '\\\\\\', '////', '...', '----', '++++', 'xxxx', '\\\\\\', '////', '...']
    hatches = ['----', 'xxxx', '/////', '...', '\\\\\\', 'oo', '----', 'xxxx', '/////', '...', '\\\\\\', 'oo']
    hatches = ['----', 'xxxx', '/////', '...', '\\\\\\', '----', 'xxxx', '/////', '...', '\\\\\\']

    for bar, hatch in zip(bars, hatches):  # loop over bars and hatches to set hatches in correct order
        bar.set_hatch(hatch)


    plt.xlabel("Different Reduction Strategies", fontsize=16)
    plt.ylabel("MTP(million)", fontsize=16)
    plt.title("", fontsize=20)
    plt.savefig(file_path + '.pdf', bbox_inches='tight')


def MTP2():
    df = pd.DataFrame(np.random.randint(0, 20, size=(5, 2)), columns=list('AB'))
    fig, ax = plt.subplots()
    ax = df.sort_values('B', ascending=True).plot.barh(rot=0, ax=ax)
    # get all bars in the plot
    bars = ax.patches
    patterns = ['/', '|']  # set hatch patterns in the correct order
    hatches = []  # list for hatches in the order of the bars
    for h in patterns:  # loop over patterns to create bar-ordered hatches
        for i in range(int(len(bars) / len(patterns))):
            hatches.append(h)
    for bar, hatch in zip(bars, hatches):  # loop over bars and hatches to set hatches in correct order
        bar.set_hatch(hatch)
    # generate legend. this is important to set explicitly, otherwise no hatches will be shown!
    ax.legend()
    plt.show()


def read_time_mtp_csv(filename):
    df = pd.read_csv(filename)

    if 'random' in filename: _path = '==Paper_results/1-random.csv'
    if 'ctcr' in filename: _path = '==Paper_results/1-ctcr.csv'
    if 'ietcr' in filename: _path = '==Paper_results/1-ietcr.csv'

    flag = 1
    for pid in pid_list:
        used_mtp_list = []
        used_time_list = []
        mutant_num_list = []
        all_time_list = []
        all_mtp_list = []
        fail_time_list = []
        ftmes_mtp_list = []
        used_pass_test_num_list = []
        all_test_list = []

        fail_test_list = []
        pass_test_list = []

        for i in range(len(df)):
            line = df.iloc[i, :].values.tolist()
            pid_name = line[0]

            if pid_name == pid:
                fail_test = line[3]
                mutant_num = line[8]
                used_mtp = line[10]
                all_mtp = line[9]
                all_time = line[13]
                used_time = line[14]
                fail_time = line[18]
                used_pass_test_num = line[6]
                all_test = line[2]
                pass_test = line[4]

                # if pid == 'Compress':
                # print(fail_time,used_time+fail_time)

                used_mtp_list.append(used_mtp + fail_test * mutant_num)
                used_time_list.append(used_time + fail_time)
                mutant_num_list.append(mutant_num)
                all_time_list.append(all_time)
                all_mtp_list.append(all_mtp + fail_test * mutant_num)
                fail_time_list.append(fail_time)
                ftmes_mtp_list.append(fail_test * mutant_num)
                used_pass_test_num_list.append(used_pass_test_num)
                all_test_list.append(all_test)

                fail_test_list.append(fail_test)
                pass_test_list.append(pass_test)

        header = ['Project', 'Version No.', 'Used MTP', 'All MTP', 'FTMES MTP', 'Used Time(min)', 'All Time(min)',
                  'Fail Time(min)', 'All Mutants', 'Used Pass No.', 'All tests', 'All Fail tests', 'All Pass tests']
        row = [pid, str(len(used_mtp_list)), str(sum(used_mtp_list)), str(sum(all_mtp_list)), str(sum(ftmes_mtp_list)),
               str(np.around(sum(used_time_list), 2)), str(np.around(sum(all_time_list), 2)),
               str(np.around(sum(fail_time_list), 2)), str(sum(mutant_num_list)), str(sum(used_pass_test_num_list)),
               str(sum(all_test_list)), str(sum(fail_test_list)), str(sum(pass_test_list))]
        write_csv(_path, header, row)

        print(pid, sum(used_mtp_list), sum(used_time_list), sum(fail_time_list))


def read_time_map():
    read_time_mtp_csv('==Paper_results/ctcr.csv')


def read_project_lines():
    filename = 'sloc.csv'
    df = pd.read_csv(filename)

    _path = '==Paper_results/LineofCode.csv'

    flag = 1
    for pid in pid_list:
        slocLoadedClasses_list, slocTotal_list = [], []

        for i in range(len(df)):
            line = df.iloc[i, :].values.tolist()
            pid_name = line[0]

            if pid_name == pid:
                slocLoadedClasses = line[2]
                slocTotal = line[3]

                slocLoadedClasses_list.append(slocLoadedClasses)
                slocTotal_list.append(slocTotal)
        print(pid)
        print(type(slocTotal_list[0]), type(slocLoadedClasses_list[0]))

        header = ['Project', 'Version No.', 'slocLoadedClasses(Avg.)', 'slocLoadedClasses(All)', 'slocTotal(Avg.)',
                  'slocTotal(All)']


        row = [pid, str(len(slocTotal_list)), (np.mean(np.array(slocLoadedClasses_list))),
               str(sum(slocLoadedClasses_list)), (np.mean(np.array(slocTotal_list))), str(sum(slocTotal_list))]
        write_csv(_path, header, row)


if __name__ == '__main__':
    title_size = 20
    y_title_size = 18
    xy_axis_size = 15

    from matplotlib import rcParams

    config = {
        "font.family": 'Times New Roman',
        "font.size": xy_axis_size,
    }
    rcParams.update(config)

    Formulas_used = ['tarantula', 'ochiai', 'barinel', 'dstar2', 'gp13', 'opt2']

    pid_list = ['Lang', 'Chart', 'Time', 'Math', 'Closure', 'Cli', 'Codec', 'Compress',
                'Csv', 'Gson', 'JacksonCore', 'JacksonXml', 'Jsoup', 'JxPath']

    # pid_list = [ 'Compress']
    pid_list_lower = ['lang', 'chart', 'time', 'math', 'closure', 'cli', 'codec', 'compress',
                      'csv', 'gson', 'jacksonCore', 'jacksonXml', 'jsoup', 'jxPath']


    bid_list = [65, 26, 27, 106, 133, 39, 18, 47, 16, 18, 26, 6, 93, 22]
    Techniques = ['SBFL', 'MBFL', 'MUSE', 'MCBFL', 'FTMES']

    reduce_rate_list = [0.1, 0.2, 0.3]


    #####  test code  #####
    Formulas_used = ['dstar2', 'gp13', 'ochiai']



    base_techniques = ['MBFL', 'MCBFL']


    RQ_index = 4

    if RQ_index == 3 or RQ_index == 5: reduce_rate_list = [0.1]
    if RQ_index == 4:
        MTP()

    # read_project_lines()

    for base_technique in base_techniques:
        for formula in Formulas_used:

            ctcr_different_reduce_rate_exam = []
            ctcr_different_reduce_rate_topnmap = []

            for reduce_rate in reduce_rate_list:
                print('=============================>>>>>>>> 约减率：', reduce_rate, base_technique, formula)
                exam_random, exam_ietcr, exam_ftmes, exam_ctcr = read_exam_reduce()
                exam_sbfl, exam_mbfl, exam_muse, exam_mcbfl = read_exam_base_technique()
                exam_tarantula, exam_ochiai, exam_dstar2, exam_gp13, exam_opt2 = read_exam_reduce_formula()

                ctcr_different_reduce_rate_exam.append(exam_ctcr)

                if RQ_index == 1:
                    pvalue()
                    read_topnmap_base_technique()
                    read_topnmap_reduce()

                    # read_topnmap_reduce_formula()
                if RQ_index == 3:

                    pvalue()
                    # read_topnmap_base_technique()
                    # read_topnmap_reduce()

                if RQ_index == 5:
                    pvalue()
                    read_topnmap_base_technique()
                    read_topnmap_reduce()
                    read_topnmap_reduce_formula()


            if RQ_index == 2:
                pvalue()
                read_topnmap_base_technique()
