#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
Name        : MBFL-test-reduction-CBTCR-code.py
Author      : WiZ
Created     : 2024/2/25 21:08
Version     : 2.3
Description : some words
"""

import gzip
import collections
import re
import itertools
import pandas as pd
import numpy as np
import random
import time
import shutil

PASS = 'PASS'
FAIL = 'FAIL'

def distill_type(trace):
  m = re.match(r'[\w.$]+', trace)
  return (m.group() if m else trace)

def distill_type_message(trace):
  m = re.match(r'(?P<first_line>.*?) at [\w.$]+\([^)]*\.java:\d+\)', trace)
  return (m.group('first_line') if m else trace)

def distill_type_message_location(trace):
  m = re.match(r'(?P<first_line>.*?) at (?P<location>[\w.$]+\([^)]*\.java:\d+\))', trace)
  return (m.group('first_line', 'location') if m else trace)

STACK_TRACE_DISTILLING_FUNCTIONS = {
  'all': (lambda trace: ''),
  'type': distill_type,
  'type+message': distill_type_message,
  'type+message+location': distill_type_message,
  'exact': (lambda trace: trace)
}
ERROR_PARTITION_SCHEMES = set(STACK_TRACE_DISTILLING_FUNCTIONS.keys())
ERROR_PARTITION_SCHEMES.add('passfail')

Outcome = collections.namedtuple(
  'Outcome',
  ('test_case', 'mutant_id', 'timeout',
   'category', 'runtime', 'output_hash', 'covered_mutants', 'stack_trace'))

def parse_outcome_line(line):
  result = Outcome(*line.split(',', 7))
  return result._replace(
    mutant_id=int(result.mutant_id),
    timeout=int(result.timeout),
    runtime=int(result.runtime),
    covered_mutants=(set(int(n) for n in result.covered_mutants.split(' ')) if result.covered_mutants else set()))

def are_outcomes_equivalent(outcome1, outcome2, error_partition_scheme):
  if error_partition_scheme == 'passfail':
    pass_fail = '0'
    if outcome1.category=='PASS' and outcome2.category=='FAIL':
        pass_fail = '1'
    if outcome1.category=='FAIL' and outcome2.category=='PASS': pass_fail = '2'
    return pass_fail

  if outcome1.category == outcome2.category == FAIL:
    key = STACK_TRACE_DISTILLING_FUNCTIONS[error_partition_scheme]
    return key(outcome1.stack_trace) == key(outcome2.stack_trace)
  else:
    return outcome1.category == outcome2.category

def find_killed_mutants(original_outcome, mutated_outcomes, error_partition_scheme):
    return set(outcome.mutant_id for outcome in mutated_outcomes if not are_outcomes_equivalent(outcome, original_outcome, error_partition_scheme))

# ================================== MUSE
def find_killed_mutants_muse(original_outcome, mutated_outcomes, error_partition_scheme):
    tmp = 1
    mutants = set()
    pass_fail_list = []
    for outcome in mutated_outcomes:

        mutants.add(outcome.mutant_id)
        pass_fail  = (are_outcomes_equivalent(outcome, original_outcome, error_partition_scheme))
        pass_fail_list.append(pass_fail)
    return mutants,pass_fail_list



def format_kill_matrix_row_muse(killed_mutants,pass_fail_list, n_mutants, originally_passing):
  words = []
  for i in range(1, n_mutants + 1):
      if i in killed_mutants:
          m_ind = list(killed_mutants).index(i)
          words.append(pass_fail_list[m_ind])
      else:
          words.append('0')
  words.append('+' if originally_passing else '-')
  return ' '.join(words)

def format_kill_matrix_row(killed_mutants, n_mutants, originally_passing):
  words = ['1' if i in killed_mutants else '0' for i in range(1, n_mutants+1)]
  words.append('+' if originally_passing else '-')
  return ' '.join(words)

def group_outcomes_by_test_case(outcomes):
  for _test_case, group in itertools.groupby(outcomes, key=(lambda outcome: outcome.test_case)):
    original_outcome = next(group)
    if original_outcome.mutant_id != 0:
      raise ValueError('expected first outcome for test case to be have mutant_id 0, but was not: {}'.format(original_outcome))
    yield (original_outcome, group)

def count_mutants(mutants_file):
  return max(
    int(match.group()) for match in (
      re.match(r'\d+(?=:)', line) for line in mutants_file)
    if match)

def open_killmap(path):
  return gzip.open(path) if path.endswith('.gz') else open(path)


def unzip_gz(file_name):
    f_name = file_name.replace(".gz", "")
    g_file = gzip.GzipFile(file_name)
    open(f_name, "wb+").write(g_file.read())
    g_file.close()


def outcome_matrix_to_kill_matrix(bid_num):
    for bid in range(1, bid_num + 1):
        # for bid in range(1, 5):
        bid = str(bid)
        print('====>>>>', formula, Formulas_used.index(formula)+1, '/', len(Formulas_used), '---------', pid,
              pid_list.index(pid)+1, '/', len(pid_list), '----', bid, '/', bid_num)

        try:
            data_path = './data/' + pid_lower + '/' + bid + '/'
            stmt_cov_path = './killmap/' + pid_lower + '/'
            if not os.path.exists(stmt_cov_path): os.mkdir(stmt_cov_path)

            error_partition_scheme = 'exact'
            outcomes = data_path + pid_lower + '-' + (bid) + '-matrix'
            unzip_gz(outcomes + '.gz')
            mutants = data_path + pid + '-' + (bid) + '-mutants.log'
            output = stmt_cov_path + pid + '-' + (bid) + '-killmap'

            # =====================

            import argparse
            import itertools

            parser = argparse.ArgumentParser()
            parser.add_argument('--error-partition-scheme', choices=ERROR_PARTITION_SCHEMES,
                                default=error_partition_scheme)
            parser.add_argument('--outcomes', required=False, help='path to the outcome matrix produced by Killmap',
                                default=outcomes)
            parser.add_argument('--mutants', required=False, help='path to a Major mutants.log file', default=mutants)
            parser.add_argument('--output', required=False, help='file to write output matrix to', default=output)

            args = parser.parse_args()

            with open(args.mutants) as mutants_file:
                n_mutants = count_mutants(mutants_file)

            with open_killmap(args.outcomes) as outcome_matrix_file, open(args.output, 'w') as output_file:
                all_outcomes = (parse_outcome_line(line) for line in outcome_matrix_file)
                for original_outcome, mutated_outcomes in group_outcomes_by_test_case(all_outcomes):
                    killed_mutants = find_killed_mutants(original_outcome, mutated_outcomes,
                                                         args.error_partition_scheme)
                    output_file.write(
                        format_kill_matrix_row(killed_mutants, n_mutants, original_outcome.category == PASS))
                    output_file.write('\n')

        except Exception as e:
            print('Something wrong in:', pid, bid,e)


# ==================2.crush matrix================================
import collections
import re
import sys
import traceback

from formulas import *

def crush_row(formula, hybrid_scheme, passed, failed, totalpassed, totalfailed, passed_covered=None, failed_covered=None, totalpassed_covered=0.0, totalfailed_covered=0.0):
  try:
    if hybrid_scheme is None:
      return FORMULAS[formula](passed, failed, totalpassed, totalfailed)
    elif hybrid_scheme == 'numerator':
      return HYBRID_NUMERATOR_FORMULAS[formula](passed, failed, totalpassed, totalfailed, failed_covered > 0)
    elif hybrid_scheme == 'constant':
      return FORMULAS[formula](passed, failed, totalpassed, totalfailed) + (1 if failed_covered > 0 else 0)
    elif hybrid_scheme == 'mirror':
      return (FORMULAS[formula](passed, failed, totalpassed, totalfailed) +
              FORMULAS[formula](passed_covered, failed_covered, totalpassed_covered, totalfailed_covered))/2.
    elif hybrid_scheme == 'coverage-only':
      return FORMULAS[formula](passed_covered, failed_covered, totalpassed_covered, totalfailed_covered)
    raise ValueError('unrecognized hybrid scheme name: {!r}'.format(hybrid_scheme))
  except ZeroDivisionError as zeroDivisionError:
    # sys.stderr.write("Warn: catch integer division or modulo by zero for " + formula + "\n")
    # sys.stderr.write("Passed: " + str(passed) + "\nFailed: " + str(failed) + "\nTotalPassed: " + str(totalpassed) + "\nTotalFailed: " + str(totalfailed) + "\n")
    return 0
  except:
    traceback.print_exc()
    # sys.stderr.write("Passed: " + str(passed) + "\nFailed: " + str(failed) + "\nTotalPassed: " + str(totalpassed) + "\nTotalFailed: " + str(totalfailed) + "\n")
    sys.exit(1)

def suspiciousnesses_from_tallies(formula, hybrid_scheme, tally, hybrid_coverage_tally):
  '''Returns a dict mapping element-number to suspiciousness.
  '''
  if hybrid_coverage_tally is None:
    passed_covered = failed_covered = collections.defaultdict(lambda: None)
    totalpassed_covered = totalfailed_covered = 0
  else:
    passed_covered = hybrid_coverage_tally.passed
    failed_covered = hybrid_coverage_tally.failed
    totalpassed_covered = hybrid_coverage_tally.totalpassed
    totalfailed_covered = hybrid_coverage_tally.totalfailed

  return {
    element: crush_row(
      formula=formula, hybrid_scheme=hybrid_scheme,
      passed=float(tally.passed[element]), failed=float(tally.failed[element]),
      totalpassed=float(tally.totalpassed), totalfailed=float(tally.totalfailed),
      passed_covered=passed_covered[element], failed_covered=failed_covered[element],
      totalpassed_covered=float(totalpassed_covered), totalfailed_covered=float(totalfailed_covered))
    for element in range(tally.n_elements)}


TestSummary = collections.namedtuple('TestSummary', ('triggering', 'covered_elements'))
def parse_test_summary(line, n_elements):
  words = line.strip().split(' ')
  coverages, sign = words[:-1], words[-1]
  if len(coverages) != n_elements:
    raise ValueError("expected {expected} elements in each row, got {actual} in {line!r}".format(expected=n_elements, actual=len(coverages), line=line))
  return TestSummary(
    triggering=(sign == '-'),
    covered_elements=set(i for i in range(len(words)) if words[i]=='1'))

def parse_test_summary_muse(line, n_elements):
  # print(line)
  words = line.strip().split(' ')
  coverages, sign = words[:-1], words[-1]
  if len(coverages) != n_elements:
    raise ValueError("expected {expected} elements in each row, got {actual} in {line!r}".format(expected=n_elements, actual=len(coverages), line=line))
  return TestSummary(
    triggering=(sign == '-'),
    covered_elements=set(i for i in range(len(words)) if words[i]=='1'))




PassFailTally = collections.namedtuple('PassFailTally', ('n_elements', 'passed', 'failed', 'totalpassed', 'totalfailed'))
def tally_matrix(matrix_file, total_defn, n_elements):
  summaries = (parse_test_summary(line, n_elements) for line in matrix_file)

  passed = {i: 0 for i in range(n_elements)}
  failed = {i: 0 for i in range(n_elements)}
  totalpassed = 0
  totalfailed = 0
  for summary in summaries:
    if summary.triggering:
      totalfailed += (1 if total_defn == 'tests' else len(summary.covered_elements))
      for element_number in summary.covered_elements:
        failed[element_number] += 1
    else:
      totalpassed += (1 if total_defn == 'tests' else len(summary.covered_elements))
      for element_number in summary.covered_elements:
        passed[element_number] += 1


  return PassFailTally(n_elements, passed, failed, totalpassed, totalfailed)

def tally_matrix_muse(matrix_file, total_defn, n_elements):
  '''Returns a PassFailTally describing how many passing/failing tests there are, and how many of each cover each code element.

  ``total_defn`` may be "tests" (in which case the tally's ``totalpassed`` will be the number of passing tests) or "elements" (in which case it'll be the number of times a passing test covers a code element) (and same for ``totalfailed``).

  ``n_elements`` is the number of code elements that each row of the matrix should indicate coverage for.
  '''
  summaries = (parse_test_summary_muse(line, n_elements) for line in matrix_file)

  # print(n_elements)

  passed = {i: 0 for i in range(n_elements)}
  failed = {i: 0 for i in range(n_elements)}
  totalpassed = 0
  totalfailed = 0
  for summary in summaries:
    if summary.triggering:
      totalfailed += (1 if total_defn == 'tests' else len(summary.covered_elements))
      for element_number in summary.covered_elements:
        failed[element_number] += 1
    else:
      totalpassed += (1 if total_defn == 'tests' else len(summary.covered_elements))
      for element_number in summary.covered_elements:
        passed[element_number] += 1

  # print(totalpassed, totalfailed)
  return totalpassed, totalfailed



def crush_matrix(bid_num):
    # delete_csv(1)
    for bid in range(1, bid_num + 1):
        # for bid in range(1, 2):
        bid = str(bid)
        print('====>>>>', formula, Formulas_used.index(formula)+1, '/', len(Formulas_used), '---------', pid,
              pid_list.index(pid)+1, '/', len(pid_list), '----', bid, '/', bid_num)

        try:
            data_path = './==Test reduce==/SBFL/stmt-log/' + pid_lower + '/'
            stmt_cov_path = './==Test reduce==/SBFL/to-1-stmt/' + pid + '/'
            stmt_sus_path = formula_path + pid_lower + '/'
            if not os.path.exists(stmt_sus_path): os.mkdir(stmt_sus_path)
            # =====================
            matrix = stmt_cov_path + pid + '-' + (bid) + '-coverage'
            element_type = 'Mutant'
            element_names = data_path + pid + '-' + (bid) + '-log'
            total_defn = 'tests'
            output = stmt_sus_path + pid + '-' + bid + '-stmt-sus'
            if not os.path.exists(stmt_sus_path): os.mkdir(stmt_sus_path)
            import argparse
            import csv

            parser = argparse.ArgumentParser()
            parser.add_argument('--formula', required=False, default=formula, choices=set(FORMULAS.keys()))
            parser.add_argument('--matrix', required=False, default=matrix, help='path to the coverage/kill-matrix')
            parser.add_argument('--hybrid-scheme', choices=['numerator', 'constant', 'mirror', 'coverage-only'])
            # parser.add_argument('--hybrid-scheme', choices=['coverage-only'])
            parser.add_argument('--hybrid-coverage-matrix', help='optional coverage matrix for hybrid techniques')
            parser.add_argument('--element-type', required=False, default=element_type, choices=['Statement', 'Mutant'],
                                help='file enumerating names for matrix columns')
            # parser.add_argument('--element-type', required=True, choices=['Statement'], help='file enumerating names for matrix columns')
            parser.add_argument('--element-names', required=False, default=element_names,
                                help='file enumerating names for matrix columns')
            parser.add_argument('--total-defn', required=False, default=total_defn, choices=['tests', 'elements'],
                                help='whether totalpassed/totalfailed should counts tests or covered/killed elements')
            parser.add_argument('--output', required=False, default=output,
                                help='file to write suspiciousness vector to')

            args = parser.parse_args()

            # 'tarantula', '/Users/wanghaifeng/PycharmProjects/Study/Defects4J/root-Lang-1-developer-1051/gzoltars/Lang/1/matrix', 'coverage-only',

            if (args.hybrid_scheme is None) != (args.hybrid_coverage_matrix is None):
                raise RuntimeError('--hybrid-scheme and --hybrid-coverage-matrix should come together or not at all')

            with open(args.element_names) as name_file:
                element_names = {i: name.strip() for i, name in enumerate(name_file)}

            n_elements = len(element_names)

            with open(args.matrix) as matrix_file:
                tally = tally_matrix(matrix_file, args.total_defn, n_elements=n_elements)

            if args.hybrid_scheme is None:
                hybrid_coverage_tally = None
            else:
                with open(args.hybrid_coverage_matrix) as coverage_matrix_file:
                    hybrid_coverage_tally = tally_matrix(coverage_matrix_file, args.total_defn, n_elements)

            suspiciousnesses = suspiciousnesses_from_tallies(
                formula=args.formula, hybrid_scheme=args.hybrid_scheme,
                tally=tally, hybrid_coverage_tally=hybrid_coverage_tally)

            with open(args.output, 'w') as output_file:
                writer = csv.DictWriter(output_file, [args.element_type, 'Suspiciousness'])
                writer.writeheader()
                for element in range(n_elements):
                    writer.writerow({
                        args.element_type: element_names[element],
                        'Suspiciousness': suspiciousnesses[element]})

        except Exception as e:
            print('Something wrong in:', pid, bid,e)
        # python crush_matrix.py --formula opt2 --matrix 'killmap' --element-type Mutant --element-names 'Lang-1-mutants.log' --total-defn tests --output output



def crush_matrix_mcbfl(bid_num):
    # delete_csv(1)
    for bid in range(1, bid_num + 1):
        # for bid in range(1, 2):
        bid = str(bid)
        print('====>>>>', formula, Formulas_used.index(formula)+1, '/', len(Formulas_used), '---------', pid,
              pid_list.index(pid)+1, '/', len(pid_list), '----', bid, '/', bid_num)

        try:
            # data path
            sbfl_path = './==Test reduce==/SBFL/stmt-sus/' + formula + '/' + pid_lower + '/' + pid + '-' + bid + '-stmt-sus'
            mbfl_path = './==Test reduce==/MBFL/stmtsus/' + formula + '/' + pid + '/' + pid + '-' + bid


            # output path
            formula_path = './==Test reduce==/MCBFL/stmtsus/' + formula + '/'
            mkdir(formula_path)
            stmt_sus_path = formula_path + pid_lower + '/'
            mkdir(stmt_sus_path)
            stmt_sus = stmt_sus_path + pid+'-'+bid

            # read write
            df_sbfl = pd.read_csv(sbfl_path, header=None)
            df_mbfl = pd.read_csv(mbfl_path, header=None)
            df_sbfl_list = df_sbfl.values.tolist()
            df_mbfl_list = df_mbfl.values.tolist()

            mcbfl_sus_list = []
            for sbfl_line,mbfl_line in zip(df_sbfl_list,df_mbfl_list):
                if mbfl_line[0] == 'Statement': mcbfl_sus_list.append(mbfl_line)
                else:
                    sbfl_sus = float(sbfl_line[1])
                    mbfl_sus = float(mbfl_line[1])

                    mcbfl_sus = str((sbfl_sus+mbfl_sus)/2)
                    mcbfl_sus_list.append([mbfl_line[0],mcbfl_sus])

            with open(stmt_sus, 'w') as output_file:
                for line in mcbfl_sus_list:
                    output_file.write(",".join(line))
                    output_file.write('\n')

        except Exception as e:
            print('Something wrong in:', pid, bid,e)
        # python crush_matrix.py --formula opt2 --matrix 'killmap' --element-type Mutant --element-names 'Lang-1-mutants.log' --total-defn tests --output output



def crush_matrix_mcbfl_ftmes(bid_num):
    # delete_csv(1)
    for bid in range(1, bid_num + 1):
        # for bid in range(1, 2):
        bid = str(bid)
        print('====>>>>', formula, Formulas_used.index(formula)+1, '/', len(Formulas_used), '---------', pid,
              pid_list.index(pid)+1, '/', len(pid_list), '----', bid, '/', bid_num)

        try:
            # data path
            sbfl_path = './==Test reduce==/SBFL/stmt-sus/' + formula + '/' + pid_lower + '/' + pid + '-' + bid + '-stmt-sus'
            mbfl_path = './==Test reduce==/FTMES/stmtsus/' + formula + '/' + pid + '/' + pid + '-' + bid


            # output path
            formula_path = './==Test reduce==/FTMES-MCBFL/stmtsus/' + formula + '/'
            mkdir(formula_path)
            stmt_sus_path = formula_path + pid_lower + '/'
            mkdir(stmt_sus_path)
            stmt_sus = stmt_sus_path + pid+'-'+bid

            # read write
            df_sbfl = pd.read_csv(sbfl_path, header=None)
            df_mbfl = pd.read_csv(mbfl_path, header=None)
            df_sbfl_list = df_sbfl.values.tolist()
            df_mbfl_list = df_mbfl.values.tolist()

            mcbfl_sus_list = []
            for sbfl_line,mbfl_line in zip(df_sbfl_list,df_mbfl_list):
                if mbfl_line[0] == 'Statement': mcbfl_sus_list.append(mbfl_line)
                else:
                    sbfl_sus = float(sbfl_line[1])
                    mbfl_sus = float(mbfl_line[1])

                    mcbfl_sus = str((sbfl_sus+mbfl_sus)/2)
                    mcbfl_sus_list.append([mbfl_line[0],mcbfl_sus])

            with open(stmt_sus, 'w') as output_file:
                for line in mcbfl_sus_list:
                    output_file.write(",".join(line))
                    output_file.write('\n')

        except Exception as e:
            print('Something wrong in:', pid, bid,e)
        # python crush_matrix.py --formula opt2 --matrix 'killmap' --element-type Mutant --element-names 'Lang-1-mutants.log' --total-defn tests --output output




def crush_matrix_mcbfl_reduce(bid_num):
    # delete_csv(1)
    # for bid in range(1, bid_num + 1):
    #     # for bid in range(1, 2):
    #     bid = str(bid)
    #     print('====>>>>', formula, Formulas_used.index(formula)+1, '/', len(Formulas_used), '---------', pid,
    #           pid_list.index(pid)+1, '/', len(pid_list), '----', bid, '/', bid_num)

        try:
            # data path
            sbfl_path = './==Test reduce==/SBFL/stmt-sus/' + formula + '/' + pid_lower + '/' + pid + '-' + bid + '-stmt-sus'
            mbfl_path = './==Test reduce==/data_reduce/MBFL/'+reduce_technique_times+'/stmtsus/' + str(reduce_rate) + '/' + formula + '/' + pid+'/'+ pid+'-'+bid


            # output path
            formula_path = './==Test reduce==/data_reduce/MCBFL/'+reduce_technique_times+'/stmtsus/' + str(reduce_rate) + '/' + formula + '/'
            mkdir(formula_path)
            stmt_sus_path = formula_path + pid + '/'
            mkdir(stmt_sus_path)
            stmt_sus = stmt_sus_path + pid+'-'+bid

            # read write
            df_sbfl = pd.read_csv(sbfl_path, header=None)
            df_mbfl = pd.read_csv(mbfl_path, header=None)
            df_sbfl_list = df_sbfl.values.tolist()
            df_mbfl_list = df_mbfl.values.tolist()

            mcbfl_sus_list = []
            for sbfl_line,mbfl_line in zip(df_sbfl_list,df_mbfl_list):
                if mbfl_line[0] == 'Statement': mcbfl_sus_list.append(mbfl_line)
                else:
                    sbfl_sus = float(sbfl_line[1])
                    mbfl_sus = float(mbfl_line[1])

                    mcbfl_sus = str((sbfl_sus+mbfl_sus)/2)
                    mcbfl_sus_list.append([mbfl_line[0],mcbfl_sus])

            with open(stmt_sus, 'w') as output_file:
                for line in mcbfl_sus_list:
                    output_file.write(",".join(line))
                    output_file.write('\n')

        except Exception as e:
            a=1
            # print('Something wrong in:', pid, bid,e)
        # python crush_matrix.py --formula opt2 --matrix 'killmap' --element-type Mutant --element-names 'Lang-1-mutants.log' --total-defn tests --output output




# hybird_kill_coverage=====================================
import collections
import re
import gzip

PASS = 'PASS'

Outcome = collections.namedtuple(
  'Outcome',
  ('test_case', 'mutant_id', 'timeout',
   'category', 'runtime', 'output_hash', 'covered_mutants', 'stack_trace'))

def parse_outcome_line(line):
  result = Outcome(*line.split(',', 7))
  return result._replace(
    mutant_id=int(result.mutant_id),
    timeout=int(result.timeout),
    runtime=int(result.runtime),
    covered_mutants=(set(int(n) for n in result.covered_mutants.split(' ')) if result.covered_mutants else set()))

def find_covered_mutants(original_outcome, mutated_outcomes, error_partition_scheme):
  return set(
    outcome.mutant_id for outcome in mutated_outcomes
    if not are_outcomes_equivalent(outcome, original_outcome, error_partition_scheme))

def format_coverage_matrix_row(covered_mutants, n_mutants, originally_passing):
  words = ['1' if i in covered_mutants else '0' for i in range(1, n_mutants+1)]
  words.append('+' if originally_passing else '-')
  return ' '.join(words)



def outcome_matrix_to_coverage_matrix(bid_num):
    for bid in range(1, bid_num + 1):
        # for bid in range(1, 5):
        bid = str(bid)
        print('====>>>>', formula, Formulas_used.index(formula)+1, '/', len(Formulas_used), '---------', pid,
              pid_list.index(pid)+1, '/', len(pid_list), '----', bid, '/', bid_num)

        try:
            data_path = './data/' + pid_lower + '/' + bid + '/'
            stmt_cov_path = './killmap/' + pid_lower + '/'
            stmt_sus_path = './mutantsus/' + pid_lower + '/'
            output_path = './kill_coverage/' + pid + '/'
            if not os.path.exists(output_path): os.mkdir(output_path)
            # =====================

            outcomes = data_path + pid_lower + '-' + (bid) + '-matrix'

            mutants = data_path + pid + '-' + (bid) + '-mutants.log'
            # mutant_susps = stmt_sus_path + pid + '-' + bid + '-mutantsus'
            # print(mutant_susps)
            # source_code_lines_path = './source-code-lines/' + pid + '-' + bid + 'b.source-code.lines'
            # loaded_classes_path = './loaded_classes/' + pid_lower + '/' + bid + '.src'
            # output = result_path + pid + '-' + bid
            output = output_path + pid + '-' + bid + '-killcoverage'

            # =================================================

            import argparse
            import itertools

            parser = argparse.ArgumentParser()
            parser.add_argument('--outcomes', default=outcomes, required=False,
                                help='path to the outcome matrix produced by Killmap')
            parser.add_argument('--mutants', default=mutants, required=False, help='path to a Major mutants.log file')
            parser.add_argument('--output', default=output, required=False,
                                help='file to write output coverage matrix to')

            args = parser.parse_args()

            with open(args.mutants) as mutants_file:
                n_mutants = count_mutants(mutants_file)

            with (gzip.open(args.outcomes, 'rt') if args.outcomes.endswith('.gz') else open(
                    args.outcomes)) as outcome_matrix_file, open(args.output, 'w') as output_file:
                outcomes = (parse_outcome_line(line) for line in outcome_matrix_file)
                original_outcomes = (o for o in outcomes if o.mutant_id == 0)
                for outcome in original_outcomes:
                    output_file.write(
                        format_coverage_matrix_row(outcome.covered_mutants, n_mutants, outcome.category == PASS))
                    output_file.write('\n')

        except:
            print('Something wrong in:', pid, bid)



# ================3.aggregate mutant susps by stmt==================================

import collections
# from __future__ import division
import re
import csv

AGGREGATORS = {
  'avg': (lambda xs: sum(xs)/len(xs)),
  'max': max
}

def strip_dollar(classname):
  return re.sub(r'\$.*', '', classname)

def read_mutant_susps(mutant_suspiciousness_file):
  return {int(row['Mutant'].split(':')[0]): float(row['Suspiciousness'])
          for row in csv.DictReader(mutant_suspiciousness_file)}


# Major's mutants.log files look like:
#   1:ROR:==(java.lang.Object,java.lang.Object):FALSE(java.lang.Object,java.lang.Object):org.apache.commons.lang3.StringUtils@isEmpty:217:cs == null |==> false
# i.e.
#   (id):...:...:...:(class)@(method):(line):...
mutants_log_parser_pattern = re.compile(r'''
  (?P<id>\d+):            # id :
  [^:]*:[^:]*:[^:]*:      # type of mutation : before : after :
  (?P<classname>[\w.$]+)  # class
  (@[^:]*)?:              # @ method (sometimes absent) :
  (?P<line_number>\d+)    # line
  ''', flags=re.X)
def read_mutant_lines(mutants_log_file):
  '''Reads a mutants.log file, returns a mapping from mutant id to containing statement.

  Example output key/value pair: {1: "mypackage.MyClass#202"}
  '''
  matches = (mutants_log_parser_pattern.match(line) for line in mutants_log_file)
  return {int(m.group('id')): '{}#{}'.format(m.group('classname'), m.group('line_number'))
          for m in matches if m is not None}

def _line_to_classname(s):
  s = s.replace('/', '.')
  s = s.replace('.java', '')
  s = re.sub(r'\$[^#]*', '', s)
  return s
def mutant_lines_to_mutant_stmts(mutant_lines, source_code_lines):
  '''Turns a {mutant: line} dict into a {mutant: line} dict where all lines are statement-roots. Each mutant is mapped to the line on which the containing statement begins.
  '''
  stmt_roots = {}
  for line in source_code_lines:
    root_path, spanned_path = line.strip().split(':')
    root, spanned = _line_to_classname(root_path), _line_to_classname(spanned_path)
    stmt_roots[spanned] = root
  return {
    mutant: stmt_roots[mutant_line] if mutant_line in stmt_roots else mutant_line
    for mutant, mutant_line in mutant_lines.items()}

def invert_dict(d):
  result = collections.defaultdict(set)
  for key, value in d.items():
    result[value].add(key)
  return result

def aggregate_stmt_susps(mutant_stmts, mutant_susps, aggregator):
  mutants_by_stmt = invert_dict(mutant_stmts)
  return {
    stmt: aggregator(list(sorted(mutant_susps[mutant] for mutant in mutants)))
    for stmt, mutants in mutants_by_stmt.items()}

def get_irrelevant_stmts(stmts, loaded_classes):
  loaded_classes = set(strip_dollar(classname) for classname in loaded_classes)
  return set(stmt for stmt in stmts if strip_dollar(stmt[:stmt.index('#')]) not in loaded_classes)


def aggregate_mutant_susps_by_stmt(bid_num):
    for bid in range(1, bid_num + 1):
      bid = str(bid)
      print('====>>>>', formula, Formulas_used.index(formula)+1, '/', len(Formulas_used), '---------', pid,
          pid_list.index(pid)+1, '/', len(pid_list), '----', bid, '/', bid_num)

      try:
        data_path = './data/' + pid_lower + '/' + bid + '/'
        stmt_cov_path = './killmap/' + pid_lower + '/'
        stmt_sus_path = './mutantsus_kill_coverage/' + pid_lower + '/'
        result_path = './==Test reduce==/SBFL/to-1-stmt/'+pid+'/'
        if not os.path.exists(result_path): os.mkdir(result_path)
        # =====================
        accumulator = 'max'
        mutants = data_path + pid + '-' + (bid) + '-mutants.log'
        mutant_susps = stmt_sus_path + pid + '-' + bid + '-mutantsus-killcoverage'
        source_code_lines_path = './source-code-lines/'+pid+'-'+bid+'b.source-code.lines'
        loaded_classes_path = './loaded_classes/'+pid_lower+'/'+bid+'.src'
        output = result_path + pid+'-'+bid


        # =====================

        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--accumulator', required=False,default=accumulator, choices=set(AGGREGATORS.keys()))
        parser.add_argument('--mutants', required=False,default=mutants, help='path to the Major mutants.log file')
        parser.add_argument('--mutant-susps', required=False,default=mutant_susps, help='path to mutant-suspiciousness vector')
        parser.add_argument('--source-code-lines', required=False,default=source_code_lines_path, help='path to statement-span file')
        parser.add_argument('--loaded-classes', required=False,default=loaded_classes_path, help='path to file listing classes loaded by failing tests')
        parser.add_argument('--output', required=False,default=output, help='path to file to write output vector')

        args = parser.parse_args()
        aggregator = AGGREGATORS[args.accumulator]

        with open(args.mutants) as mutants_file:
          mutant_lines = read_mutant_lines(mutants_file)
        with open(args.source_code_lines) as source_code_lines:
          mutant_stmts = mutant_lines_to_mutant_stmts(mutant_lines, source_code_lines)
        with open(args.loaded_classes) as loaded_classes_file:
          loaded_classes = [line.strip() for line in loaded_classes_file if line]
        with open(args.mutant_susps) as mutant_susps_file:
          mutant_susps = read_mutant_susps(mutant_susps_file)

        stmt_susps = aggregate_stmt_susps(mutant_stmts, mutant_susps, aggregator)


        mutants_by_stmt = invert_dict(mutant_stmts)

        all_stmts = []
        all_mutants = []

        for stmt, mutants in mutants_by_stmt.items():
            all_stmts.append(stmt)
            all_mutants.append(list(mutants))

        stmt_cov_path = './kill_coverage/' + pid_lower + '/'
        matrix = stmt_cov_path + pid + '-' + (bid) + '-killcoverage'

        df = pd.read_csv(matrix,header=None)
        df_list = df[0].values.tolist()
        df_T_list = list(map(list, zip(*df_list)))

        stmt_mutant_selected = []
        mutant_only = []
        for m in all_mutants:
            stmt_mutant_selected.append(df_T_list[m[0]*2-2])
            mutant_only.append(df_T_list[m[0]*2-2])
            stmt_mutant_selected.append(df_T_list[1])
        stmt_mutant_selected.append(df_T_list[-1])
        mutant_only.append(df_T_list[-1])
        df_new_list = list(map(list, zip(*stmt_mutant_selected)))
        output_path = './==Test reduce==/SBFL/to-1-stmt/' + pid_lower + '/'
        if not os.path.exists(stmt_cov_path): os.mkdir(output_path)
        output = output_path + pid + '-' + (bid) + '-coverage'

        with open(output, 'w') as output_file:
            for line in df_new_list:
                output_file.write("".join(line))
                output_file.write('\n')

      except Exception as ex:
        print('Something wrong in:', pid, bid,ex)





def aggregate_output_stmt_file(bid_num):
    for bid in range(1, bid_num + 1):
    # for bid in range(1, 2):
      bid = str(bid)
      print('====>>>>', formula, Formulas_used.index(formula)+1, '/', len(Formulas_used), '---------', pid,
          pid_list.index(pid)+1, '/', len(pid_list), '----', bid, '/', bid_num)

      try:
        data_path = './data/' + pid_lower + '/' + bid + '/'
        stmt_cov_path = './killmap/' + pid_lower + '/'
        stmt_sus_path = './mutantsus_kill_coverage/' + pid_lower + '/'
        result_path = './==Test reduce==/SBFL/to-1-stmt/'+pid+'/'
        if not os.path.exists(result_path): os.mkdir(result_path)
        # =====================
        accumulator = 'max'
        # accumulator = 'avg'
        mutants = data_path + pid + '-' + (bid) + '-mutants.log'
        mutant_susps = stmt_sus_path + pid + '-' + bid + '-mutantsus-killcoverage'
        # print(mutant_susps)
        source_code_lines_path = './source-code-lines/'+pid+'-'+bid+'b.source-code.lines'
        loaded_classes_path = './loaded_classes/'+pid_lower+'/'+bid+'.src'
        output = result_path + pid+'-'+bid


        # =====================

        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--accumulator', required=False,default=accumulator, choices=set(AGGREGATORS.keys()))
        parser.add_argument('--mutants', required=False,default=mutants, help='path to the Major mutants.log file')
        parser.add_argument('--mutant-susps', required=False,default=mutant_susps, help='path to mutant-suspiciousness vector')
        parser.add_argument('--source-code-lines', required=False,default=source_code_lines_path, help='path to statement-span file')
        parser.add_argument('--loaded-classes', required=False,default=loaded_classes_path, help='path to file listing classes loaded by failing tests')
        parser.add_argument('--output', required=False,default=output, help='path to file to write output vector')

        args = parser.parse_args()
        aggregator = AGGREGATORS[args.accumulator]

        with open(args.mutants) as mutants_file:
          mutant_lines = read_mutant_lines(mutants_file)
        with open(args.source_code_lines) as source_code_lines:
          mutant_stmts = mutant_lines_to_mutant_stmts(mutant_lines, source_code_lines)
        with open(args.loaded_classes) as loaded_classes_file:
          loaded_classes = [line.strip() for line in loaded_classes_file if line]
        with open(args.mutant_susps) as mutant_susps_file:
          mutant_susps = read_mutant_susps(mutant_susps_file)



        '''将变异体的结果归纳为一条语句'''
        mutants_by_stmt = invert_dict(mutant_stmts)

        all_stmts = []
        all_mutants = []

        tmp = 1
        for stmt, mutants in mutants_by_stmt.items():
            # print(stmt, mutants)
            stmt_line = str(tmp)+':'+stmt
            all_stmts.append(stmt_line)
            # print(stmt_line)
            tmp += 1

        output_path = './==Test reduce==/SBFL/stmt-log/' + pid_lower + '/'
        if not os.path.exists(output_path): os.mkdir(output_path)
        output = output_path + pid + '-' + (bid) + '-log'


        # df_new_list.to_csv(output)

        with open(output, 'w') as output_file:
            for line in all_stmts:
                output_file.write("".join(line))
                output_file.write('\n')



      except Exception as ex:
        print('Something wrong in:', pid, bid,ex)
      # python aggregate-mutant-susps-by-stmt.py --accumulator max --mutants 'Lang-1-mutants.log' --mutant-susps 'output' --source-code-lines 'Lang-1b.source-code.lines' --loaded-classes '1.src' --output stmt_sups



# ======================  FTMES  ==========================

def replace_killmap_use_coverage(bid_num):
    for bid in range(1, bid_num + 1):
        bid = str(bid)
        print('====>>>>', formula, Formulas_used.index(formula)+1, '/', len(Formulas_used), '---------', pid,
              pid_list.index(pid)+1, '/', len(pid_list), '----', bid, '/', bid_num)

        try:
            # 读取覆盖矩阵
            mutant_cov_path = './kill_coverage/' + pid_lower + '/'
            matrix_cov = mutant_cov_path + pid + '-' + (bid) + '-killcoverage'

            df = pd.read_csv(matrix_cov, header=None)
            df_cov_list = df[0].values.tolist()
            df_T_cov_list = list(map(list, zip(*df_cov_list)))


            # 读取杀死矩阵
            mutant_killmap_path = './killmap/' + pid_lower + '/'
            matrix_killmap = mutant_killmap_path + pid + '-' + (bid) + '-killmap'

            df = pd.read_csv(matrix_killmap, header=None)
            df_killmap_list = df[0].values.tolist()
            df_T_killmap_list = list(map(list, zip(*df_killmap_list)))


            new_matrix = []
            for c,k in zip(df_cov_list,df_killmap_list):
                if k[-1] == '+': new_matrix.append(c)
                if k[-1] == '-': new_matrix.append(k)


            # ============================   output  ================================
            output_path = './==Test reduce==/FTMES/killcoverage/' + pid_lower + '/'
            if not os.path.exists(output_path): os.mkdir(output_path)
            output = output_path + pid + '-' + (bid) + '-killcoverage'

            with open(output, 'w') as output_file:
                for line in new_matrix:
                    output_file.write("".join(line))
                    output_file.write('\n')

        except Exception as ex:
            print('Something wrong in:', pid, bid, ex)


def crush_matrix_FTMES(bid_num):
    # delete_csv(1)
    for bid in range(1, bid_num + 1):
        # for bid in range(1, 2):
        bid = str(bid)
        print('====>>>>', formula, Formulas_used.index(formula)+1, '/', len(Formulas_used), '---------', pid,
              pid_list.index(pid)+1, '/', len(pid_list), '----', bid, '/', bid_num)

        try:
            formula_path = './==Test reduce==/FTMES/mutantsus/' + formula + '/'
            mkdir(formula_path)

            data_path = './data/' + pid_lower + '/' + bid + '/'
            killmap_path = './==Test reduce==/FTMES/killcoverage/' + pid_lower + '/'
            mutant_sus_path = formula_path + pid_lower + '/'
            if not os.path.exists(mutant_sus_path): os.mkdir(mutant_sus_path)
            # =====================
            # 预定义参数
            # formula = 'gp13'
            # formula = 'ochiai'
            matrix = killmap_path + pid + '-' + (bid) + '-killcoverage'
            element_type = 'Mutant'
            element_names = data_path + pid + '-' + (bid) + '-mutants.log'
            total_defn = 'tests'
            output = mutant_sus_path + pid + '-' + bid + '-mutantsus-killcoverage'
            if not os.path.exists(mutant_sus_path): os.mkdir(mutant_sus_path)

            # =====================

            import argparse
            import csv

            parser = argparse.ArgumentParser()
            parser.add_argument('--formula', required=False, default=formula, choices=set(FORMULAS.keys()))
            parser.add_argument('--matrix', required=False, default=matrix, help='path to the coverage/kill-matrix')
            parser.add_argument('--hybrid-scheme', choices=['numerator', 'constant', 'mirror', 'coverage-only'])
            # parser.add_argument('--hybrid-scheme', choices=['coverage-only'])
            parser.add_argument('--hybrid-coverage-matrix', help='optional coverage matrix for hybrid techniques')
            parser.add_argument('--element-type', required=False, default=element_type, choices=['Statement', 'Mutant'],
                                help='file enumerating names for matrix columns')
            # parser.add_argument('--element-type', required=True, choices=['Statement'], help='file enumerating names for matrix columns')
            parser.add_argument('--element-names', required=False, default=element_names,
                                help='file enumerating names for matrix columns')
            parser.add_argument('--total-defn', required=False, default=total_defn, choices=['tests', 'elements'],
                                help='whether totalpassed/totalfailed should counts tests or covered/killed elements')
            parser.add_argument('--output', required=False, default=output,
                                help='file to write suspiciousness vector to')

            args = parser.parse_args()

            # 'tarantula', '/Users/wanghaifeng/PycharmProjects/Study/Defects4J/root-Lang-1-developer-1051/gzoltars/Lang/1/matrix', 'coverage-only',

            if (args.hybrid_scheme is None) != (args.hybrid_coverage_matrix is None):
                raise RuntimeError('--hybrid-scheme and --hybrid-coverage-matrix should come together or not at all')

            with open(args.element_names) as name_file:
                element_names = {i: name.strip() for i, name in enumerate(name_file)}

            n_elements = len(element_names)

            with open(args.matrix) as matrix_file:
                tally = tally_matrix(matrix_file, args.total_defn, n_elements=n_elements)

            if args.hybrid_scheme is None:
                hybrid_coverage_tally = None
            else:
                with open(args.hybrid_coverage_matrix) as coverage_matrix_file:
                    hybrid_coverage_tally = tally_matrix(coverage_matrix_file, args.total_defn, n_elements)

            suspiciousnesses = suspiciousnesses_from_tallies(
                formula=args.formula, hybrid_scheme=args.hybrid_scheme,
                tally=tally, hybrid_coverage_tally=hybrid_coverage_tally)

            with open(args.output, 'w') as output_file:
                writer = csv.DictWriter(output_file, [args.element_type, 'Suspiciousness'])
                writer.writeheader()
                for element in range(n_elements):
                    writer.writerow({
                        args.element_type: element_names[element],
                        'Suspiciousness': suspiciousnesses[element]})

        except Exception as e:
            print(e)
            print('Something wrong in:', pid, bid)
        # python crush_matrix.py --formula opt2 --matrix 'killmap' --element-type Mutant --element-names 'Lang-1-mutants.log' --total-defn tests --output output


def aggregate_mutant_susps_by_stmt_ftmes(bid_num):
    for bid in range(1, bid_num + 1):
    # for bid in range(1, 2):
        bid = str(bid)
        print('====>>>>', formula, Formulas_used.index(formula)+1, '/', len(Formulas_used), '---------', pid,
          pid_list.index(pid)+1, '/', len(pid_list), '----', bid, '/', bid_num)

        try:
            formula_path = './==Test reduce==/FTMES/stmtsus/' + formula + '/'
            mkdir(formula_path)


            data_path = './data/' + pid_lower + '/' + bid + '/'
            killmap_path = './killmap/' + pid_lower + '/'
            mutant_sus_path = './==Test reduce==/FTMES/mutantsus/' + formula + '/' + pid_lower + '/'
            result_path = formula_path + pid+'/'
            if not os.path.exists(result_path): os.mkdir(result_path)
            # =====================
            accumulator = 'max'
            # accumulator = 'avg'
            mutants = data_path + pid + '-' + (bid) + '-mutants.log'
            mutant_susps = mutant_sus_path + pid + '-' + bid + '-mutantsus-killcoverage'
            # print(mutant_susps)
            source_code_lines_path = './source-code-lines/'+pid+'-'+bid+'b.source-code.lines'
            loaded_classes_path = './loaded_classes/'+pid_lower+'/'+bid+'.src'
            output = result_path + pid+'-'+bid

            # =====================

            import argparse

            parser = argparse.ArgumentParser()
            parser.add_argument('--accumulator', required=False,default=accumulator, choices=set(AGGREGATORS.keys()))
            parser.add_argument('--mutants', required=False,default=mutants, help='path to the Major mutants.log file')
            parser.add_argument('--mutant-susps', required=False,default=mutant_susps, help='path to mutant-suspiciousness vector')
            parser.add_argument('--source-code-lines', required=False,default=source_code_lines_path, help='path to statement-span file')
            parser.add_argument('--loaded-classes', required=False,default=loaded_classes_path, help='path to file listing classes loaded by failing tests')
            parser.add_argument('--output', required=False,default=output, help='path to file to write output vector')

            args = parser.parse_args()
            aggregator = AGGREGATORS[args.accumulator]

            with open(args.mutants) as mutants_file:
              mutant_lines = read_mutant_lines(mutants_file)
            with open(args.source_code_lines) as source_code_lines:
              mutant_stmts = mutant_lines_to_mutant_stmts(mutant_lines, source_code_lines)
            with open(args.loaded_classes) as loaded_classes_file:
              loaded_classes = [line.strip() for line in loaded_classes_file if line]
            with open(args.mutant_susps) as mutant_susps_file:
              mutant_susps = read_mutant_susps(mutant_susps_file)

            stmt_susps = aggregate_stmt_susps(mutant_stmts, mutant_susps, aggregator)
            irrelevant_stmts = get_irrelevant_stmts(stmt_susps.keys(), loaded_classes)
            if irrelevant_stmts:
              print('irrelevant statements: {}'.format(irrelevant_stmts))
              for irrelevant_stmt in irrelevant_stmts:
                stmt_susps.pop(irrelevant_stmt)

            with open(args.output, 'w') as stmt_susps_file:
              writer = csv.DictWriter(stmt_susps_file, ['Statement', 'Suspiciousness'])
              writer.writeheader()
              for stmt, susp in stmt_susps.items():
                writer.writerow({
                  'Statement': stmt,
                  'Suspiciousness': susp})
        except Exception as e:
            print('Something wrong in:', pid, bid,e)
        # python aggregate-mutant-susps-by-stmt.py --accumulator max --mutants 'Lang-1-mutants.log' --mutant-susps 'output' --source-code-lines 'Lang-1b.source-code.lines' --loaded-classes '1.src' --output stmt_sups

def crush_matrix_mbfl(bid_num):
    for bid in range(1, bid_num + 1):
        # for bid in range(1, 2):
        bid = str(bid)
        print('====>>>>', formula, Formulas_used.index(formula)+1,'/',len(Formulas_used), '---------',pid, pid_list.index(pid)+1,'/',len(pid_list), '----',bid, '/', bid_num)

        try:
            formula_path = './==Test reduce==/MBFL/mutantsus/' + formula + '/'
            mkdir(formula_path)


            data_path = './data/' + pid_lower + '/' + bid + '/'
            killmap_path = './killmap/' + pid_lower + '/'
            mutant_sus_path = formula_path + pid_lower + '/'
            if not os.path.exists(mutant_sus_path): os.mkdir(mutant_sus_path)
            # =====================
            matrix = killmap_path + pid + '-' + (bid) + '-killmap'
            element_type = 'Mutant'
            element_names = data_path + pid + '-' + (bid) + '-mutants.log'
            total_defn = 'tests'
            output = mutant_sus_path + pid + '-' + bid + '-mutantsus'
            # =====================

            import argparse
            import csv

            parser = argparse.ArgumentParser()
            parser.add_argument('--formula', required=False, default=formula, choices=set(FORMULAS.keys()))
            parser.add_argument('--matrix', required=False, default=matrix, help='path to the coverage/kill-matrix')
            parser.add_argument('--hybrid-scheme', choices=['numerator', 'constant', 'mirror', 'coverage-only'])
            # parser.add_argument('--hybrid-scheme', choices=['coverage-only'])
            parser.add_argument('--hybrid-coverage-matrix', help='optional coverage matrix for hybrid techniques')
            parser.add_argument('--element-type', required=False, default=element_type, choices=['Statement', 'Mutant'],
                                help='file enumerating names for matrix columns')
            # parser.add_argument('--element-type', required=True, choices=['Statement'], help='file enumerating names for matrix columns')
            parser.add_argument('--element-names', required=False, default=element_names,
                                help='file enumerating names for matrix columns')
            parser.add_argument('--total-defn', required=False, default=total_defn, choices=['tests', 'elements'],
                                help='whether totalpassed/totalfailed should counts tests or covered/killed elements')
            parser.add_argument('--output', required=False, default=output,
                                help='file to write suspiciousness vector to')

            args = parser.parse_args()

            # 'tarantula', '/Users/wanghaifeng/PycharmProjects/Study/Defects4J/root-Lang-1-developer-1051/gzoltars/Lang/1/matrix', 'coverage-only',

            if (args.hybrid_scheme is None) != (args.hybrid_coverage_matrix is None):
                raise RuntimeError('--hybrid-scheme and --hybrid-coverage-matrix should come together or not at all')

            with open(args.element_names) as name_file:
                element_names = {i: name.strip() for i, name in enumerate(name_file)}

            n_elements = len(element_names)

            with open(args.matrix) as matrix_file:
                tally = tally_matrix(matrix_file, args.total_defn, n_elements=n_elements)

            # print(tally)


            if args.hybrid_scheme is None:
                hybrid_coverage_tally = None
            else:
                with open(args.hybrid_coverage_matrix) as coverage_matrix_file:
                    hybrid_coverage_tally = tally_matrix(coverage_matrix_file, args.total_defn, n_elements)

            # print(hybrid_coverage_tally)
            # aa
            suspiciousnesses = suspiciousnesses_from_tallies(
                formula=args.formula, hybrid_scheme=args.hybrid_scheme,
                tally=tally, hybrid_coverage_tally=hybrid_coverage_tally)

            with open(args.output, 'w') as output_file:
                writer = csv.DictWriter(output_file, [args.element_type, 'Suspiciousness'])
                writer.writeheader()
                for element in range(n_elements):
                    writer.writerow({
                        args.element_type: element_names[element],
                        'Suspiciousness': suspiciousnesses[element]})

        except:
            a = 1
            # print('Something wrong in:', pid, bid)
        # python crush_matrix.py --formula opt2 --matrix 'killmap' --element-type Mutant --element-names 'Lang-1-mutants.log' --total-defn tests --output output




def aggregate_mutant_susps_by_stmt_mbfl(bid_num):
    for bid in range(1, bid_num + 1):
    # for bid in range(1, 2):
      bid = str(bid)
      print('====>>>>', formula, Formulas_used.index(formula)+1, '/', len(Formulas_used), '---------', pid,
          pid_list.index(pid)+1, '/', len(pid_list), '----', bid, '/', bid_num)
      try:

        formula_path = './==Test reduce==/MBFL/stmtsus/' + formula + '/'
        mkdir(formula_path)


        data_path = './data/' + pid_lower + '/' + bid + '/'
        killmap_path = './killmap/' + pid_lower + '/'
        mutant_sus_path = './==Test reduce==/MBFL/mutantsus/' + formula + '/' + pid_lower + '/'
        result_path = formula_path + pid+'/'
        if not os.path.exists(result_path): os.mkdir(result_path)
        # =====================
        accumulator = 'max'
        # accumulator = 'avg'
        mutants = data_path + pid + '-' + (bid) + '-mutants.log'
        mutant_susps = mutant_sus_path + pid + '-' + bid + '-mutantsus'
        # print(mutant_susps)
        source_code_lines_path = './source-code-lines/'+pid+'-'+bid+'b.source-code.lines'
        loaded_classes_path = './loaded_classes/'+pid_lower+'/'+bid+'.src'
        output = result_path + pid+'-'+bid

        # =====================

        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--accumulator', required=False,default=accumulator, choices=set(AGGREGATORS.keys()))
        parser.add_argument('--mutants', required=False,default=mutants, help='path to the Major mutants.log file')
        parser.add_argument('--mutant-susps', required=False,default=mutant_susps, help='path to mutant-suspiciousness vector')
        parser.add_argument('--source-code-lines', required=False,default=source_code_lines_path, help='path to statement-span file')
        parser.add_argument('--loaded-classes', required=False,default=loaded_classes_path, help='path to file listing classes loaded by failing tests')
        parser.add_argument('--output', required=False,default=output, help='path to file to write output vector')

        args = parser.parse_args()
        aggregator = AGGREGATORS[args.accumulator]

        with open(args.mutants) as mutants_file:
          mutant_lines = read_mutant_lines(mutants_file)
        with open(args.source_code_lines) as source_code_lines:
          mutant_stmts = mutant_lines_to_mutant_stmts(mutant_lines, source_code_lines)
        with open(args.loaded_classes) as loaded_classes_file:
          loaded_classes = [line.strip() for line in loaded_classes_file if line]
        with open(args.mutant_susps) as mutant_susps_file:
          mutant_susps = read_mutant_susps(mutant_susps_file)

        stmt_susps = aggregate_stmt_susps(mutant_stmts, mutant_susps, aggregator)
        irrelevant_stmts = get_irrelevant_stmts(stmt_susps.keys(), loaded_classes)
        if irrelevant_stmts:
          print('irrelevant statements: {}'.format(irrelevant_stmts))
          for irrelevant_stmt in irrelevant_stmts:
            stmt_susps.pop(irrelevant_stmt)

        with open(args.output, 'w') as stmt_susps_file:
          writer = csv.DictWriter(stmt_susps_file, ['Statement', 'Suspiciousness'])
          writer.writeheader()
          for stmt, susp in stmt_susps.items():
            writer.writerow({
              'Statement': stmt,
              'Suspiciousness': susp})
      except:
        a = 1
        # print('Something wrong in:', pid, bid)
      # python aggregate-mutant-susps-by-stmt.py --accumulator max --mutants 'Lang-1-mutants.log' --mutant-susps 'output' --source-code-lines 'Lang-1b.source-code.lines' --loaded-classes '1.src' --output stmt_sups



# ======================  MBFL  ==========================

import csv
import re
from pyfl import formats



def outcome_matrix_to_kill_matrix_muse(bid_num):
    for bid in range(1, bid_num + 1):
        # for bid in range(1, 5):
        bid = str(bid)
        print('====>>>>', formula, Formulas_used.index(formula)+1, '/', len(Formulas_used), '---------', pid,
              pid_list.index(pid)+1, '/', len(pid_list), '----', bid, '/', bid_num)

        try:
            data_path = './data/' + pid_lower + '/' + bid + '/'
            muse_pf_path = './==Test reduce==/MUSE/passfailmap/' + pid_lower + '/'
            if not os.path.exists(muse_pf_path): os.mkdir(muse_pf_path)
            # =====================
            # 预定义参数

            error_partition_scheme = 'passfail'
            outcomes = data_path + pid_lower + '-' + (bid) + '-matrix'
            unzip_gz(outcomes + '.gz')
            mutants = data_path + pid + '-' + (bid) + '-mutants.log'
            output = muse_pf_path + pid + '-' + (bid) + '-passfailmap'

            # =====================

            import argparse
            import itertools

            parser = argparse.ArgumentParser()
            parser.add_argument('--error-partition-scheme', choices=ERROR_PARTITION_SCHEMES,
                                default=error_partition_scheme)
            parser.add_argument('--outcomes', required=False, help='path to the outcome matrix produced by Killmap',
                                default=outcomes)
            parser.add_argument('--mutants', required=False, help='path to a Major mutants.log file', default=mutants)
            parser.add_argument('--output', required=False, help='file to write output matrix to', default=output)

            args = parser.parse_args()

            with open(args.mutants) as mutants_file:
                n_mutants = count_mutants(mutants_file)

            with open_killmap(args.outcomes) as outcome_matrix_file, open(args.output, 'w') as output_file:

                # for line in outcome_matrix_file:
                #     print(line)
                #     print('parse outcome line',parse_outcome_line(line))
                all_outcomes = (parse_outcome_line(line) for line in outcome_matrix_file)



                tmp = 1
                for original_outcome, mutated_outcomes in group_outcomes_by_test_case(all_outcomes):
                    killed_mutants,pass_fail_list = find_killed_mutants_muse(original_outcome, mutated_outcomes,
                                                         args.error_partition_scheme)
                    # print(tmp,'==================',killed_mutants)
                    tmp+=1
                    output_file.write(
                        format_kill_matrix_row_muse(killed_mutants,pass_fail_list, n_mutants, original_outcome.category == PASS))
                    output_file.write('\n')
                    # break

        except Exception as e:
            print('Something wrong in:', pid, bid,e)


def crush_matrix_muse_mutant_sus(file_path,totalpassed, totalfailed):
    df = pd.read_csv(file_path, header=None)
    df_list = df[0].values.tolist()
    df_T_list = list(map(list, zip(*df_list)))

    del(df_T_list[-1])

    mutants_sus = []

    for m in df_T_list:
        if m[0] != ' ':
            p2f = 0
            f2p = 0
            for t in m:
                if t == '1': p2f += 1
                if t == '2': f2p += 1
            a = 0.5
            m_sus = f2p/totalfailed - a*(p2f/totalpassed)
            mutants_sus.append(m_sus)
    return mutants_sus



def crush_matrix_muse(bid_num):
    for bid in range(1, bid_num + 1):
        # for bid in range(1, 2):
        bid = str(bid)
        print('====>>>>', formula, Formulas_used.index(formula)+1,'/',len(Formulas_used), '---------',pid, pid_list.index(pid)+1,'/',len(pid_list), '----',bid, '/', bid_num)

        try:
            formula_path = './==Test reduce==/MUSE/mutantsus/'
            mkdir(formula_path)

            data_path = './data/' + pid_lower + '/' + bid + '/'
            killmap_path = './==Test reduce==/MUSE/passfailmap/' + pid_lower + '/'
            mutant_sus_path = formula_path + pid_lower + '/'
            if not os.path.exists(mutant_sus_path): os.mkdir(mutant_sus_path)
            # =====================
            matrix = killmap_path + pid + '-' + (bid) + '-passfailmap'
            element_type = 'Mutant'
            element_names = data_path + pid + '-' + (bid) + '-mutants.log'
            total_defn = 'tests'
            output = mutant_sus_path + pid + '-' + bid + '-mutantsus'
            # =====================




            import argparse
            import csv

            parser = argparse.ArgumentParser()
            parser.add_argument('--formula', required=False, default=formula, choices=set(FORMULAS.keys()))
            parser.add_argument('--matrix', required=False, default=matrix, help='path to the coverage/kill-matrix')
            parser.add_argument('--hybrid-scheme', choices=['numerator', 'constant', 'mirror', 'coverage-only'])
            # parser.add_argument('--hybrid-scheme', choices=['coverage-only'])
            parser.add_argument('--hybrid-coverage-matrix', help='optional coverage matrix for hybrid techniques')
            parser.add_argument('--element-type', required=False, default=element_type, choices=['Statement', 'Mutant'],
                                help='file enumerating names for matrix columns')
            # parser.add_argument('--element-type', required=True, choices=['Statement'], help='file enumerating names for matrix columns')
            parser.add_argument('--element-names', required=False, default=element_names,
                                help='file enumerating names for matrix columns')
            parser.add_argument('--total-defn', required=False, default=total_defn, choices=['tests', 'elements'],
                                help='whether totalpassed/totalfailed should counts tests or covered/killed elements')
            parser.add_argument('--output', required=False, default=output,
                                help='file to write suspiciousness vector to')

            args = parser.parse_args()

            # 'tarantula', '/Users/wanghaifeng/PycharmProjects/Study/Defects4J/root-Lang-1-developer-1051/gzoltars/Lang/1/matrix', 'coverage-only',

            if (args.hybrid_scheme is None) != (args.hybrid_coverage_matrix is None):
                raise RuntimeError('--hybrid-scheme and --hybrid-coverage-matrix should come together or not at all')

            with open(args.element_names) as name_file:
                element_names = {i: name.strip() for i, name in enumerate(name_file)}

            n_elements = len(element_names)

            with open(args.matrix) as matrix_file:
                totalpassed, totalfailed = tally_matrix_muse(matrix_file, args.total_defn, n_elements=n_elements)


            # print(tally)


            if args.hybrid_scheme is None:
                hybrid_coverage_tally = None
            else:
                with open(args.hybrid_coverage_matrix) as coverage_matrix_file:
                    hybrid_coverage_tally = tally_matrix_muse(coverage_matrix_file, args.total_defn, n_elements)

            # print(hybrid_coverage_tally)
            # aa
            # suspiciousnesses = suspiciousnes            ses_from_tallies(
                #     formula=args.formula, hybrid_scheme=args.hybrid_scheme,
                #     tally=tally, hybrid_coverage_tally=hybrid_coverage_tally)

            suspiciousnesses=crush_matrix_muse_mutant_sus(matrix,totalpassed, totalfailed)


            with open(args.output, 'w') as output_file:
                writer = csv.DictWriter(output_file, [args.element_type, 'Suspiciousness'])
                writer.writeheader()
                for element in range(n_elements):
                    writer.writerow({
                        args.element_type: element_names[element],
                        'Suspiciousness': suspiciousnesses[element]})


        except Exception as e:
            print('Something wrong in:', pid, bid,e)




def aggregate_mutant_susps_by_stmt_muse(bid_num):
    for bid in range(1, bid_num + 1):
    # for bid in range(1, 2):
      bid = str(bid)
      print('====>>>>', formula, Formulas_used.index(formula)+1, '/', len(Formulas_used), '---------', pid,
          pid_list.index(pid)+1, '/', len(pid_list), '----', bid, '/', bid_num)
      try:

        formula_path = './==Test reduce==/MUSE/stmtsus/'
        mkdir(formula_path)


        data_path = './data/' + pid_lower + '/' + bid + '/'
        killmap_path = './killmap/' + pid_lower + '/'
        mutant_sus_path = './==Test reduce==/MUSE/mutantsus/' + pid_lower + '/'
        result_path = formula_path + pid+'/'
        if not os.path.exists(result_path): os.mkdir(result_path)
        accumulator = 'avg'
        mutants = data_path + pid + '-' + (bid) + '-mutants.log'
        mutant_susps = mutant_sus_path + pid + '-' + bid + '-mutantsus'
        source_code_lines_path = './source-code-lines/'+pid+'-'+bid+'b.source-code.lines'
        loaded_classes_path = './loaded_classes/'+pid_lower+'/'+bid+'.src'
        output = result_path + pid+'-'+bid

        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--accumulator', required=False,default=accumulator, choices=set(AGGREGATORS.keys()))
        parser.add_argument('--mutants', required=False,default=mutants, help='path to the Major mutants.log file')
        parser.add_argument('--mutant-susps', required=False,default=mutant_susps, help='path to mutant-suspiciousness vector')
        parser.add_argument('--source-code-lines', required=False,default=source_code_lines_path, help='path to statement-span file')
        parser.add_argument('--loaded-classes', required=False,default=loaded_classes_path, help='path to file listing classes loaded by failing tests')
        parser.add_argument('--output', required=False,default=output, help='path to file to write output vector')

        args = parser.parse_args()
        aggregator = AGGREGATORS[args.accumulator]

        with open(args.mutants) as mutants_file:
          mutant_lines = read_mutant_lines(mutants_file)
        with open(args.source_code_lines) as source_code_lines:
          mutant_stmts = mutant_lines_to_mutant_stmts(mutant_lines, source_code_lines)
        with open(args.loaded_classes) as loaded_classes_file:
          loaded_classes = [line.strip() for line in loaded_classes_file if line]
        with open(args.mutant_susps) as mutant_susps_file:
          mutant_susps = read_mutant_susps(mutant_susps_file)

        stmt_susps = aggregate_stmt_susps(mutant_stmts, mutant_susps, aggregator)
        irrelevant_stmts = get_irrelevant_stmts(stmt_susps.keys(), loaded_classes)
        if irrelevant_stmts:
          print('irrelevant statements: {}'.format(irrelevant_stmts))
          for irrelevant_stmt in irrelevant_stmts:
            stmt_susps.pop(irrelevant_stmt)

        with open(args.output, 'w') as stmt_susps_file:
          writer = csv.DictWriter(stmt_susps_file, ['Statement', 'Suspiciousness'])
          writer.writeheader()
          for stmt, susp in stmt_susps.items():
            writer.writerow({
              'Statement': stmt,
              'Suspiciousness': susp})
      except:
        print('Something wrong in:', pid, bid)



def matrix_muse(bid_num):
    for bid in range(1, bid_num + 1):
        bid = str(bid)
        print('====>>>>', formula, Formulas_used.index(formula)+1, '/', len(Formulas_used), '---------', pid,
          pid_list.index(pid)+1, '/', len(pid_list), '----', bid, '/', bid_num)


        data_path = './data/' + pid_lower + '/' + bid + '/'
        mutants_log_path = data_path + pid + '-' + (bid) + '-mutants.log'

        killmap_path = './killmap/' + pid_lower + '/'
        killmap_path = killmap_path + pid + '-' + (bid) + '-killmap'


        # with open(mutants_log_path) as f:
        #     ids_of_mutants_in_buggy_lines = set(m.id for m in formats.iter_mutants_log_lines(f) if
        #                                         formats.Line(formats.java_classname_to_path(m.classname),
        #                                                      m.lineno) in buggy_lines)
        with gzip.open(killmap_path, mode='rt') as f:
            runs = formats.iter_killmap_test_runs(f)
            print(runs)

            try:
                current_unmutated_run = next(runs)
                if current_unmutated_run.category != 'FAIL':
                    return False
                for run in runs:
                    if run.test == current_unmutated_run.test:
                        # if run.category == 'PASS' and run.mutant_id in ids_of_mutants_in_buggy_lines:
                        if run.category == 'PASS' :
                            print('Pass test',run.test)
                    elif run.category == 'FAIL':
                        current_unmutated_run = run
                        print('Fail test', run.test)
                    else:
                        print('Wrong test', run.test)
            except (StopIteration, csv.Error):
                return False
        try:
            formula_path = './==Test reduce==/MBFL/stmtsus/' + formula + '/'
            mkdir(formula_path)


            data_path = './data/' + pid_lower + '/' + bid + '/'
            killmap_path = './killmap/' + pid_lower + '/'
            mutant_sus_path = './==Test reduce==/MBFL/mutantsus/' + formula + '/' + pid_lower + '/'
            result_path = formula_path + pid+'/'
            if not os.path.exists(result_path): os.mkdir(result_path)
            # =====================
            # 预定义参数
            accumulator = 'max'
            # accumulator = 'avg'
            mutants = data_path + pid + '-' + (bid) + '-mutants.log'
            mutant_susps = mutant_sus_path + pid + '-' + bid + '-mutantsus'
            # print(mutant_susps)
            source_code_lines_path = './source-code-lines/'+pid+'-'+bid+'b.source-code.lines'
            loaded_classes_path = './loaded_classes/'+pid_lower+'/'+bid+'.src'
            output = result_path + pid+'-'+bid

            # =====================

            import argparse

            parser = argparse.ArgumentParser()
            parser.add_argument('--accumulator', required=False,default=accumulator, choices=set(AGGREGATORS.keys()))
            parser.add_argument('--mutants', required=False,default=mutants, help='path to the Major mutants.log file')
            parser.add_argument('--mutant-susps', required=False,default=mutant_susps, help='path to mutant-suspiciousness vector')
            parser.add_argument('--source-code-lines', required=False,default=source_code_lines_path, help='path to statement-span file')
            parser.add_argument('--loaded-classes', required=False,default=loaded_classes_path, help='path to file listing classes loaded by failing tests')
            parser.add_argument('--output', required=False,default=output, help='path to file to write output vector')

            args = parser.parse_args()
            aggregator = AGGREGATORS[args.accumulator]

            with open(args.mutants) as mutants_file:
              mutant_lines = read_mutant_lines(mutants_file)
            with open(args.source_code_lines) as source_code_lines:
              mutant_stmts = mutant_lines_to_mutant_stmts(mutant_lines, source_code_lines)
            with open(args.loaded_classes) as loaded_classes_file:
              loaded_classes = [line.strip() for line in loaded_classes_file if line]
            with open(args.mutant_susps) as mutant_susps_file:
              mutant_susps = read_mutant_susps(mutant_susps_file)

            stmt_susps = aggregate_stmt_susps(mutant_stmts, mutant_susps, aggregator)
            irrelevant_stmts = get_irrelevant_stmts(stmt_susps.keys(), loaded_classes)
            if irrelevant_stmts:
              print('irrelevant statements: {}'.format(irrelevant_stmts))
              for irrelevant_stmt in irrelevant_stmts:
                stmt_susps.pop(irrelevant_stmt)

            with open(args.output, 'w') as stmt_susps_file:
              writer = csv.DictWriter(stmt_susps_file, ['Statement', 'Suspiciousness'])
              writer.writeheader()
              for stmt, susp in stmt_susps.items():
                writer.writerow({
                  'Statement': stmt,
                  'Suspiciousness': susp})
        except:
            print('Something wrong in:', pid, bid)

# =============4.score rank====================================
import os

def buggy_lines(buggylines):
    with open(buggylines) as f:
        all_lines = f.readlines()
        # del all_lines[0]
        buggy_lines_dict = []
        all_sus = []
        for line in all_lines:
            line_num = int(line.split('#')[1])
            pkg_name = line.split('.')[0].replace('/', '.')
            # # sus = float(line.split('#')[1].split(',')[1])
            # print(pkg_name)
            # print(line_num)
            buggy_lines_dict.append([line_num,pkg_name])
    buggy_lines_dict.sort(key=lambda x: x[1], reverse=False)
    return buggy_lines_dict

def sort_sus(lines_sus):
    with open(lines_sus) as f:
        all_lines = f.readlines()
        del all_lines[0]
        all_lines_dict = []
        all_sus = []
        for line in all_lines:
            pkg_name = line.split('#')[0]
            line_num = int(line.split('#')[1].split(',')[0])
            sus = float(line.split('#')[1].split(',')[1])
            # print(pkg_name)
            # print(line_num)
            # print(sus)
            all_lines_dict.append([sus, line_num, pkg_name])
            all_sus.append(sus)

        # print(all_lines_dict)
    all_lines_dict.sort(key=lambda x: x[0], reverse=True)

    import collections
    countDict = collections.Counter(all_sus)
    sus_num = []
    for item in countDict.items():
        item_sus = item[0]
        item_num = item[1]
        sus_num.append([item_sus, item_num])
    sus_num.sort(key=lambda x: x[0], reverse=True)

    return all_lines_dict,sus_num

def not_sort_sus(lines_sus):
    with open(lines_sus) as f:
        all_lines = f.readlines()
        del all_lines[0]
        all_lines_dict = []
        all_sus = []
        for line in all_lines:
            pkg_name = line.split('#')[0]
            line_num = int(line.split('#')[1].split(',')[0])
            sus = float(line.split('#')[1].split(',')[1])
            # print(pkg_name)
            # print(line_num)
            # print(sus)
            all_lines_dict.append([sus, line_num, pkg_name])
            all_sus.append(sus)

        # print(all_lines_dict)
    # all_lines_dict.sort(key=lambda x: x[0], reverse=True)

    import collections
    countDict = collections.Counter(all_sus)
    sus_num = []
    for item in countDict.items():
        item_sus = item[0]
        item_num = item[1]
        sus_num.append([item_sus, item_num])
    sus_num.sort(key=lambda x: x[0], reverse=True)

    return all_lines_dict,sus_num

def rank(sus,sus_num):
    rank_temp = 0
    for j in sus_num:
        if sus == j[0]:
            rank_first = rank_temp + 1
            rank_last = rank_temp + j[1]
            rank_mean = rank_temp + j[1] / 2
        rank_temp += j[1]
    return rank_first,rank_last,rank_mean

def topn(rank_list):
    top1 = 0
    top3 = 0
    top5 = 0
    top10 = 0
    for rank in rank_list:
        if rank < 1 or rank == 1: top1 += 1
        if rank < 3 or rank == 3: top3 += 1
        if rank < 5 or rank == 5: top5 += 1
        if rank < 10 or rank == 10: top10 += 1
    return top1,top3,top5,top10


def read_sloc_csv(sloc_csv,pid,bid):
    import pandas as pd
    ds = pd.read_csv(sloc_csv,header=None)
    ds_all = ds.values.tolist()

    for loc in ds_all:
        if pid == loc[0] and bid == loc[1]:
            excu_line_num = int(loc[2])
            all_line_num = int(loc[3])
            break
    return excu_line_num, all_line_num

def write_csv_entropy(row):
    filename_path = './==Test reduce==/SBFL/reduce/IETCR/' + pid + '/'
    mkdir(filename_path)
    filename = filename_path + pid+'-'+bid+ '-entropy.csv'
    if not os.path.exists(filename):
        header = ['Project', 'Version','Formula', 'Time','Test No.','Entropy(remove)']
        with open(filename,'w') as f:
            write = csv.writer(f)
            write.writerow(header)
            write.writerow(row)
    else:
        with open(filename, 'a+') as f:
            write = csv.writer(f)
            write.writerow(row)


def write_csv_exam(row):
    filename = './==Test reduce==/==Result/exam/'+formula+'.csv'
    if not os.path.exists(filename):
        header = ['Project', 'Version', 'Bug No.','pkg name', 'buggy line', 'SBFL','MBFL','MUSE','MCBFL','FTMES']
        with open(filename,'w') as f:
            write = csv.writer(f)
            write.writerow(header)
            write.writerow(row)
    else:
        with open(filename, 'a+') as f:
            write = csv.writer(f)
            write.writerow(row)

def write_csv_exam_reduce(row):
    filename_path = './==Test reduce==/data_reduce/=Result/'+base_technique+'/exam/'+str(reduce_rate) +'/'
    mkdir(filename_path)
    filename = filename_path +str(reduce_rate)+'-'+formula+'.csv'
    if not os.path.exists(filename):
        header = ['Project', 'Version', 'Bug No.','pkg name', 'buggy line', 'Random','IETCR','FTMES','CBTCR']
        with open(filename,'w') as f:
            write = csv.writer(f)
            write.writerow(header)
            write.writerow(row)
    else:
        with open(filename, 'a+') as f:
            write = csv.writer(f)
            write.writerow(row)

def write_csv_exam_reduce_formulas(row):
    filename_path = './==Test reduce==/data_reduce/=Result/'+base_technique+'-formula'+'/exam/'+str(reduce_rate) +'/'
    mkdir(filename_path)
    filename = filename_path +str(reduce_rate)+'-'+formula+'.csv'
    if not os.path.exists(filename):
        header = ['Project', 'Version', 'Bug No.','pkg name', 'buggy line', 'tarantula', 'ochiai', 'dstar2', 'gp13', 'opt2']
        with open(filename,'w') as f:
            write = csv.writer(f)
            write.writerow(header)
            write.writerow(row)
    else:
        with open(filename, 'a+') as f:
            write = csv.writer(f)
            write.writerow(row)


def write_csv_topn_map(row):
    filename = './==Test reduce==/==Result/topn-map/'+formula+'-topn-map.csv'
    if not os.path.exists(filename):
        header = ['Project', 'Formula','Technique','Top-1','Top-3','Top-5','Top-10','MAP']
        with open(filename,'w') as f:
            write = csv.writer(f)
            write.writerow(header)
            write.writerow(row)
    else:
        with open(filename, 'a+') as f:
            write = csv.writer(f)
            write.writerow(row)

def write_csv_topn_map_reduce(row):
    filename_path = './==Test reduce==/data_reduce/=Result/'+base_technique+'/topn-map/' + str(reduce_rate) + '/'
    mkdir(filename_path)
    filename = filename_path + str(reduce_rate) + '-' + formula + '-topn-map.csv'
    if not os.path.exists(filename):
        header = ['Project', 'Formula','Technique','Top-1','Top-3','Top-5','Top-10','MAP']
        with open(filename,'w') as f:
            write = csv.writer(f)
            write.writerow(header)
            write.writerow(row)
    else:
        with open(filename, 'a+') as f:
            write = csv.writer(f)
            write.writerow(row)

def write_csv_topn_map_reduce_formulas(row):
    filename_path = './==Test reduce==/data_reduce/=Result/'+base_technique+'-formula'+'/topn-map/' + str(reduce_rate) + '/'
    mkdir(filename_path)
    filename = filename_path + str(reduce_rate) + '-' + formula + '-topn-map.csv'
    if not os.path.exists(filename):
        header = ['Project', 'Formula','Technique','Top-1','Top-3','Top-5','Top-10','MAP']
        with open(filename,'w') as f:
            write = csv.writer(f)
            write.writerow(header)
            write.writerow(row)
    else:
        with open(filename, 'a+') as f:
            write = csv.writer(f)
            write.writerow(row)


def write_csv_test_number(row):
    filename_path = './==Test reduce==/data_reduce/=Result/'+'Test_number' +  '/'
    mkdir(filename_path)
    # filename = filename_path + str(reduce_rate)+ '-CBTCR-test-number.csv'
    filename = filename_path +'RANDOM'+ str(reduce_rate)+ '-test-number.csv'
    filename = filename_path +'ietcr-'+ str(reduce_rate)+ '-test-number.csv'
    if not os.path.exists(filename):
        header = ['Project', 'Version','All','Fails','Passes','Rate(give)','Used Passes','Rate(calc.)','Mutants', 'All MTP', 'Used MTP','All Time(s)', 'Used Time(s)','All Time(min)', 'Used Time(min)','Reduced Time(s)','Reduced Time(min)','Fail Time(s)','Fail Time(min)']
        with open(filename,'w') as f:
            write = csv.writer(f)
            write.writerow(header)
            write.writerow(row)
    else:
        with open(filename, 'a+') as f:
            write = csv.writer(f)
            write.writerow(row)

def write_csv_nofound(pid, bid,bug_num,pkg_name,line_num,excu_line_num,all_line_num):
    if not os._exists(formula + '-mbfl-' + pid + '-killcoverage.csv'):
        with open(formula + '-mbfl-' + pid + '-killcoverage.csv', 'a+') as f:
            row = [pid, bid,bug_num,pkg_name,line_num,excu_line_num,all_line_num,'no found']
            write = csv.writer(f)
            write.writerow(row)

def build_csv():
    filename = './==Test reduce==/==Result/' + formula + '.csv'
    if not os.path.exists(filename):
        header = ['Project', 'Bug', 'pkg name', 'buggy line', 'SBFL','MBFL','MUSE','MCBFL','FTMES']
        with open(filename) as f:
            write = csv.writer(f)
            write.writerow(header)

def delete_csv(ifdelete):
    if ifdelete == 1:
        os.remove(formula + '-mbfl-' + pid + '-killcoverage.csv')

################################ 约减部分内容 #########################
### IETCR
def standardization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def entropy(lines_dict):
    sus_dict = []
    for i in lines_dict:
        if ':' in i[2]: i[2] = i[2].split(':')[1]

        sus_dict.append(i[0])

    lines_dict= (standardization(np.array(sus_dict))).tolist()
    # print(lines_dict)
    # print(len(lines_dict))

    totalSus = 0.0
    for line in lines_dict:
        totalSus += line

    # 计算信息熵总和
    entropy = 0.0
    for line in lines_dict:
        try:
            probability = line / totalSus
            if probability == 0.0: probability = 0.1e-301
            else:pass
            # print(probability)

            entropy += -(probability * math.log2(probability))
        except ZeroDivisionError:
            probability = 0.0
            if probability == 0.0: probability = 0.1e-301
            else:
                pass
            # print(probability)

            entropy += -(probability * math.log2(probability))

    return entropy



def crush_matrix_sbfl_test(index,formula):
        try:
            data_path = './==Test reduce==/SBFL/stmt-log/' + pid_lower + '/'
            stmt_sus_path = './==Test reduce==/SBFL/reduce/IETCR/' + pid+ '/' +bid + '/'+'stmtsus' +'/'+formula+'/'
            mkdir(stmt_sus_path)
            # =====================
            # 预定义参数
            # formula = 'gp13'
            # formula = 'ochiai'
            matrix = './==Test reduce==/SBFL/reduce/IETCR/' + pid+ '/' +bid + '/'+'coverage' +'/'+ pid + '-remove-'+str(index)+ '-coverage'
            element_type = 'Mutant'
            element_names = data_path + pid + '-' + (bid) + '-log'
            total_defn = 'tests'
            output = stmt_sus_path + pid + '-remove-'+str(index) + '-stmt-sus'

            # =====================

            import argparse
            import csv

            parser = argparse.ArgumentParser()
            parser.add_argument('--formula', required=False, default=formula, choices=set(FORMULAS.keys()))
            parser.add_argument('--matrix', required=False, default=matrix, help='path to the coverage/kill-matrix')
            parser.add_argument('--hybrid-scheme', choices=['numerator', 'constant', 'mirror', 'coverage-only'])
            # parser.add_argument('--hybrid-scheme', choices=['coverage-only'])
            parser.add_argument('--hybrid-coverage-matrix', help='optional coverage matrix for hybrid techniques')
            parser.add_argument('--element-type', required=False, default=element_type, choices=['Statement', 'Mutant'],
                                help='file enumerating names for matrix columns')
            # parser.add_argument('--element-type', required=True, choices=['Statement'], help='file enumerating names for matrix columns')
            parser.add_argument('--element-names', required=False, default=element_names,
                                help='file enumerating names for matrix columns')
            parser.add_argument('--total-defn', required=False, default=total_defn, choices=['tests', 'elements'],
                                help='whether totalpassed/totalfailed should counts tests or covered/killed elements')
            parser.add_argument('--output', required=False, default=output,
                                help='file to write suspiciousness vector to')

            args = parser.parse_args()

            # 'tarantula', '/Users/wanghaifeng/PycharmProjects/Study/Defects4J/root-Lang-1-developer-1051/gzoltars/Lang/1/matrix', 'coverage-only',

            if (args.hybrid_scheme is None) != (args.hybrid_coverage_matrix is None):
                raise RuntimeError('--hybrid-scheme and --hybrid-coverage-matrix should come together or not at all')

            with open(args.element_names) as name_file:
                element_names = {i: name.strip() for i, name in enumerate(name_file)}

            n_elements = len(element_names)

            with open(args.matrix) as matrix_file:
                tally = tally_matrix(matrix_file, args.total_defn, n_elements=n_elements)

            if args.hybrid_scheme is None:
                hybrid_coverage_tally = None
            else:
                with open(args.hybrid_coverage_matrix) as coverage_matrix_file:
                    hybrid_coverage_tally = tally_matrix(coverage_matrix_file, args.total_defn, n_elements)

            suspiciousnesses = suspiciousnesses_from_tallies(
                formula=args.formula, hybrid_scheme=args.hybrid_scheme,
                tally=tally, hybrid_coverage_tally=hybrid_coverage_tally)

            with open(args.output, 'w') as output_file:
                writer = csv.DictWriter(output_file, [args.element_type, 'Suspiciousness'])
                writer.writeheader()
                for element in range(n_elements):
                    writer.writerow({
                        args.element_type: element_names[element],
                        'Suspiciousness': suspiciousnesses[element]})

        except Exception as e:
            print('Something wrong in:', pid, bid,e)

def entropy_each_test():

        matrix = './==Test reduce==/SBFL/to-1-stmt/' + pid + '/'+ pid + '-' + (bid) + '-coverage'
        df = pd.read_csv(matrix, header=None)
        df_list = df[0].values.tolist()
        pass_tests = []
        fail_tests = []
        fail_tests2 = []
        for test in df_list:
            if test[-1] == '+': pass_tests.append(test)
            if test[-1] == '-':
                fail_tests.append(test)
                fail_tests2.append(test)

        pass_test_index = [i for i in range(len(pass_tests))]


        for index in range(len(pass_tests)):
            full_pass_test_index = [i for i in range(len(pass_tests))]
            # print(index,pass_test_index)
            pass_test_index.remove(index)

            # print(len(fail_tests2))
            df_new_list = [i for i in fail_tests2]


            # df_new_list = fail_tests
            # print(len(df_new_list))

            for p in pass_test_index:df_new_list.append(pass_tests[p])


            pass_test_index = full_pass_test_index

            # print(len(df_new_list))

            output_path = './==Test reduce==/SBFL/reduce/IETCR/' + pid+ '/' +bid + '/'+'coverage' +'/'
            mkdir(output_path)

            output = output_path + pid + '-remove-'+str(index)+ '-coverage'

            with open(output, 'w') as output_file:
                for line in df_new_list:
                    output_file.write("".join(line))
                    output_file.write('\n')
            for formula in Formulas_used:
                s = time.time()

                crush_matrix_sbfl_test(index,formula)

                stmt_sus_path = './==Test reduce==/SBFL/reduce/IETCR/' + pid + '/' + bid + '/' + 'stmtsus' + '/' + formula + '/'+ pid + '-remove-'+str(index) + '-stmt-sus'
                all_lines_dict, sus_num = sort_sus(stmt_sus_path)
                testEntropy = entropy(all_lines_dict)
                e = time.time()
                row = [pid,bid,formula,str(e-s),str(index),str(testEntropy)]
                write_csv_entropy(row)

            os.remove(output)
            os.remove(stmt_sus_path)


def ietcr(bid_num):
    try:
        stmt_sus_path = './==Test reduce==/SBFL/stmt-sus/' + formula + '/' + pid_lower + '/' + pid + '-' + bid + '-stmt-sus'
        all_lines_dict, sus_num = sort_sus(stmt_sus_path)

        all_entropy = entropy(all_lines_dict)

        try:
            test_en_path = './==Test reduce==/SBFL/reduce/IETCR/' + pid + '/' + pid + '-' + bid + '-entropy.csv'
            df = pd.read_csv(test_en_path, header=None)
            df_list = df.values.tolist()

            test_entropy_no = []
            for test in df_list:
                if formula == test[2]:
                    test_en = float(test[3])
                    test_no = int(test[4])
                    # test_entropy_no.append([test_no,test_en-all_entropy])
                    test_entropy_no.append([test_no,all_entropy-test_en])

            test_entropy_no_sorted =  sorted(test_entropy_no,key=lambda x: x[1], reverse=False)
            only_test_no = [i[0] for i in test_entropy_no_sorted]

            return only_test_no

        except Exception as e:
            print(e)

    except Exception as e:
        print('Something wrong in:', pid, bid,e)

##################################
def cbtcr(bid_num):

    matrix = './==Test reduce==/SBFL/to-1-stmt/' + pid + '/' + pid + '-' + (bid) + '-coverage'
    df = pd.read_csv(matrix, header=None)
    df_list = df[0].values.tolist()
    pass_tests = []
    fail_tests = []
    for test in df_list:
        if test[-1] == '+': pass_tests.append(test)
        if test[-1] == '-': fail_tests.append(test)


    stmt_sus_path = './==Test reduce==/SBFL/stmt-sus/' + contribution_formula + '/' + pid_lower + '/' + pid + '-' + bid + '-stmt-sus'
    # all_lines_dict, sus_num = sort_sus(stmt_sus_path)
    # print((all_lines_dict))
    all_lines_dict, sus_num = not_sort_sus(stmt_sus_path)
    test_id = 0
    test_id_contribution = []
    for test in pass_tests:
        join_test =  ''.join(test.strip('+').split())
        contribution = 0.0
        for cover,sus in zip(join_test,all_lines_dict):
            contribution += float(cover)*float(sus[0])
        # print('Test',test_id,'contribution:',contribution)
        test_id_contribution.append([test_id,contribution])
        test_id+=1
    test_id_contribution.sort(key=lambda x: x[1], reverse=True)
    # print(test_id_contribution)
    test_index = [i[0] for i in test_id_contribution]
    return test_index




##################################

def reduce_test_number(bid_num):
    try:
        stmt_cov_path = './killmap/' + pid_lower + '/'
        killmap = stmt_cov_path + pid + '-' + (bid) + '-killmap'


        # 读取原始文件
        df = pd.read_csv(killmap, header=None)
        df_list = df[0].values.tolist()

        # 划分pass和fail测试
        pass_tests = []
        fail_tests = []
        test_ind = 0
        fail_tests_inds = []
        for test in df_list:
            if test[-1] == '+': pass_tests.append(test)
            if test[-1] == '-':
                fail_tests.append(test)
                fail_tests_inds.append(test_ind)
            test_ind+= 1


        # print(len(pass_tests),len(fail_tests))

        sample_num = round(len(pass_tests) * reduce_rate)

        data_path = './data/' + pid_lower + '/' + bid + '/'
        mutants = data_path + pid + '-' + (bid) + '-mutants.log'
        with open(mutants) as mutants_file:
            n_mutants = count_mutants(mutants_file)

        data_path = './data/' + pid_lower + '/' + bid + '/'
        err_log = data_path + pid + '-' + (bid) + '-err.txt'
        with open(err_log) as f:
            all_lines = f.readlines()

            test_info_list = []
            exe_time_list = []
            for line in all_lines:
                if 'starting test' in line:
                    test_index = int((line.split('test ')[1]).split(']')[0].split('/')[0])-1
                    test_name = (line.split('test ')[1]).split(']')[0].split(': ')[1]
                    test_info_list.append([test_index,test_name])
                # if 'should take at most:' in line:
                #     exe_time = float((line.split('= ')[1]).split('s]')[0])
                #     exe_time_list.append(exe_time)

                if 'actually took' in line:
                    exe_time = float((line.split('s;')[0]).split('took ')[1])
                    exe_time_list.append(exe_time)

            all_test_info = []
            for test_in,test_ti in zip(test_info_list,exe_time_list):
                all_test_info.append([test_in[0],test_in[1],test_ti])
            # for i in all_test_info: print(i)


        # ===================================
        # CBTCR
        # all_test_index = cbtcr(bid_num)
        # if sample_num == len(pass_tests):
        #     test_index = [i for i in range(len(pass_tests))]
        # else:
        #     test_index = [all_test_index[i] for i in range(sample_num)]



        # if sample_num == len(pass_tests): test_index = [i for i in range(len(pass_tests))]
        # else: test_index = sorted(random.sample(range(0, len(pass_tests)-1), sample_num))
        #
        # # ================================
        all_test_index = ietcr(bid_num)
        if sample_num == len(pass_tests): test_index = [i for i in range(len(pass_tests))]
        else:test_index = [all_test_index[i] for i in range(sample_num)]



        all_time = 0.0
        for i in all_test_info: all_time += i[2]

        use_time = 0.0
        for ind in test_index:
            use_time += all_test_info[ind][2]

        fail_times = 0.0
        for ind in fail_tests_inds:
            fail_times += all_test_info[ind][2]

        reduce_time = all_time - use_time

        # ===================================

        row = [pid,bid,str(len(df_list)),str(len(fail_tests)),str(len(pass_tests)),str(reduce_rate),str(sample_num),str(np.around(sample_num/len(df_list),4)),
               str(n_mutants),str(n_mutants*len(df_list)),str(n_mutants*sample_num),str(np.around(all_time,4)),str(np.around(use_time,4)),str(np.around(all_time/60,4)),str(np.around(use_time/60,4)),str(np.around(reduce_time,4)),str(np.around(reduce_time/60,4)),str(np.around(fail_times,4)),str(np.around(fail_times/60,4))]

        write_csv_test_number(row)


        return sample_num
    except Exception as e:
        a = 1
        print('Something wrong in:', pid, bid, e)


def reduce_test_mbfl(bid_num):

        try:
            stmt_cov_path = './killmap/' + pid_lower + '/'
            killmap = stmt_cov_path + pid + '-' + (bid) + '-killmap'

            output_path = './==Test reduce==/data_reduce/MBFL/'+reduce_technique_times+'/killmap/' + str(reduce_rate) + '/'
            mkdir(output_path)
            output_path2 = output_path + pid + '/'
            mkdir(output_path2)

            output = output_path2 + pid + '-' + (bid) + '-killmap'
            df = pd.read_csv(killmap, header=None)
            df_list = df[0].values.tolist()

            pass_tests = []
            fail_tests = []
            for test in df_list:
                if test[-1] == '+': pass_tests.append(test)
                if test[-1] == '-': fail_tests.append(test)


            sample_num = round(len(pass_tests)*reduce_rate)

            # ########### random
            # if sample_num == len(pass_tests): test_index = [i for i in range(len(pass_tests))]
            # else: test_index = sorted(random.sample(range(0, len(pass_tests)-1), sample_num))
            # # print(test_index)
            #
            # ########## IETCR
            all_test_index = ietcr(bid_num)
            if sample_num == len(pass_tests): test_index = [i for i in range(len(pass_tests))]
            else:test_index = [all_test_index[i] for i in range(sample_num)]

            # ########### CBTCR
            # all_test_index = cbtcr(bid_num)
            # if sample_num == len(pass_tests): test_index = [i for i in range(len(pass_tests))]
            # else:test_index = [all_test_index[i] for i in range(sample_num)]

            df_new_list = fail_tests
            for index in test_index: df_new_list.append(pass_tests[index])

            with open(output, 'w') as output_file:
                for line in df_new_list:
                    output_file.write("".join(line))
                    output_file.write('\n')


        except Exception as e:
            print('Something wrong in:', pid, bid, e)


def crush_matrix_mbfl_reduce(bid_num):

        try:
            formula_path = './==Test reduce==/data_reduce/MBFL/'+reduce_technique_times+'/mutantsus/' + str(reduce_rate) + '/'+ formula + '/'
            mkdir(formula_path)


            data_path = './data/' + pid_lower + '/' + bid + '/'
            killmap_path = './==Test reduce==/data_reduce/MBFL/'+reduce_technique_times+'/killmap/' + str(reduce_rate) + '/'+ pid + '/'
            mutant_sus_path = formula_path + pid + '/'
            if not os.path.exists(mutant_sus_path): os.mkdir(mutant_sus_path)
            # =====================
            matrix = killmap_path + pid + '-' + (bid) + '-killmap'
            element_type = 'Mutant'
            element_names = data_path + pid + '-' + (bid) + '-mutants.log'
            total_defn = 'tests'
            output = mutant_sus_path + pid + '-' + bid + '-mutantsus'
            # =====================

            import argparse
            import csv

            parser = argparse.ArgumentParser()
            parser.add_argument('--formula', required=False, default=formula, choices=set(FORMULAS.keys()))
            parser.add_argument('--matrix', required=False, default=matrix, help='path to the coverage/kill-matrix')
            parser.add_argument('--hybrid-scheme', choices=['numerator', 'constant', 'mirror', 'coverage-only'])
            parser.add_argument('--hybrid-coverage-matrix', help='optional coverage matrix for hybrid techniques')
            parser.add_argument('--element-type', required=False, default=element_type, choices=['Statement', 'Mutant'],
                                help='file enumerating names for matrix columns')
            parser.add_argument('--element-names', required=False, default=element_names,
                                help='file enumerating names for matrix columns')
            parser.add_argument('--total-defn', required=False, default=total_defn, choices=['tests', 'elements'],
                                help='whether totalpassed/totalfailed should counts tests or covered/killed elements')
            parser.add_argument('--output', required=False, default=output,
                                help='file to write suspiciousness vector to')

            args = parser.parse_args()

            if (args.hybrid_scheme is None) != (args.hybrid_coverage_matrix is None):
                raise RuntimeError('--hybrid-scheme and --hybrid-coverage-matrix should come together or not at all')

            with open(args.element_names) as name_file:
                element_names = {i: name.strip() for i, name in enumerate(name_file)}

            n_elements = len(element_names)

            with open(args.matrix) as matrix_file:
                tally = tally_matrix(matrix_file, args.total_defn, n_elements=n_elements)


            if args.hybrid_scheme is None:
                hybrid_coverage_tally = None
            else:
                with open(args.hybrid_coverage_matrix) as coverage_matrix_file:
                    hybrid_coverage_tally = tally_matrix(coverage_matrix_file, args.total_defn, n_elements)

            suspiciousnesses = suspiciousnesses_from_tallies(
                formula=args.formula, hybrid_scheme=args.hybrid_scheme,
                tally=tally, hybrid_coverage_tally=hybrid_coverage_tally)

            with open(args.output, 'w') as output_file:
                writer = csv.DictWriter(output_file, [args.element_type, 'Suspiciousness'])
                writer.writeheader()
                for element in range(n_elements):
                    writer.writerow({
                        args.element_type: element_names[element],
                        'Suspiciousness': suspiciousnesses[element]})

        except Exception as e:
            print('Something wrong in:', pid, bid,e)




def aggregate_mutant_susps_by_stmt_mbfl_reduce(bid_num):
      try:
        formula_path = './==Test reduce==/data_reduce/MBFL/'+reduce_technique_times+'/stmtsus/' + str(reduce_rate) + '/' + formula + '/'
        mkdir(formula_path)

        data_path = './data/' + pid_lower + '/' + bid + '/'
        killmap_path = './killmap/' + pid_lower + '/'
        mutant_sus_path = './==Test reduce==/data_reduce/MBFL/'+reduce_technique_times+'/mutantsus/' + str(reduce_rate) +'/'+ formula + '/'+pid+'/'
        result_path = formula_path + pid+'/'
        if not os.path.exists(result_path): os.mkdir(result_path)
        # =====================
        # 预定义参数
        accumulator = 'max'
        # accumulator = 'avg'
        mutants = data_path + pid + '-' + (bid) + '-mutants.log'
        mutant_susps = mutant_sus_path + pid + '-' + bid + '-mutantsus'
        # print(mutant_susps)
        source_code_lines_path = './source-code-lines/'+pid+'-'+bid+'b.source-code.lines'
        loaded_classes_path = './loaded_classes/'+pid_lower+'/'+bid+'.src'
        output = result_path + pid+'-'+bid

        # =====================

        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--accumulator', required=False,default=accumulator, choices=set(AGGREGATORS.keys()))
        parser.add_argument('--mutants', required=False,default=mutants, help='path to the Major mutants.log file')
        parser.add_argument('--mutant-susps', required=False,default=mutant_susps, help='path to mutant-suspiciousness vector')
        parser.add_argument('--source-code-lines', required=False,default=source_code_lines_path, help='path to statement-span file')
        parser.add_argument('--loaded-classes', required=False,default=loaded_classes_path, help='path to file listing classes loaded by failing tests')
        parser.add_argument('--output', required=False,default=output, help='path to file to write output vector')

        args = parser.parse_args()
        aggregator = AGGREGATORS[args.accumulator]

        with open(args.mutants) as mutants_file:
          mutant_lines = read_mutant_lines(mutants_file)
        with open(args.source_code_lines) as source_code_lines:
          mutant_stmts = mutant_lines_to_mutant_stmts(mutant_lines, source_code_lines)
        with open(args.loaded_classes) as loaded_classes_file:
          loaded_classes = [line.strip() for line in loaded_classes_file if line]
        with open(args.mutant_susps) as mutant_susps_file:
          mutant_susps = read_mutant_susps(mutant_susps_file)

        stmt_susps = aggregate_stmt_susps(mutant_stmts, mutant_susps, aggregator)
        irrelevant_stmts = get_irrelevant_stmts(stmt_susps.keys(), loaded_classes)
        if irrelevant_stmts:
          print('irrelevant statements: {}'.format(irrelevant_stmts))
          for irrelevant_stmt in irrelevant_stmts:
            stmt_susps.pop(irrelevant_stmt)

        with open(args.output, 'w') as stmt_susps_file:
          writer = csv.DictWriter(stmt_susps_file, ['Statement', 'Suspiciousness'])
          writer.writeheader()
          for stmt, susp in stmt_susps.items():
            writer.writerow({
              'Statement': stmt,
              'Suspiciousness': susp})
      except Exception as e:
        a = 1
        # print('Something wrong in:', pid, bid,e)
      # python aggregate-mutant-susps-by-stmt.py --accumulator max --mutants 'Lang-1-mutants.log' --mutant-susps 'output' --source-code-lines 'Lang-1b.source-code.lines' --loaded-classes '1.src' --output stmt_sups


def random_avg():
        times_list = [1,2,3,4,5]

        for reduce_rate in reduce_rate_list:
            for program_index in range(0, len(pid_list)):
                pid = pid_list[program_index]
                pid_lower = pid_list_lower[program_index]
                bid_num = bid_list[program_index]
                for bid in range(1, bid_num + 1):
                    bid = str(bid)

                    try:
                        for formula in Formulas_used:
                            sus_list = []
                            for times in times_list:
                                formula_path = './==Test reduce==/data_reduce/MCBFL/'+reduce_technique + str(times)+'/stmtsus/' + str(reduce_rate) + '/' + formula + '/'
                                result_path = formula_path + pid + '/'
                                output = result_path + pid + '-' + bid
                                # print(output)
                                df = pd.read_csv(output, header=None)
                                df_list = df.values.tolist()
                                sus_list.append(df_list)
                            final_list = []
                            for x1,x2,x3,x4,x5 in zip(sus_list[0],sus_list[1],sus_list[2],sus_list[3],sus_list[4]):
                                if x1[0] == 'Statement': final_list.append(x1)
                                else:
                                    sus_avg = str((float(x1[1])+float(x2[1])+float(x3[1])+float(x4[1])+float(x5[1]))/5.0)
                                    final_list.append([x1[0],sus_avg])
                            formula_path = './==Test reduce==/data_reduce/MCBFL/' + reduce_technique + str(
                                0) + '/stmtsus/' + str(reduce_rate) + '/' + formula + '/'
                            result_path = formula_path + pid + '/'
                            mkdir(result_path)
                            output = result_path + pid + '-' + bid
                            with open(output, 'w') as output_file:
                                for line in final_list:
                                    output_file.write(",".join(line))
                                    output_file.write('\n')


                    except Exception as e:
                        a = 1
                        print('Something wrong in:', pid, bid,e)






#####################################################################



import os
def mkdir(path):

    folder = os.path.exists(path)
    if not folder: os.makedirs(path)
    # else: print("No build---- ",path)



def score_rank(technique,bid_num):
    pkg_list = []
    exam_list = []
    rank_list = []
    map_list = []

    for bid in range(1, bid_num + 1):
        # for bid in range(1, 2):
        bid = str(bid)
        print('====>>>>', formula, Formulas_used.index(formula)+1, '/', len(Formulas_used), '---------', pid,
              pid_list.index(pid)+1, '/', len(pid_list), '----', bid, '/', bid_num)

        try:
            buggylines = '../SBFL/buggy-lines/' + pid + '-' + bid + '.buggy.lines'
            buggy_lines_dict = buggy_lines(buggylines)

            sloc_csv = 'sloc.csv'
            excu_line_num, all_line_num = read_sloc_csv(sloc_csv, pid, bid)
            # print(excu_line_num, all_line_num)

            if technique == 'SBFL': stmt_sus_path = './==Test reduce==/SBFL/stmt-sus/' + formula + '/' + pid_lower + '/' + pid + '-' + bid + '-stmt-sus'
            if technique == 'FTMES': stmt_sus_path = './==Test reduce==/FTMES/stmtsus/' + formula + '/' + pid+'/' + pid+'-'+bid
            if technique == 'MBFL': stmt_sus_path = './==Test reduce==/MBFL/stmtsus/' + formula + '/' + pid+'/' + pid+'-'+bid
            if technique == 'MCBFL': stmt_sus_path = './==Test reduce==/MCBFL/stmtsus/' + formula + '/' + pid+'/' + pid+'-'+bid
            if technique == 'MUSE': stmt_sus_path = './==Test reduce==/MUSE/stmtsus/' + pid+'/' + pid+'-'+bid


            all_lines_dict, sus_num = sort_sus(stmt_sus_path)
            # all_lines_dict, sus_num = sort_sus(stmt_sus_path_mbfl)


            bug_num = len(buggy_lines_dict)


            valid_line = 0
            map_first, map_last, map_mean = 0,0,0
            for buggy_line in buggy_lines_dict:
                pkg_name = buggy_line[1]
                line_num = buggy_line[0]

                flag = 0
                # 对所有语句进行搜索
                for elements in all_lines_dict:

                    if technique == 'SBFL':
                        if ':' in elements[2]: elements[2] = elements[2].split(':')[1]

                    if pkg_name == elements[2] and line_num == elements[1]:
                        # buggy_sus = elements[0]

                        rank_first, rank_last, rank_mean = rank(elements[0], sus_num)
                        exam_first, exam_last, exam_mean = rank_first / excu_line_num, rank_last / excu_line_num, rank_mean / excu_line_num
                        stmt_sus_info = [formula,pid, bid, bug_num, pkg_name, line_num, excu_line_num, all_line_num, exam_first, exam_last, exam_mean, rank_first, rank_last, rank_mean,str(len(all_lines_dict))]
                        flag = 1
                        valid_line += 1

                        map_first,map_last,map_mean = 1/rank_first+map_first, 1/rank_last+map_last, 1/rank_mean+map_mean

                        row = [formula,technique,pid, bid, bug_num, pkg_name, line_num, excu_line_num, all_line_num, exam_first,exam_last, exam_mean, rank_first, rank_last, rank_mean,str(len(all_lines_dict))]

                        pkg_list.append([pid, bid, bug_num, pkg_name,line_num])
                        exam_list.append(exam_mean)
                        rank_list.append(rank_mean)

                        # write_csv(row)
                        break

            map_first, map_last, map_mean = map_first/valid_line,map_last/valid_line,map_mean/valid_line
            map_list.append(map_mean)




        except Exception as e:
            print('Something wrong in', pid, bid,'------>>>',e)

    return pkg_list, exam_list, rank_list, map_list



def score_rank_reduce(technique,bid_num):
    pkg_list = []
    exam_list = []
    rank_list = []
    map_list = []

    for bid in range(1, bid_num + 1):
        # for bid in range(1, 2):
        bid = str(bid)
        # print('====>>>>', formula, Formulas_used.index(formula)+1, '/', len(Formulas_used), '---------', pid,
        #       pid_list.index(pid)+1, '/', len(pid_list), '----', bid, '/', bid_num)

        try:
            # 每个版本对应的错误的位置
            buggylines = '../SBFL/buggy-lines/' + pid + '-' + bid + '.buggy.lines'
            # 错误的集合：<行号> <类名>
            buggy_lines_dict = buggy_lines(buggylines)

            # 对应版本的代码行：<实际执行的行数><所有代码行数>
            sloc_csv = 'sloc.csv'
            excu_line_num, all_line_num = read_sloc_csv(sloc_csv, pid, bid)
            # print(excu_line_num, all_line_num)


            # 各种方法语句怀疑度路径

            if technique == 'random': stmt_sus_path = './==Test reduce==/data_reduce/'+base_technique+'/random-2/stmtsus/' + str(reduce_rate) + '/' + formula + '/'+ pid + '/'+ pid + '-' + bid
            if technique == 'ietcr': stmt_sus_path = './==Test reduce==/data_reduce/'+base_technique+'/ietcr-2/stmtsus/' + str(reduce_rate) + '/' + formula + '/'+ pid + '/'+ pid + '-' + bid
            if technique == 'FTMES' and base_technique == 'MBFL': stmt_sus_path = './==Test reduce==/FTMES/stmtsus/' + formula + '/' + pid+'/' + pid+'-'+bid
            if technique == 'FTMES' and base_technique == 'MCBFL': stmt_sus_path = './==Test reduce==/FTMES-MCBFL/stmtsus/' + formula + '/' + pid+'/' + pid+'-'+bid
            if technique == 'cbtcr': stmt_sus_path = './==Test reduce==/data_reduce/'+base_technique+'/cbtcr-1/stmtsus/' + str(reduce_rate) + '/' + formula + '/'+ pid + '/'+ pid + '-' + bid
            if technique == 'cbtcr-tarantula': stmt_sus_path = './==Test reduce==/data_reduce/'+base_technique+'/cbtcr-tarantula-1/stmtsus/' + str(reduce_rate) + '/' + formula + '/'+ pid + '/'+ pid + '-' + bid
            if technique == 'cbtcr-ochiai': stmt_sus_path = './==Test reduce==/data_reduce/'+base_technique+'/cbtcr-ochiai-1/stmtsus/' + str(reduce_rate) + '/' + formula + '/'+ pid + '/'+ pid + '-' + bid
            if technique == 'cbtcr-dstar2': stmt_sus_path = './==Test reduce==/data_reduce/'+base_technique+'/cbtcr-dstar2-1/stmtsus/' + str(reduce_rate) + '/' + formula + '/'+ pid + '/'+ pid + '-' + bid
            if technique == 'cbtcr-gp13': stmt_sus_path = './==Test reduce==/data_reduce/'+base_technique+'/cbtcr-gp13-1/stmtsus/' + str(reduce_rate) + '/' + formula + '/'+ pid + '/'+ pid + '-' + bid
            if technique == 'cbtcr-opt2': stmt_sus_path = './==Test reduce==/data_reduce/'+base_technique+'/cbtcr-opt2-1/stmtsus/' + str(reduce_rate) + '/' + formula + '/'+ pid + '/'+ pid + '-' + bid

            all_lines_dict, sus_num = sort_sus(stmt_sus_path)
            bug_num = len(buggy_lines_dict)


            valid_line = 0
            map_first, map_last, map_mean = 0,0,0
            for buggy_line in buggy_lines_dict:
                pkg_name = buggy_line[1]
                line_num = buggy_line[0]

                flag = 0
                # 对所有语句进行搜索
                for elements in all_lines_dict:

                    if technique == 'SBFL':
                        if ':' in elements[2]: elements[2] = elements[2].split(':')[1]

                    if pkg_name == elements[2] and line_num == elements[1]:
                        # buggy_sus = elements[0]

                        rank_first, rank_last, rank_mean = rank(elements[0], sus_num)
                        exam_first, exam_last, exam_mean = rank_first / excu_line_num, rank_last / excu_line_num, rank_mean / excu_line_num
                        stmt_sus_info = [formula,pid, bid, bug_num, pkg_name, line_num, excu_line_num, all_line_num, exam_first, exam_last, exam_mean, rank_first, rank_last, rank_mean,str(len(all_lines_dict))]
                        flag = 1
                        valid_line += 1

                        map_first,map_last,map_mean = 1/rank_first+map_first, 1/rank_last+map_last, 1/rank_mean+map_mean

                        row = [formula,technique,pid, bid, bug_num, pkg_name, line_num, excu_line_num, all_line_num, exam_first,exam_last, exam_mean, rank_first, rank_last, rank_mean,str(len(all_lines_dict))]

                        pkg_list.append([pid, bid, bug_num, pkg_name,line_num])
                        exam_list.append(exam_mean)
                        rank_list.append(rank_mean)

                        # write_csv(row)

                        break

            map_first, map_last, map_mean = map_first/valid_line,map_last/valid_line,map_mean/valid_line
            map_list.append(map_mean)
        except Exception as e:
            print('Something wrong in', pid, bid,'------>>>',e)

    return pkg_list, exam_list, rank_list, map_list



# 0 Lang 65
# 1 Chart 26
# 2 Time 27
# 3 Math 106
# 4 Closure 133
# 5 Cli 39
# 6 Codec 18
# 7 Compress 47
# 8 Csv 16
# 9 Gson 18
# 10 JacksonCore 26
# 11 JacksonXml 6
# 12 Jsoup 93
# 13 JxPath 22


if __name__ == '__main__':


    Formulas_used = ['tarantula', 'ochiai', 'barinel', 'dstar2', 'gp13', 'opt2']

    pid_list = ['Lang', 'Chart', 'Time', 'Math', 'Closure', 'Cli', 'Codec', 'Compress',
                'Csv', 'Gson', 'JacksonCore', 'JacksonXml', 'Jsoup', 'JxPath']
    pid_list_lower = ['lang', 'chart', 'time', 'math', 'closure', 'cli', 'codec', 'compress',
                      'csv', 'gson', 'jacksonCore', 'jacksonXml', 'jsoup', 'jxPath']

    bid_list = [65, 26, 27, 106, 133, 39, 18, 47, 16, 18, 26, 6, 93, 22]
    Techniques = ['SBFL','MBFL','MUSE','MCBFL','FTMES']

    times_list = [2]
    contribution_formulas = ['tarantula', 'ochiai', 'dstar2', 'gp13', 'opt2']

    reduce_technique = 'ietcr-'
    # reduce_technique = 'cbtcr-'+contribution_formula +'-'
    # reduce_technique = 'random-'

    reduce_rate_list = [0.1,0.2,0.3]

    is_run = 0
    for contribution_formula in contribution_formulas:
        reduce_technique = 'cbtcr-' + contribution_formula + '-'
        reduce_technique = 'ietcr' + '-'

        for times in times_list:
            if is_run != 1: break
            reduce_technique_times = reduce_technique+str(times)
        ##########################################

            for reduce_rate in reduce_rate_list:
                for program_index in range(0, len(pid_list)):
                    pid = pid_list[program_index]
                    pid_lower = pid_list_lower[program_index]
                    bid_num = bid_list[program_index]
                    # for bid in range(1, 1 + 1):
                    for bid in range(1, bid_num + 1):

                        bid = str(bid)
                        print('====>>>>',reduce_technique_times,'约减率:', str(reduce_rate)+'/0.9', '---', pid,
                              pid_list.index(pid) + 1, '/', len(pid_list), '---', bid, '/', bid_num)

                        # reduce_test_mbfl(bid_num)
                        for formula in Formulas_used:
                            ##################################################################
                            # 约减的算法
                            # 目的：对矩阵的测试用例列进行抽样，然后计算变异体怀疑度，语句怀疑度
                            # data_reduce: 约减的中间文件存储
                            # ietcr(bid_num)
                            # cbtcr(bid_num)


                            # '''=========== MBFL ==========='''
                            # outcome_matrix_to_kill_matrix(bid_num)
                            # output: killmap

                            ####### 读killmap，重写killmap
                            # reduce_test_mbfl(bid_num)
                            # crush_matrix_mbfl_reduce(bid_num)
                            # # ####### output: mutantsus
                            # aggregate_mutant_susps_by_stmt_mbfl_reduce(bid_num)
                            ###### output: stmt sus


                            '''=========== MCBFL ==========='''
                            crush_matrix_mcbfl_reduce(bid_num)
        #######################################################################################




                            '''=========== SBFL ==========='''
                            # bid_num = 1
                            # aggregate_mutant_susps_by_stmt(bid_num)
                            # aggregate_output_stmt_file(bid_num)
                            ######## output: result
                            # crush_matrix(bid_num)
                            ####### output: mutantsus

                            '''=========== FTMES ==========='''
                            # replace_killmap_use_coverage(bid_num)
                            # crush_matrix_FTMES(bid_num)
                            # aggregate_mutant_susps_by_stmt_ftmes(bid_num)


                            '''=========== MBFL ==========='''
                            # outcome_matrix_to_kill_matrix(bid_num)
                            # output: killmap

                            # crush_matrix_mbfl(bid_num)
                            # output: mutantsus
                            # aggregate_mutant_susps_by_stmt_mbfl(bid_num)
                            # output: stmt sus


                            '''=========== MUSE ==========='''

                            # outcome_matrix_to_kill_matrix_muse(bid_num)
                            # crush_matrix_muse(bid_num)
                            # aggregate_mutant_susps_by_stmt_muse(bid_num)


                            '''=========== MCBFL ==========='''
                            # crush_matrix_mcbfl(bid_num)
                            # crush_matrix_mcbfl_ftmes(bid_num)


                            """Exam score Top-N MAP"""
                            # exam_all_list = []
                            #
                            # for technique in Techniques:
                            #     pkg_list, exam_list, rank_list, map_list = score_rank(technique,bid_num)
                            #     exam_all_list.append(exam_list)
                            #
                            #     top1, top3, top5, top10 = topn(rank_list)
                            #     map = np.around(np.mean(map_list),4)
                            #     row = [pid,formula,technique,top1, top3, top5, top10,map]
                            #     write_csv_topn_map(row)
                            #
                            # sbfl = exam_all_list[0]
                            # mbfl = exam_all_list[1]
                            # muse = exam_all_list[2]
                            # mcbfl = exam_all_list[3]
                            # ftmes = exam_all_list[4]
                            # for pkg,e1,e2,e3,e4,e5 in zip(pkg_list,sbfl,mbfl,muse,mcbfl,ftmes):
                            #     row = pkg+[e1,e2,e3,e4,e5]
                            #     write_csv_exam(row)



    # ########################################

    base_technique = 'MBFL'
    # base_technique = 'MCBFL'
    is_run = 1
    for reduce_rate in reduce_rate_list:
        if is_run != 1: break

        for program_index in range(0, len(pid_list)):
            pid = pid_list[program_index]
            pid_lower = pid_list_lower[program_index]
            bid_num = bid_list[program_index]
            for formula in Formulas_used:

                exam_all_list = []
                Techniques = ['random','ietcr','FTMES','cbtcr-dstar2']
                for technique in Techniques:
                    pkg_list, exam_list, rank_list, map_list = score_rank_reduce(technique,bid_num)
                    exam_all_list.append(exam_list)
                    top1, top3, top5, top10 = topn(rank_list)
                    map = np.around(np.mean(map_list),4)
                    row = [pid,formula,technique,top1, top3, top5, top10,map]
                    write_csv_topn_map_reduce(row)
                rand = exam_all_list[0]
                ietc = exam_all_list[1]
                ftmes = exam_all_list[2]
                ctc = exam_all_list[3]
                # mcbfl = exam_all_list[3]
                # ftmes = exam_all_list[4]
                for pkg,e1,e2,e3,e4 in zip(pkg_list,rand,ietc,ftmes,ctc):
                    row = pkg+[e1,e2,e3,e4]
                    write_csv_exam_reduce(row)


    base_technique = 'MBFL'
    base_technique = 'MCBFL'
    # aa
    is_run = 0
    for reduce_rate in reduce_rate_list:
        if is_run != 1: break

        for program_index in range(0, len(pid_list)):
            pid = pid_list[program_index]
            pid_lower = pid_list_lower[program_index]
            bid_num = bid_list[program_index]
            for formula in Formulas_used:

                exam_all_list = []
                Techniques = ['cbtcr-tarantula', 'cbtcr-ochiai', 'cbtcr-dstar2', 'cbtcr-gp13','cbtcr-opt2']
                for technique in Techniques:
                    pkg_list, exam_list, rank_list, map_list = score_rank_reduce(technique, bid_num)
                    exam_all_list.append(exam_list)
                    #
                    top1, top3, top5, top10 = topn(rank_list)
                    map = np.around(np.mean(map_list), 4)
                    row = [pid, formula, technique, top1, top3, top5, top10, map]
                    write_csv_topn_map_reduce_formulas(row)
                #
                tan = exam_all_list[0]
                och = exam_all_list[1]
                dsr = exam_all_list[2]
                gp = exam_all_list[3]
                op = exam_all_list[4]
                for pkg, e1, e2, e3, e4,e5 in zip(pkg_list, tan,och,dsr,gp,op):
                    row = pkg + [e1, e2, e3, e4,e5]
                    write_csv_exam_reduce_formulas(row)
