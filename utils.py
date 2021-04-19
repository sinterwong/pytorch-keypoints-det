'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import cv2


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def visualize_hand_pose(frame, hand_, x, y):
    thick = 2
    colors = [(0, 215, 255), (255, 115, 55),
                (5, 255, 55), (25, 15, 255), (225, 15, 55)]

    cv2.line(frame, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),
                (int(hand_['1']['x']+x), int(hand_['1']['y']+y)), colors[0], thick)
    cv2.line(frame, (int(hand_['1']['x']+x), int(hand_['1']['y']+y)),
                (int(hand_['2']['x']+x), int(hand_['2']['y']+y)), colors[0], thick)
    cv2.line(frame, (int(hand_['2']['x']+x), int(hand_['2']['y']+y)),
                (int(hand_['3']['x']+x), int(hand_['3']['y']+y)), colors[0], thick)
    cv2.line(frame, (int(hand_['3']['x']+x), int(hand_['3']['y']+y)),
                (int(hand_['4']['x']+x), int(hand_['4']['y']+y)), colors[0], thick)

    cv2.line(frame, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),
                (int(hand_['5']['x']+x), int(hand_['5']['y']+y)), colors[1], thick)
    cv2.line(frame, (int(hand_['5']['x']+x), int(hand_['5']['y']+y)),
                (int(hand_['6']['x']+x), int(hand_['6']['y']+y)), colors[1], thick)
    cv2.line(frame, (int(hand_['6']['x']+x), int(hand_['6']['y']+y)),
                (int(hand_['7']['x']+x), int(hand_['7']['y']+y)), colors[1], thick)
    cv2.line(frame, (int(hand_['7']['x']+x), int(hand_['7']['y']+y)),
                (int(hand_['8']['x']+x), int(hand_['8']['y']+y)), colors[1], thick)

    cv2.line(frame, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),
                (int(hand_['9']['x']+x), int(hand_['9']['y']+y)), colors[2], thick)
    cv2.line(frame, (int(hand_['9']['x']+x), int(hand_['9']['y']+y)),
                (int(hand_['10']['x']+x), int(hand_['10']['y']+y)), colors[2], thick)
    cv2.line(frame, (int(hand_['10']['x']+x), int(hand_['10']['y']+y)),
                (int(hand_['11']['x']+x), int(hand_['11']['y']+y)), colors[2], thick)
    cv2.line(frame, (int(hand_['11']['x']+x), int(hand_['11']['y']+y)),
                (int(hand_['12']['x']+x), int(hand_['12']['y']+y)), colors[2], thick)

    cv2.line(frame, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),
                (int(hand_['13']['x']+x), int(hand_['13']['y']+y)), colors[3], thick)
    cv2.line(frame, (int(hand_['13']['x']+x), int(hand_['13']['y']+y)),
                (int(hand_['14']['x']+x), int(hand_['14']['y']+y)), colors[3], thick)
    cv2.line(frame, (int(hand_['14']['x']+x), int(hand_['14']['y']+y)),
                (int(hand_['15']['x']+x), int(hand_['15']['y']+y)), colors[3], thick)
    cv2.line(frame, (int(hand_['15']['x']+x), int(hand_['15']['y']+y)),
                (int(hand_['16']['x']+x), int(hand_['16']['y']+y)), colors[3], thick)

    cv2.line(frame, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),
                (int(hand_['17']['x']+x), int(hand_['17']['y']+y)), colors[4], thick)
    cv2.line(frame, (int(hand_['17']['x']+x), int(hand_['17']['y']+y)),
                (int(hand_['18']['x']+x), int(hand_['18']['y']+y)), colors[4], thick)
    cv2.line(frame, (int(hand_['18']['x']+x), int(hand_['18']['y']+y)),
                (int(hand_['19']['x']+x), int(hand_['19']['y']+y)), colors[4], thick)
    cv2.line(frame, (int(hand_['19']['x']+x), int(hand_['19']['y']+y)),
                (int(hand_['20']['x']+x), int(hand_['20']['y']+y)), colors[4], thick)

def visualisation(frame, points, pty='hand'):
    pts_hand = {}  # 构建关键点连线可视化结构
    for i, p in enumerate(points):
        cv2.circle(frame, (int(p[0]), int(
            p[1])), 3, (255, 50, 60), -1)
        cv2.circle(frame, (int(p[0]), int(
            p[1])), 1, (255, 150, 180), -1)
        pts_hand[str(i)] = {}
        pts_hand[str(i)] = {
            "x": p[0],
            "y": p[1],
        }
    if pty == "hand":
        visualize_hand_pose(frame, pts_hand, 0, 0)
    return frame