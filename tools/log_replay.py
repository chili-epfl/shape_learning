#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval

import sys
import re
from collections import OrderedDict

#actions = re.compile('201\d-\d\d-\d\d (?P<time>\d\d:\d\d:\d\d).*INFO - (?P<letter>\w):.*')

action = re.compile('.*INFO - (?P<letter>\w):.*(?P<type>demonstration|generated|learning).*arams: (?P<params>\[[\d., -]*\]). Path: (?P<path>\[[\d., -]*\])')


actions_letters = {}

if __name__ == "__main__":

    with open(sys.argv[1], 'r') as log:

        for line in log.readlines():

            found = action.search(line)
            if found:
                letter = found.group('letter')
                type = found.group('type')
                params = literal_eval(found.group('params'))
                path = literal_eval(found.group('path'))
                actions_letters.setdefault(letter, []).append((type, params))


for letter, value in actions_letters.items():
    print("%s: %s" % (letter, str(value)))
