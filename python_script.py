#!/usr/bin/python

import sys

number = int(sys.argv[1])

if number <= 10:
    print number
else:
    raise ValueError('Number is higher than 10!')
    sys.exit(1)
