#!/bin/python

import os

for a in [0,1,2]:
    for b in [0,1,2,3,4,5,6,7]:
        if a!=2 or b<=4:
            for c in [0,1,2,3,4,5]:
                for m in ['BIN','POI','PO-ORD-ACL','PO-VS-SL','OH-BIN','OH-POI','OH-ORD-ACL','OH-VS-SL','ACL','ORD-ACL','SL','VS-SL']:
                    os.system("python ./train-test.py %s %d %d %d"%(m,a,b,c))

