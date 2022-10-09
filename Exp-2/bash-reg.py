import os

for m in ['AD','SQ']:
    for a in [0,1,2]:
        for b in [0,1,2,3,4,5,6,7]:
            if a!=2 or b<=4:
                for c in [0,1,2,3,4,5]:
                    os.system("python ./train-test-reg.py %s %d %d %d"%(m,a,b,c))