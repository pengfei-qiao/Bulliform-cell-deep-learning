## for LD calculation http://pbgworks.org/sites/pbgworks.org/files/measuresoflinkagedisequilibrium-111119214123-phpapp01_0.pdf

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

genome = pd.read_csv('GBS_for_LD.txt',sep='\t',index_col=0).transpose()

def get_nearby_snp(target,window):
    nearby = []
    for i in genome.columns:
        if genome[i][1] == genome[target][1] and genome[i][2] > (genome[target][2]-window) and genome[i][2] < (genome[target][2]+window):
            nearby.append(i)
    return nearby

bases = ['A','T','C','G']
def get_nearby_LD(target,window):
    LD = {}
    for i in get_nearby_snp(target,window):
        linkage = []
        targetsnp = []
        nearbysnp = []
        for j in xrange(genome.shape[0]-10):
            if genome[target][10+j] in bases and genome[i][10+j] in bases:
                linkage.append(genome[target][10+j]+genome[i][10+j])
                targetsnp.append(genome[target][10+j])
                nearbysnp.append(genome[i][10+j])
        target_snp = {}
        for snp in set(targetsnp):
            target_snp[snp] = targetsnp.count(snp)/float(len(targetsnp))
        nearby_snp = {}
        for snp in set(nearbysnp):
            nearby_snp[snp] = nearbysnp.count(snp)/float(len(nearbysnp))    
        linkage_snp = {}
        for pair in set(linkage):
            linkage_snp[pair] = linkage.count(pair)/float(len(linkage))
        if len(target_snp.keys()) == 2 and len(nearby_snp.keys()) == 2:
            s12 = target_snp.keys()[0]+nearby_snp.keys()[1]
            s21 = target_snp.keys()[1]+nearby_snp.keys()[0]
            s11 = target_snp.keys()[0]+nearby_snp.keys()[0]
            s22 = target_snp.keys()[1]+nearby_snp.keys()[1]
            if s12 in linkage_snp.keys():
                p12 = linkage_snp[s12]
            else:
                p12 = 0
            if s21 in linkage_snp.keys():
                p21 = linkage_snp[s21]
            else:
                p21 = 0
            if s11 in linkage_snp.keys():
                p11 = linkage_snp[s11]
            else:
                p11 = 0
            if s22 in linkage_snp.keys():
                p22 = linkage_snp[s22]
            else:
                p22 = 0
            D = p11*p22-p12*p21
            # print i
            p1 = target_snp[s12[0]]
            p2 = target_snp[s22[0]]
            q1 = nearby_snp[s11[1]]
            q2 = nearby_snp[s22[1]]
            r2 = D**2/(p1*p2*q1*q2)
        LD[genome[i][2]] = r2
    return LD
