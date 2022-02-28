#Script for dinucleotides breakage computation
#ARG_1 - Directory with BAM-files. All files must be indexed
#ARG_2 - File for output
import numpy as np
from pyfaidx import Fasta
import vcf, os, sys
from os import listdir
from os.path import isfile, join

genome = Fasta('references/human_g1k_v37.fasta')

meth_file = 'references/bcell_cg_minRead5.txt'
reads_file = 'filtered.sam'
wind = 100
max_num_reads = 500000

def iterate_reads(vcf_obj, meth_dict, genome, limit=True):
	file = open(reads_file)
	dic_counts, dic_fracs, out_dic, out_count_dir = {}, {}, {}, {}
	for ind, line in enumerate(file):
		if limit and ind > max_num_reads:
			break
		a = line.strip().split()
		if (len(a) > 3):
			chr, coor, coor1, coor2 = a[2], int(a[3]) - 1, int(a[3]) - wind, int(a[3]) + wind
			dic_fracs, dic_counts = info_from_read(genome, meth_dict, chr, coor, coor1, coor2, dic_fracs, dic_counts)
	for key in dic_counts.keys():
		if (key.count('N') == 0) and (len(key) == 2):
			out_dic[key] = dic_counts[key]/(dic_fracs[key]) # N number of reads is included in dic_fracs
			out_count_dir[key] = dic_counts[key]
	return (out_dic, out_count_dir)


def info_from_read(genome, meth_dict, chr, coor, coor1, coor2, dic_fracs, dic_counts):
	seq = genome[chr][coor1-1:coor2].seq
	di = genome[chr][coor-1:coor+1].seq
	if di == 'CG':
		new_nuc = meth_dict[chr].get(coor, 'C')
		di = new_nuc + di[1]
	for i in range(len(seq)):
		if seq[i:i+2] == 'CG':
			new_nuc = meth_dict[chr].get(coor1+i, 'C')
			di_iter = new_nuc + seq[i+1:i+2]
		else:
			di_iter = seq[i:i+2]
		dic_fracs[di_iter] = dic_fracs.get(di_iter, 0) + 1/(len(seq)-1)
	dic_counts[di] = dic_counts.get(di, 0) + 1
	return [dic_fracs, dic_counts]


def make_dict_meth(meth_file):
	file = open(meth_file)
	dict = {}
	for line in file:
		a = line.strip().split()
		chr, coor1, all, unm = a[0].strip('chr'), int(a[1]), int(a[4]), int(a[5])
		dict[chr] = dict.get(chr, {})
		if all > 10 and unm/all >= 0.9:
			dict[chr][coor1] = 'M'
		elif all > 10 and unm/all <= 0.1:
			dict[chr][coor1] = 'U'
		else:
			dict[chr][coor1] = 'C'
	return dict

def vcf_snps_read():
	vcf_reader = vcf.Reader(filename='references/ALL.wgs.phase3_shapeit2_mvncall_integrated_v5b.20130502.sites.vcf.gz')
	return vcf_reader

vcf_snps = vcf_snps_read()
meth_dict = make_dict_meth(meth_file)

ff = open(sys.argv[2], 'w')

for filename in [sys.argv[1]+'/'+f for f in listdir(sys.argv[1]) if (isfile(join(sys.argv[1], f))) and (".bam" == f[-4:])]:
	print(filename)
	print("samtools view -f 35 -F 4 %s {1..22} > tmp.sam" % filename)
	print("shuf -n %i tmp.sam > %s &" % (max_num_reads, reads_file))
	os.system("samtools view -f 35 -F 4 %s {1..22} > tmp.sam" % filename)
	os.system("shuf -n %i tmp.sam > %s" % (max_num_reads, reads_file))
	breakages, counts = iterate_reads(vcf_snps, meth_dict, genome)
	for key in sorted(breakages.keys()):
		ff.write("%s\t" % key)
	ff.write("\n")
	for key in sorted(breakages.keys()):
		ff.write("%s\t" % breakages[key])
	for key in sorted(breakages.keys()):
		ff.write("%s\t" % counts[key])
	ff.write("\n")
