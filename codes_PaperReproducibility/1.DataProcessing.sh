#!/bin/bash

### Author: Jianfeng Xu and Yumei Li
### Date: 06/27/2022
### Softwares needed: TrimGalore, BSMAP, Samtools, Bedtools, deepTools, 

# Reference genome combining human genome (hg19), Labmda, and pUC19 sequences
refFasta = /shared/reference/Homo_sapiens/UCSC/hg19/Sequence/WholeGenomeFasta/hg19_Lambda_pUC19/hg19_Lambda_pUC19.fa

#1. Pre-mapping quality control
dir = "/data/fastqFiles/" #This the directory storing all the fq files.
cd dir
size_range='80_200'
fastqc *fq.gz

mkdir trim_galore_out
sample_list_1=($(ls -1 | grep '_R1_001_1.fq.gz'))
sample_list_2=($(ls -1 | grep '_R2_001_2.fq.gz'))

for((i=0;i<${#sample_list_1[@]};i++))
do
		trim_galore --paired --fastqc -q 20  --clip_R1 5 --clip_R2 10 --three_prime_clip_R1 30 --three_prime_clip_R2 30 -o trim_galore_out/ ${sample_list_1[i]}  ${sample_list_2[i]}
done

mv trim_galore_out/*val_1.fq.gz .
mv trim_galore_out/*val_2.fq.gz .

#2. Read mapping (BSMAP)
mkdir bsmap_out
sample_list_1=($(ls -1 | grep '_R1_001_1_val_1.fq.gz'))
sample_list_2=($(ls -1 | grep '_R2_001_2_val_2.fq.gz'))
for((i=0;i<${#sample_list_1[@]};i++))
do
		name=$(echo ${sample_list_1[i]} | cut -d'.' -f1)
		bsmap -a ${sample_list_1[i]} -b ${sample_list_2[i]} -d $refFasta -R -p 16 -o bsmap_out/${name}'.bam' &> ${name}'_bsmap_log.txt'
done

#3. Bam file filter
cd bsmap_out
f_panel='/data/Probe_panel_V2_hg19_lambda_pUC19.bed' # The bed file for the target regions.
ls *bam | sed 's/.bam//'|while read name;do
		samtools view -F 268 -b ${name}'.bam' > ${name}'_mapped.bam'
		samtools view -b -L ${f_panel} ${name}'_mapped.bam'  > ${name}'_mapped_onTarget.bam'
		samtools sort ${name}'_mapped_onTarget.bam' ${name}'_psorted'
		samtools index ${name}'_psorted.bam'
		### Fragment length distribution
		bamPEFragmentSize  --histogram ${name}'_fragmentSize.png' -T 'Fragment size' --maxFragmentLength 500 -b ${name}'_psorted.bam' --samplesLabel ${name} --outRawFragmentLengths ${name}'_fragment_length.tsv'
		### Select fragments in a specific sizee range and remove duplicates 
		alignmentSieve -b ${name}'_psorted.bam' --samFlagExclude 260 --ignoreDuplicates --maxFragmentLength 185 --minFragmentLength 65 -o ${name}'_psorted_filtered_'${size_range}'.bam' --filterMetrics metrics.txt
done


