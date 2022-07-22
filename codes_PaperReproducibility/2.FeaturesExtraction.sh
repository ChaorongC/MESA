#!/bin/bash

### Author: Jianfeng Xu and Yumei Li
### Date: 07/21/2022
### Softwares needed: deepTools, bedtools, DANPOS2, UCSC tools, 

# Reference genome combining human genome (hg19), Labmda sequences
refFasta = /shared/reference/Homo_sapiens/UCSC/hg19/Sequence/WholeGenomeFasta/hg19_Lambda.fa
dir = "/data/fastqFiles/" # This is the current working directtory
size_range = '80_200'
name = "sampleID" # This is the name of the current processed sample.
f_bam=${dir}"/bsmap_out/"${name}'_psorted_filtered_'${size_range}'.bam'
f_panel='/data/Probe_panel_hg19.bed' # The bed file for the target regions.

#1. DNA methylation
f_out=${dir}"/bsmap_out/"${name}"_meth_combined_length_"${size_range}".txt"
python /shared/software/bsmap-2.90/bin/methratio.py -d ${refFasta} -x CG -i no-action -u -g -p -o ${f_out} ${f_bam}
## select CpG sites in the target regions
awk '{if(NR >= 2) print $1"\t"$2"\t"($2+1)"\t"$5"\t"$6"\t"$3}' $name'_meth_combined_length_'${size_range}'.txt' > $name'_meth_combined_length_'${size_range}'.bed'
bedtools intersect -wa -wb -a ${f_panel} -b $name'_meth_combined_length_'${size_range}'.bed' > $name'_meth_combined_length_'${size_range}'_selected.bed'

#2. Nucleosome occupancy
~/miniconda3/bin/python3 danpos.py dpos $f_bam --paired 1 -u 0 -c 1000000 -o Rst_danpos2 >${name}.danpos2.log 2>${name}.danpos2.err
wigToBigWig -clip Rst_danpos2/pooled/${name}*wig /dfs5/weil21-lab/yumeil1/data/chr.size/hg19.chrom.sizes ${file}.bw >${name}.wigTobw.log 2>${name}.wigTobw.err

mkdir features-occupancy
### For Cohort 1, the average value for each nucleosome organization target region.
regionFile="targetPanel_Cohort1_nulceosome_TSS_PAS.bed"
bigWigAverageOverBed Rst_danpos2/pooled/${name}.bw $regionFile features-occupancy/${name}.Cohort1.occupancy.tsv
### For Cohort 2, the average value for each nucleosome organization target region.
regionFile="targetPanel_Cohort2_nulceosome_TSS_PAS.bed"
awk -v OFS="\t" '{for(i=1;i<=101;i++){print $1,$2+10*(i-1),$2+10*(i-1)+1000,$4":"i,$5,$6}}' $regionFile >features-occupancy/targetPanel_Cohort2_nulceosome_TSS_PAS.1kbSlidingWindow.bed6
bigWigAverageOverBed Rst_danpos2/pooled/${name}.bw features-occupancy/targetPanel_Cohort2_nulceosome_TSS_PAS.1kbSlidingWindow.bed6 features-occupancy/${name}.Cohort2.1kbSlidingWindow.occupancy.tsv

#3. Nuucleosome fuzziness
mkdir features-fuzziness & cd features-fuzziness
mkdir intersectRst meanByMarker
### For Cohort 1, the average value for each nucleosome organization target region.
regionFile="targetPanel_Cohort1_nulceosome_TSS_PAS.bed"
awk -v OFS="\t" 'NR>1{print $1,$4-1,$4,$5,$6}' ../Rst_danpos2/pooled/${name}*.positions.xls|grep -vE "Lambda_NEB|pUC19"|bedtools intersect -a $regionFile -b stdin -wo >intersectRst/${name}.Cohhort1.intersect.tsv
cut -f4,11 intersectRst/${name}.Cohort1.intersect.tsv | Rscript scripts/columnMeanByFactor.R -c=2 -f=1 -o=meanByMarker/${name}.Cohort1.meanFuzziness.tsv
### For Cohort 2, the average value for each nucleosome organization target region.
awk -v OFS="\t" 'NR>1{print $1,$4-1,$4,$5,$6}' ../Rst_danpos2/pooled/${name}*.positions.xls|grep -vE "Lambda_NEB|pUC19"|bedtools intersect -a ../features-occupancy/targetPanel_Cohort2_nulceosome_TSS_PAS.1kbSlidingWindow.bed6 -b stdin -wo >intersectRst/${name}.Cohort2.intersect.tsv
cut -f4,11 intersectRst/${name}.Cohort2.intersect.tsv | Rscript scripts/columnMeanByFactor.R -c=2 -f=1 -o=meanByMarker/${name}.Cohort2.1kbSlidingWindow.meanFuzziness.tsv

#4. Fragmentation
mkdir features-fragmentation & cd features-fragmentation
multiBamSummary BED-file -b $f_bam -o ${name}.80_150_rawCount.npz --BED $f_panel --smartLabels -p 10 --outRawCounts ${name}.80_150_rawCount.tsv -e --minFragmentLength 65 --maxFragmentLength 135
multiBamSummary BED-file -b $f_bam -o ${name}.150_200_rawCount.npz --BED $f_panel --smartLabels -p 10 --outRawCounts ${name}.150_200_rawCount.tsv -e --minFragmentLength 135 --maxFragmentLength 185
ls *rawCount.tsv|while read file;do
	awk -v ORS="" -v OFS="" '{print $1":"$2"-"$3;for(i=4;i<=NF;i++){print "\t"$i}print "\n"}' $file|sed "s/'//g"|sed 's/#chr:start-end/featureID/'|sort -k1,1r >tmp
	mv tmp $file
done;
echo -e 'short<-read.delim(file="\$name.80_150_rawCount.tsv", header=T, check.names=F)
			long<-read.delim(file="\$name.150_200_rawCount.tsv", header=T, check.names=F)
			ratio=(short[,-1]+1)/(long[,-1]+1)
			write.table(cbind(short[,1], ratio), file="\$name.fragmentation_150_StoL.tsv", row.names=F, col.names=T, quote=F, sep="\\t")
'|Rscript -

