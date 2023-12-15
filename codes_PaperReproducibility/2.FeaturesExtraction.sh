#!/bin/bash

### Author: Jianfeng Xu and Yumei Li
### Time: 12/03/2023
### Softwares needed: deepTools, bedtools, DANPOS2, UCSC tools, 

# Reference genome combining human genome (hg19), Labmda, and pUC19 sequences
refFasta = /shared/reference/Homo_sapiens/UCSC/hg19/Sequence/WholeGenomeFasta/hg19_Lambda_pUC19/hg19_Lambda_pUC19.fa
dir = "/data/fastqFiles/" # This is the current working directtory
size_range = '80_200'
name = "sampleID" # This is the name of the current processed sample.
f_bam=${dir}"/bsmap_out/"${name}'_psorted_filtered_'${size_range}'.bam'
f_panel='/data/Probe_panel_V2_hg19_lambda_pUC19.bed' # The bed file for the target regions.

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
### For Cohort 1, the average value for each nucleosome organization target region (1-kb sliding windows with 10 bp step)
regionFile1="targetPanel_Cohort1_nulceosome_TSS_PAS.bed"
awk -v OFS="\t" '{for(i=1;i<=101;i++){print $1,$2+10*(i-1),$2+10*(i-1)+1000,$4":"i,$5,$6}}' $regionFile1 >features-occupancy/targetPanel_Cohort1_nulceosome_TSS_PAS.1kbSlidingWindow.bed6
bigWigAverageOverBed Rst_danpos2/pooled/${name}.bw features-occupancy/targetPanel_Cohort1_nulceosome_TSS_PAS.1kbSlidingWindow.bed6 features-occupancy/${name}.Cohort1.o1kbSlidingWindow.ccupancy.tsv
### For Cohort 2, the average value for each nucleosome organization target region.
regionFile2="targetPanel_Cohort2_nulceosome_TSS_PAS.bed"
bigWigAverageOverBed Rst_danpos2/pooled/${name}.bw $regionFile2 features-occupancy/${name}.Cohort2.occupancy.tsv
### For Cohort 3, the average value for each nucleosome organization target region.
regionFile3="targetPanel_Cohort3_nulceosome_TSS_PAS.bed"
bigWigAverageOverBed Rst_danpos2/pooled/${name}.bw $regionFile3 features-occupancy/${name}.Cohort3.occupancy.tsv


#3. Nuucleosome fuzziness
mkdir features-fuzziness & cd features-fuzziness
mkdir intersectRst meanByMarker
### For Cohort 1, the average value for each nucleosome organization target region.
awk -v OFS="\t" 'NR>1{print $1,$4-1,$4,$5,$6}' ../Rst_danpos2/pooled/${name}*.positions.xls|grep -vE "Lambda_NEB|pUC19"|bedtools intersect -a features-occupancy/targetPanel_Cohort1_nulceosome_TSS_PAS.1kbSlidingWindow.bed6 -b stdin -wo >intersectRst/${name}.Cohhort1.intersect.tsv
cut -f4,11 intersectRst/${name}.Cohort1.intersect.tsv | Rscript scripts/columnMeanByFactor.R -c=2 -f=1 -o=meanByMarker/${name}.Cohort1.1kbSlidingWindow.meanFuzziness.tsv
### For Cohort 2, the average value for each nucleosome organization target region.
awk -v OFS="\t" 'NR>1{print $1,$4-1,$4,$5,$6}' ../Rst_danpos2/pooled/${name}*.positions.xls|grep -vE "Lambda_NEB|pUC19"|bedtools intersect -a $regionFile2 -b stdin -wo >intersectRst/${name}.Cohort2.intersect.tsv
cut -f4,11 intersectRst/${name}.Cohort2.intersect.tsv | Rscript scripts/columnMeanByFactor.R -c=2 -f=1 -o=meanByMarker/${name}.Cohort2.meanFuzziness.tsv
### For Cohort 3, the average value for each nucleosome organization target region.
awk -v OFS="\t" 'NR>1{print $1,$4-1,$4,$5,$6}' ../Rst_danpos2/pooled/${name}*.positions.xls|grep -vE "Lambda_NEB|pUC19"|bedtools intersect -a $regionFile3 -b stdin -wo >intersectRst/${name}.Cohort3.intersect.tsv
cut -f4,11 intersectRst/${name}.Cohort3.intersect.tsv | Rscript scripts/columnMeanByFactor.R -c=2 -f=1 -o=meanByMarker/${name}.Cohort3.meanFuzziness.tsv

#4. WPS
if [ 1 == 2 ];then 
	module load ucsc-tools/v393
	ls bamFiles*/*/*.frag.bed4|while read bedFile;do
		name=$(echo $bedFile|sed 's/.frag.bed4//');
		python /dfs5/weil21-lab/yumeil1/projects/ideaTest/cfDNA-PA/scripts/WPS.py -b $bedFile -c /dfs5/weil21-lab/yumeil1/data/chr.size/hg19.noRandom.chr.size -o ${name}.WPS.wig
		perl /dfs5/weil21-lab/yumeil1/scripts/wig_to_bedgraph.pl -i ${name}.WPS.wig |sort -k1,1 -k2,2n >${name}.WPS.bg
		totalFrag=$(wc -l $bedFile|awk '{print $1}')
		factor=$(echo "scale=6; 1000000 / $totalFrag"|bc);
		awk -v value=$factor -v OFS="\t" '{print $1,$2,$3,$4*value}' ${name}.WPS.bg >${name}.WPS.norm.bg
		bedGraphToBigWig ${name}.WPS.norm.bg /dfs5/weil21-lab/yumeil1/data/chr.size/hg19.noRandom.chr.size ${name}.WPS.norm.bw
	done;
fi
mkdir features-WPS 
### For Cohort 1, the average value for each nucleosome organization target region.
bigWigAverageOverBed Rst_danpos2/pooled/${name}.bw targetPanel_Cohort1_nulceosome_TSS_PAS.1kbSlidingWindow.bed6 features-occupancy/${name}.Cohort1.1kbSlidingWindow.ccupancy.tsv
### For Cohort 2, the average value for each nucleosome organization target region.
bigWigAverageOverBed ${name}.WPS.norm.bw $regionFile2 features-occupancy/${name}.Cohort2.WPS.tsv
### For Cohort 3, the average value for each nucleosome organization target region.
bigWigAverageOverBed Rst_danpos2/pooled/${name}.WPS.norm.bw $regionFile3 features-occupancy/${name}.Cohort3.WPS.tsv
