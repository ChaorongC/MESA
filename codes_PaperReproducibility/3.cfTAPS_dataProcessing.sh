#!/bin/bash

### Author: Yumei Li
### Date: 12/03/2023
### Softwares needed: TrimGalore, BWA, MethylDackel, Samtools, Deeptools, DANPOS2, UCSCtools, Bedtools, bwtool

# Reference human genome (hg19)
refFa=/dfs5/weil21-lab/yumeil1/data/fa/hg19/hg19full.fa

#1. Pre-mapping quality control and read mapping
mkdir 1.bam
ls EGAF*/*_1.fastq.gz |while read read1;do
		prefix=$(basename $read1|sed 's/_1.fastq.gz//');
		read2=$(ls EGAF*/${prefix}_2.fastq.gz);
		/dfs5/weil21-lab/yumeil1/tools/TrimGalore/trim_galore --cores 10 --clip_R1 10 --clip_R2 10 --gzip --trim-n --length 35 --paired -o 0.fastq $read1 $read2
		bwa mem -t 30 -I 500,120,1000,20 $refFa 0.fastq/${prefix}_*val_1.fq.gz 0.fastq/${prefix}_*val_2.fq.gz | samtools view -F 268 -@ 30 -O BAM | samtools sort -@ 30 -O BAM >1.bam/${prefix}.sorted.bam 2>1.bam/${prefix}.bwa.err
done

#2. Bam merge and filter
ls 1.bam/*run1.sorted.bam|while read file;do
		prefix=$(basename $file|sed 's/_run1.sorted.bam//');
		samtools merge -@ 40 - $file 1.bam/${prefix}_run2.sorted.bam 1.bam/${prefix}_run3.sorted.bam | samtools sort -@ 40 -o 1.bam/${prefix}.merged.sorted.bam
		samtools index -@ 40 1.bam/${prefix}.merged.sorted.bam
		alignmentSieve -b 1.bam/${prefix}.merged.sorted.bam --ignoreDuplicates --minMappingQuality 20 -p 30 -o 1.bam/${prefix}_merged_filtered.bam --filterMetrics 1.bam/${sample}_filterMetrics.dupMq.txt
		samtools sort -@ 30 -o 1.bam/${sample}_merged_filtered.sorted.bam 1.bam/${sample}_merged_filtered.bam
		###Bam files used for DANPOS2
		alignmentSieve -b 1.bam/${prefix}.merged.sorted.bam --ignoreDuplicates --maxFragmentLength 180 --minFragmentLength 60 --minMappingQuality 20 -p 40 -o 1.bam/${prefix}_filtered_80_200.bam --filterMetrics 1.bam/${prefix}_filterMetrics.txt
		samtools sort -@ 40 -o 1.bam/${prefix}_filtered_80_200.sorted.bam 1.bam/${prefix}_filtered_80_200.bam
done

#3. Run DANPOS2
ls 1.bam/*filtered_80_200.sorted.bam|while read file;do
		sample=$(basename $file|sed 's/.sorted.bam//');
		~/miniconda3/bin/python3 danpos.py dpos $file --paired 1 -u 0 -c 1000000 -o 2.danpos2 >>danpos2.log 2>>danpos2.err
	done;
ls 2.danpos2/pooled/*wig|sed 's/.wig//'|while read file;do
		wigToBigWig -clip ${file}.wig /share/data/chr.size/hg19.chrom.sizes ${file}.bw >>wigTobw.log 2>>wigTobw.err
done;

#4. Run MethylDackel
ls 1.bam/*_merged_filtered.sorted.bam|while read file;do
			sample=$(basename $file|cut -f1,2 -d '_');
			MethylDackel extract -@ 30 -q 10 -p 13 -t 2 --OT 0,131,0,131 --OB 0,131,0,131 --mergeContext -o 3.MethylDackel/${sample} $refFa $file
done
cat /share/data/gap/hg19.centromere.bed4 /share/data/blacklist/hg19-blacklist.v2.bed|cut -f1-3 >toBeFiltered.bed3
ls 3.MethylDackel/*_CpG.bedGraph|while read file;do
		prefix=$(echo $file|sed 's/.bedGraph//');
		sed -n '2,$p' $file|bedtools intersect -wa -v -a - -b toBeFiltered.bed3 >tmp.bg
		ls /share/data/SNP/human_9606_b151_GRCh37p13/bed_chr_[0-9,X,Y,MT]*.bed.gz|grep -v "Multi"|while read snp;do
			zcat $snp|sed -n '2,$p'|bedtools intersect -wa -v -a tmp.bg -b - >tmp2.bg
			mv tmp2.bg tmp.bg
		done;
		mv tmp.bg ${prefix}.filtered.bg
		gzip $file 
		gzip ${prefix}.filtered.bg
done;
ls 3.MethylDackel/*.filtered.bg.gz|while read file;do
		sample=$(echo $file|cut -f1-2 -d '.');
		gunzip -c $file |sort -k1,1 -k2,2n >tmp
		cut -f1-3,5 tmp >${sample}.mCount.bg
		bedGraphToBigWig ${sample}.mCount.bg /dfs5/weil21-lab/yumeil1/data/chr.size/hg19.chrom.sizes ${sample}.mCount.bw
		awk -v OFS="\t" '{print $1,$2,$3,$5+$6}' tmp >${sample}.tCount.bg
		bedGraphToBigWig ${sample}.tCount.bg /dfs5/weil21-lab/yumeil1/data/chr.size/hg19.chrom.sizes ${sample}.tCount.bw
done; 

#5. Methylation features
conditions=("Ctrl" "HCC" "PDAC");
grep -E "promoter:|enhancer:" /dfs5/weil21-lab/yumeil1/data/regulateRegions/homo_sapiens.GRCh37.Regulatory_Build.hg19.bed4 >hg19.Regulatory.bed4
for((i=0;i<=2;i++))
  do
      ls 3.MethylDackel/${conditions[i]}*.mCount.bw | while read file;do
          prefix=$(echo $file|sed 's/.mCount.bw//');
          bwtool summary hg19.Regulatory.bed4 $file -keep-bed -skip-median -with-sum >${prefix}.Reg.mCount.tsv
      done;
      ls 3.MethylDackel/${conditions[i]}*.tCount.bw | while read file;do
           prefix=$(echo $file|sed 's/.tCount.bw//');
          bwtool summary hg19.Regulatory.bed4 $file stdout -keep-bed -skip-median -with-sum  >${prefix}.Reg.tCount.tsv
      done;
done
for((i=0;i<=2;i++))
do
    awk '{split($4,a,":");print a[1]":"$1"_"$2"-"$3}' 3.MethylDackel/${conditions[i]}_1_CpG.Reg.mCount.tsv >features/${conditions[i]}_CpG.Reg.mCount.tsv
    awk '{split($4,a,":");print a[1]":"$1"_"$2"-"$3}' 3.MethylDackel/${conditions[i]}_1_CpG.Reg.tCount.tsv >features/${conditions[i]}_CpG.Reg.tCount.tsv
		ls 3.MethylDackel/${conditions[i]}*.Reg.mCount.tsv | while read file;do
        awk '{print $NF}' $file | paste features/${conditions[i]}_CpG.Reg.mCount.tsv - >tmp; mv tmp features/${conditions[i]}_CpG.Reg.mCount.tsv
        tFile=$(echo $file|sed 's/mCount/tCount/');
        awk '{print $NF}' $tFile | paste features/${conditions[i]}_CpG.Reg.tCount.tsv - >tmp;mv tmp features/${conditions[i]}_CpG.Reg.tCount.tsv
    done
    cat ${conditions[i]}.header features/${conditions[i]}_CpG.Reg.mCount.tsv >tmp; mv tmp features/${conditions[i]}_CpG.Reg.mCount.tsv
    cat ${conditions[i]}.header features/${conditions[i]}_CpG.Reg.tCount.tsv >tmp; mv tmp features/${conditions[i]}_CpG.Reg.tCount.tsv
done
paste Ctrl_CpG.Reg.tCount.tsv <(cut -f2- HCC_CpG.Reg.tCount.tsv) <(cut -f2- PDAC_CpG.Reg.tCount.tsv) |sed -n '2,$p'| awk '{NAs=0;for(i=2;i<=NF;i++){if($i=="NA"){NAs+=1}}print $1"\t"NAs}' >Reg.methy.noCovNum.tsv
awk -v OFS="\t" '$2<74{print $1}' Reg.methy.noCovNum.tsv >Reg.methy.filtered.txt 
echo -e 'reg <- read.delim(file="Reg.methy.filtered.txt", header=F)
            conditions=c("Ctrl","HCC","PDAC")
            for(i in 1:3){
                mCount=read.delim(file=paste0(conditions[i],"_CpG.Reg.mCount.tsv"), header=T, row.names=1, check.names=F)
                mCount <- merge(mCount, reg, by.x=0, by.y=1)
                row.names(mCount)=mCount[,1]
                mCount=mCount[,-1]
                tCount=read.delim(file=paste0(conditions[i],"_CpG.Reg.tCount.tsv"), header=T, row.names=1, check.names=F)
                tCount <- merge(tCount, reg, by.x=0, by.y=1)
                row.names(tCount)=tCount[,1]
                tCount=tCount[,-1]
                ratio=mCount/tCount
                write.table(round(ratio, digits=4), file=paste0(conditions[i],".promoterEnhancer.methyRatio.tsv"), row.names=T, col.names=T, quote=F, sep="\\t")
            }
    '|Rscript -	

#6. Nucleosome occupancy features
awk -v OFS="\t" '$1!~"random|hap|chrUn"{if($7=="+"){print $1,$2,$2+1,"TSS|"$13"|"$2+1"|"$6,"0",$6}else{print $1,$3-1,$3,"TSS|"$13"|"$3"|"$6,"0",$6}}' /share/data/structure/hg19.refGene.2020-3-1.bed12+|sort -u >hg19.refGene.TSS.bed6
sed -n '2,$p' /share/data/PAS/hg19.PAS.txt | sed 's/%//' | awk -v FS="\t" -v OFS="\t" '$14!~"5UTR|Pseudogene|LncRNA" && $16!="NoPAS" && $15>10 && $8!=""{print $2,$3-1,$3,"PAS|"$9"|"$3"|"$4,"0",$4}' >hg19.refGene.PAS.bed6
cat hg19.refGene.TSS.bed6 hg19.refGene.PAS.bed6 >all.hg19.refGene.TSS-PAS.bed6
conditions=("Ctrl" "HCC" "PDAC");
awk -v OFS="\t" '{print $1,$2-500,$3+500,$4,$5,$6}' all.hg19.refGene.TSS-PAS.bed6 >all.hg19.refGene.TSS-PAS.fl500.bed6
for((i=0;i<=2;i++))
do
    	tmpList=$(ls 2.danpos2/pooled/${conditions[i]}_*bw|awk -v ORS="," '{print $0}'|sed 's/,$//');
    	sh /dfs5/weil21-lab/yumeil1/scripts/bwMeanForMultiFiles.sh -r all.hg19.refGene.TSS-PAS.fl500.bed6 -b $tmpList -p bigWigAverageOverBed -o features/${conditions[i]}.allRefGene.TSS-PAS.fl500.occupancy.tsv
    	cut -f4,7- features/${conditions[i]}.allRefGene.TSS-PAS.fl500.occupancy.tsv|sort -k1,1 >tmp; cat ${conditions[i]}.header tmp >features/${conditions[i]}.allRefGene.TSS-PAS.fl500.occupancy.tsv
done
## Filter 
echo -e 'library(dplyr)
	features=c("fl500", "fl1kb")
	Ctrl <- read.delim(file="Ctrl.allRefGene.TSS-PAS.fl500.occupancy.tsv", header=T, row.names=1, check.names=F)
	HCC <- read.delim(file="HCC.allRefGene.TSS-PAS.fl500.occupancy.tsv", header=T, row.names=1, check.names=F)
	PDAC <- read.delim(file="PDAC.allRefGene.TSS-PAS.fl500.occupancy.tsv", header=T, row.names=1, check.names=F)
	df <- cbind(Ctrl, HCC, PDAC)
	df_mean <- df %>% filter_all(all_vars(.>mean(as.vector(as.matrix(df)))))
	write.table(df_mean[,grepl("Ctrl",colnames(df_mean))],file="Ctrl.allRefGene.TSS-PAS.fl500.occupancy.meanF.tsv", quote=F, row.names=T, col.names=T, sep="\t")
	write.table(df_mean[,grepl("HCC",colnames(df_mean))],file="HCC.allRefGene.TSS-PAS.fl500.occupancy.meanF.tsv", quote=F, row.names=T, col.names=T, sep="\t")
	write.table(df_mean[,grepl("PDAC",colnames(df_mean))],file="PDAC.allRefGene.TSS-PAS.fl500.occupancy.meanF.tsv", quote=F, row.names=T, col.names=T, sep="\t")
	'|Rscript -

#7. WPS features
### Select regions to calculate
awk -v OFS="\t" '{print $1,$2-500,$3+500}' all.hg19.refGene.TSS-PAS.fl1kb.bed6|sort -k1,1 -k2,2n |bedtools merge -i stdin >test.merge.bed
cut -f1 test.merge.bed|sort -u|while read chr;do
    awk -v value=$chr '$1==value' test.merge.bed >test.merge.${chr}.bed
done
ls 1.bam/*_filtered_80_200.sorted.bam|while read file;do
    prefix=$(echo $file|sed 's/_filtered_80_200.sorted.bam//')
    samtools sort -@ 40 -n -o test.nsorted.bam $file
    bedtools bamtobed -bedpe -mate1 -i test.nsorted.bam >${prefix}.mate1First.bedpe
    awk -v OFS="\t" '$1!="Lambda_NEB" && $1!="pUC19"{($2<$5)?start=$2:start=$5; ($3>$6)?end=$3:end=$6; print $1,start,end,$7}' ${prefix}.mate1First.bedpe >${prefix}.frag.bed4
    bedtools intersect -wa -a ${prefix}.frag.bed4 -b test.merge.bed >${prefix}.test.bed
    sort -u ${prefix}.test.bed >${prefix}.frag.TSS-PAS.fl1.5kb.bed4
    rm ${prefix}.test.bed
    sort-bed ${prefix}.frag.TSS-PAS.fl1.5kb.bed4 >${prefix}.frag.TSS-PAS.fl1.5kb.sorted.bed4
    cat /dfs5/weil21-lab/yumeil1/data/chr.size/hg19.noRandom.chr.size|while read chr size;do
         bedextract $chr ${prefix}.frag.TSS-PAS.fl1.5kb.sorted.bed4 >${prefix}.${chr}.bed4
        python /dfs5/weil21-lab/yumeil1/projects/ideaTest/cfDNA-PA/scripts/WPS_region.py -b ${prefix}.${chr}.bed4 -r test.merge.${chr}.bed -o ${prefix}.${chr}.WPS.bg
    done
    cat ${prefix}*chr*WPS.bg | grep -v "track" | sort -k1,1 -k2,2n >${prefix}.WPS.bg
    totalFrag=$(wc -l ${prefix}.frag.bed4|awk '{print $1}')
    factor=$(echo "scale=6; 1000000 / $totalFrag"|bc);
    awk -v value=$factor -v OFS="\t" '{print $1,$2,$3,$4*value}' ${prefix}.WPS.bg >${prefix}.WPS.norm.bg
    bedGraphToBigWig ${prefix}.WPS.norm.bg /dfs5/weil21-lab/yumeil1/data/chr.size/hg19.noRandom.chr.size ${prefix}.WPS.norm.bw
done

conditions=("Ctrl" "HCC" "PDAC");
for((i=0;i<=2;i++))
    do
        tmpList=$(ls 1.bam/*/${conditions[i]}*WPS.norm.bw|awk -v ORS="," '{print $0}'|sed 's/,$//');
        sh /dfs5/weil21-lab/yumeil1/scripts/bwMeanForMultiFiles.sh -r all.hg19.refGene.TSS-PAS.fl500.bed6 -b $tmpList -p bigWigAverageOverBed -o features/${conditions[i]}.allRefGene.TSS-PAS.fl500.WPS.tsv
        cut -f4,7- features/${conditions[i]}.allRefGene.TSS-PAS.fl500.WPS.tsv|sort -k1,1 >tmp; cat ${conditions[i]}.header tmp >features/${conditions[i]}.allRefGene.TSS-PAS.fl500.WPS.tsv
    done