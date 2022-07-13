#!/bin/bash

### Author: Yumei Li
### Date: 06/27/2022
### Softwares needed: TrimGalore, BWA, Samtools, Deeptools, DANPOS2, UCSCtools, Bedtools, bwtool

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

#7. Fragmentation features
ls 1.bam/*.merged.sorted.bam|while read file;do
    sample=$(echo $file|sed 's/.merged.sorted.bam//');
    samtools stats -@ 20 $file > ${file}Stats
    samtools stats -@ 20 ${sample}_merged_filtered.sorted.bam > ${sample}_merged_filtered.sorted.bamStats
done
ls 1.bam/*merged_filtered.sorted.bamStats|while read file;do  sample=$(basename $file|cut -f1-2 -d '_'); grep "^IS" $file | cut -f2- |awk '$3>0{print $1+20"\t"$3}' >features/frag_300-500/${sample}.bamStats.IS.tsv ; done;
ls 1.bam/*merged_filtered.sorted.bamStats|while read file;do  sample=$(basename $file|cut -f1-2 -d '_'); grep "^IS" $file | cut -f2- |awk '{print $1+20"\t"$2}' >features/frag_300-500/${sample}.bamStats.IS.all.tsv ; done;
ls features/frag_300-500/*.bamStats.IS.tsv|while read file;do
    sample=$(echo $file|sed 's/.bamStats.IS.tsv//');
    total=$(awk 'BEGIN{sum=0}{sum+=$2}END{print sum}' $file)
    for((i=300;i<500;i+=10))
    do
        range=$(awk -v value=$i '$1>=value && $1<value+20' $file|awk 'BEGIN{sum=0}{sum+=$2}END{print sum}');
        fraction=$(echo "scale=6;$range/$total"|bc);
        echo -e "$i\t$fraction" >>${sample}_Frac300-500.tsv
    done;
done;
cut -f1 frag_300-500/Ctrl_10_Frac300-500.tsv >Ctrl.Frac300-500.fragmentation.tsv
ls frag_300-500/Ctrl_*_Frac300-500.tsv|while read file;do
    cut -f2 $file|paste Ctrl.Frac300-500.fragmentation.tsv - >tmp
    mv tmp Ctrl.Frac300-500.fragmentation.tsv
done;
cat ../Ctrl.header Ctrl.Frac300-500.fragmentation.tsv >tmp;mv tmp Ctrl.Frac300-500.fragmentation.tsv
cut -f1 frag_300-500/HCC_10_Frac300-500.tsv >HCC.Frac300-500.fragmentation.tsv
ls frag_300-500/HCC_*_Frac300-500.tsv|while read file;do
    cut -f2 $file|paste HCC.Frac300-500.fragmentation.tsv - >tmp
    mv tmp HCC.Frac300-500.fragmentation.tsv
done;
cat ../HCC.header HCC.Frac300-500.fragmentation.tsv >tmp;mv tmp HCC.Frac300-500.fragmentation.tsv
cut -f1 frag_300-500/PDAC_10_Frac300-500.tsv >PDAC.Frac300-500.fragmentation.tsv
ls frag_300-500/PDAC_*_Frac300-500.tsv|while read file;do
    cut -f2 $file|paste PDAC.Frac300-500.fragmentation.tsv - >tmp
    mv tmp PDAC.Frac300-500.fragmentation.tsv
done;
cat ../PDAC.header PDAC.Frac300-500.fragmentation.tsv >tmp;mv tmp PDAC.Frac300-500.fragmentation.tsv
	