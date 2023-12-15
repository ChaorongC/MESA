### Author: Yumei Li
### Time: 12/03/2023

library(ggplot2)
library(factoextra)
library(pheatmap)
library(corrplot)
library(readxl)
library(ROCR)
library(gridExtra)
library(ggpubr)
library(reshape2)

theme_update(axis.text = element_text(color="black", size = 7),axis.title=element_text(size=9),text=element_text(size=7), 
             legend.text=element_text(size=7),plot.title=element_text(hjust=0.5), panel.background = element_blank(), 
             panel.border = element_rect(color = "black", fill = NA), panel.grid.major = element_blank(), panel.grid.minor = element_blank())
setwd("/yourWorkingDirectory") #set as your working directory

########## Fig. 2 ##########
plotFig2 <- function(cancerFile, nonFile, aucName){
  cancer <- t(read.delim(file = cancerFile, header = T)[,-1])
  noncancer <- t(read.delim(file = nonFile, header = T)[,-1])
  data <- rbind(cancer, noncancer)
  variances <- apply(data, 2, FUN = var, na.rm = T)
  if(length(c(which(is.na(variances)),which(variances==0)))>0){
    data <- data[,-c(which(is.na(variances)),which(variances==0))]
  }
  res_pca<-prcomp(t(na.omit(t(data))),center = T, scale=T)
  subdata<-as.data.frame(res_pca$x[,c(1,2)])
  subdata$status=factor(c(rep("Cancer",nrow(cancer)),rep("Non-Cancer",nrow(noncancer))))
  p1 <- ggplot(subdata, aes(x=PC1, y=PC2, group=status))+geom_point(aes(color=status, shape=status), size=1, alpha=0.6)+
    labs(x=paste0("PC1 (",round(summary(res_pca)$importance[2,1]*100, digits = 2),"%)"), y=paste0("PC2 (",round(summary(res_pca)$importance[2,2]*100, digits = 2),"%)"))+
    scale_shape_manual(values=c(15, 16))+scale_color_manual(values=c('darkred','gray')) + scale_size_manual(values=c(2,1))+
    theme(aspect.ratio=1,legend.position = "top", legend.title = element_blank(), legend.key = element_rect(fill = NA))
  
  data <- as.data.frame(cbind(rowMeans(data, na.rm = T),c(rep("Cancer",nrow(cancer)),rep("Non-Cancer",nrow(noncancer)))), stringsAsFactors = F)
  data$V1=as.numeric(data$V1)
  data$V2=as.factor(data$V2)
  pvalue=signif(wilcox.test(V1~V2, data = data)$p.value, digits = 2)
  p2 <- ggplot(data, aes(x=V2, y=V1, fill=V2))+geom_violin(alpha=0.6)+stat_summary(fun = "median", geom = "crossbar", width=0.2, lwd=0.3) +
    annotate("text",x=c(1,2),y=aggregate(V1~V2, data, FUN="median")[,2]+0.002, label=round(aggregate(V1~V2, data, FUN="median")[,2], digits = 4)) + 
    scale_fill_manual(values=c('darkred','gray'))+labs(x="",y="Average methylation level\nof all CpGs")+
    annotate("text", x=1.5, y=max(data$V1) ,label=paste0("p = ",pvalue))+theme(legend.position = "none")
  
  single <- read.csv(file = aucName, header = T, stringsAsFactors = F)
  pred <- prediction(single[,3], single[,2])
  perf <- performance(pred, "tpr", "fpr")
  auc <- round(performance(pred, measure = "auc")@y.values[[1]], digits = 4)
  df <- data.frame(FalsePositive=perf@x.values[[1]], TruePositive=perf@y.values[[1]])
  p3 <- ggplot(df, aes(x=FalsePositive, y=TruePositive)) + geom_line(color="#825CA6", lwd=0.5) + 
    labs(x="FPR", y="TPR")+annotate("text", x=0.7, y=0.1, label=paste0("AUC = ",auc))
  return(list(p2,p1,p3))
}
plots_1 <- plotFig2("colonCancer/Cohort1_colon.Cancer.siteMethyRatio.final.tsv",
                    "colonCancer/Cohort1_colon.Non-Cancer.siteMethyRatio.final.tsv", 
                    "Manuscript/Cohort1_probability_20220507.csv")
plots_2 <- plotFig2("colonCancer/Cohort2_colon.Cancer.siteMethyRatio.tsv",
                    "colonCancer/Cohort2_colon.Non-Cancer.siteMethyRatio.tsv", 
                    "Manuscript/Cohort2_probability_20220504.csv")
pdf(file="Manuscript/figures/Fig. 2.pdf", width = 6.5, height = 4.5)
ggarrange(plots_1[[1]], plots_1[[2]], plots_1[[3]], plots_2[[1]], plots_2[[2]], plots_2[[3]], nrow = 2, ncol = 3, 
          labels = c("A", "B", "C", "D", "E", "F"), font.label = list(size = 10, color = "black", face = "bold"))
dev.off()

########## Fig. 3 ##########
### 3A
fragData <-  read.delim(file="colonCancer/nucleosome_evaluation/all_frag_Non-Cancer.merged.tsv",header = F)
fragData$V1 = fragData$V1+15
pdf(file="Manuscript/figures/Fig. 3A.pdf", width = 5, height = 5)
plot(x=fragData$V1, fragData$V2/sum(fragData$V2)*100, type="l", xlab = "Fragment length (bp)", ylab = "% of fragments", col = "blue")
abline(v=fragData[which.max(fragData$V2),1], lty="dashed")
dev.off()
### 3B
AT<-read.delim(file="colonCancer/nucleosome_evaluation/Non-Cancer_AT_freq.tsv",header=T)
AT<-AT[order(AT[,1]),]
AT_mean<-rowMeans(AT[,-1])
GC<-read.delim(file="colonCancer/nucleosome_evaluation/Non-Cancer_AT_freq.tsv",header=T)
GC<-GC[order(GC[,1]),]
GC_mean<-rowMeans(GC[,-1])
mean<-as.data.frame(cbind(AT[,1],AT_mean,GC_mean))
colnames(mean)<-c("pos","AA/AT/TA/TT","GG/GC/CG/CC")
mean<-melt(mean,id.vars="pos")
pdf(file = "Manuscript/figures/Fig. 3B.pdf")
ggplot(mean,aes(x=pos,y=value))+geom_line(aes(col=variable))+labs(x="Position relative to fragment center",
    y="Dinucleotide frequency",col="")+theme(panel.grid.minor.x=element_line(size=0.5,
    color="white"))+scale_x_continuous(minor_breaks = seq(-1000, 1000, 10),breaks=c(-73,0,73),labels=c("-73","0","73"))
dev.off()
### 3D
data <- read.delim(file="colonCancer/nucleosome_evaluation/nucleosomeOccupancy_rawCoverage_Non-Cancer.merged.NFmarker.profile.txt", header = F)
data = data[data$V1>=-900 & data$V1<=900,]
data$V2 = data$V2/mean(data$V2)
data$V3 = data$V3/mean(data$V3)
pdf(file="Manuscript/figures/Fig. 3D.pdf", width = 4, height = 3.5)
plot(V2~V1, data, type="l", ylab = "Relative nucleosome occupancy", xlab = "Distance to TSS (bp)", ylim = c(0.5,1.5), xaxt='n', col="red")
lines(V3~V1, data, col="blue")
axis(side = 1, at=seq(-800,800,by=200))
legend("topright", legend = c("DANPOS","Raw coverage"), lty=1, col=c("red", "blue"))
dev.off()
### 3E
data <- read.delim(file="colonCancer/nucleosome_evaluation/nucleosomeOccupancy_rawCoverage_Non-Cancer.merged.PASmarker.profile.txt", header = F)
data = data[data$V1>=-900 & data$V1<=900,]
data$V2 = data$V2/mean(data$V2)
data$V3 = data$V3/mean(data$V3)
pdf(file="Manuscript/figures/Fig. 3E.pdf", width = 4, height = 3.5)
plot(V2~V1, data, type="l", ylab = "Relative nucleosome occupancy", xlab = "Distance to PAS (bp)", ylim = c(0.5,1.5), xaxt='n', col="red")
lines(V3~V1, data, col="blue")
axis(side = 1, at=seq(-800,800,by=200))
legend("topright", legend = c("DANPOS","Raw coverage"), lty=1, col=c("red", "blue"))
dev.off()

########## Fig. 4 ########## 
plotFig4ROC <- function(data){
  aucs = c()
  pred_T <- prediction(data$`Occupancy_TSS`, data$Label)
  perf_T <- performance(pred_T, "tpr", "fpr")
  aucs[1] <- round(performance(pred_T, measure = "auc")@y.values[[1]], digits = 4)
  
  pred_P <- prediction(data$`Occupancy_PAS`, data$Label)
  perf_P <- performance(pred_P, "tpr", "fpr")
  aucs[2] <- round(performance(pred_P, measure = "auc")@y.values[[1]], digits = 4)
  
  pred_occ <- prediction(data$`Occupancy_PAS+TSS`, data$Label)
  perf_occ <- performance(pred_occ, "tpr", "fpr")
  aucs[3] <- round(performance(pred_occ, measure = "auc")@y.values[[1]], digits = 4)
  
  pred_fuzT <- prediction(data$`Fuzziness_TSS`, data$Label)
  perf_fuzT <- performance(pred_fuzT, "tpr", "fpr")
  aucs[4] <- round(performance(pred_fuzT, measure = "auc")@y.values[[1]], digits = 4)
  
  
  pred_fuzP <- prediction(data$`Fuzziness_PAS`, data$Label)
  perf_fuzP <- performance(pred_fuzP, "tpr", "fpr")
  aucs[5] <- round(performance(pred_fuzP, measure = "auc")@y.values[[1]], digits = 4)
  
  pred_fuz <- prediction(data$`Fuzziness_PAS+TSS`, data$Label)
  perf_fuz <- performance(pred_fuz, "tpr", "fpr")
  aucs[6] <- round(performance(pred_fuz, measure = "auc")@y.values[[1]], digits = 4)
  
  df1 <- data.frame(type=as.factor(c(rep("TSS",length(perf_T@x.values[[1]])),rep("PAS",length(perf_P@x.values[[1]])),rep("Cmb",length(perf_occ@x.values[[1]])))),
                    FalsePositive=c(perf_T@x.values[[1]], perf_P@x.values[[1]], perf_occ@x.values[[1]]), 
                    TruePositive=c(perf_T@y.values[[1]], perf_P@y.values[[1]], perf_occ@y.values[[1]]))
  p1 <- ggplot(df1, aes(x=FalsePositive, y=TruePositive, color=type)) + geom_line(lwd=0.5) + labs(x="False positive rate", y="True positive rate") +
    scale_color_manual(values=c("red","green","blue"),name = "AUC", labels = paste0(c("Cmb (", "PAS (", "TSS ("), aucs[3:1], ")")) +
    theme(aspect.ratio=1, legend.position = c(0.7,0.2), legend.key = element_rect(fill=NA))
  df2 <- data.frame(type=as.factor(c(rep("TSS",length(perf_fuzT@x.values[[1]])),rep("PAS",length(perf_fuzP@x.values[[1]])),rep("Cmb",length(perf_fuz@x.values[[1]])))),
                    FalsePositive=c(perf_fuzT@x.values[[1]], perf_fuzP@x.values[[1]], perf_fuz@x.values[[1]]), 
                    TruePositive=c(perf_fuzT@y.values[[1]], perf_fuzP@y.values[[1]], perf_fuz@y.values[[1]]))
  p2 <- ggplot(df2, aes(x=FalsePositive, y=TruePositive, color=type)) + geom_line(lwd=0.5) + labs(x="False positive rate", y="True positive rate") +
    scale_color_manual(values=c("black","purple","#E7872B"), name = "AUC", labels = paste0(c("Cmb (", "PAS (", "TSS ("), aucs[6:4], ")")) +
    theme(aspect.ratio=1, legend.position = c(0.7,0.2), legend.key = element_rect(fill=NA))
  return(list(p1,p2))
}
data <- as.data.frame(read_excel(path="finalModelRst/20230117_result_v1.1.xlsx", sheet = "US1_proba"))
plots_1 <- plotFig4ROC(data)
data <- as.data.frame(read_excel(path="finalModelRst/20230117_result_v1.1.xlsx", sheet = "CN_proba"))
plots_2 <- plotFig4ROC(data)
pdf(file="newFigures and tables/Fig. 4C-F.pdf", width = 6.5, height = 6.5)
ggarrange(plots_1[[1]], plots_2[[1]], plots_1[[2]], plots_2[[2]], nrow = 2, ncol = 2, 
          labels = c("C", "D", "E", "F"), font.label = list(size = 10, color = "black", face = "bold"))
dev.off()

########## Fig. 5 ##########
plotROCwtWPS <- function(probData){
  data <- as.data.frame(probData)
  aucs=c()
  df_roc=data.frame()
  for(i in 1:5){
    pred <- prediction(data[,i+2], data$Label)
    perf <- performance(pred, "tpr", "fpr")
    aucs[i] <- round(performance(pred, measure = "auc")@y.values[[1]], digits = 4)
    df_roc <- rbind(df_roc,data.frame(type=rep(myModels[i], length(perf@x.values[[1]])),FalsePositive=perf@x.values[[1]], TruePositive=perf@y.values[[1]]))
  }
  df_roc$type=factor(df_roc$type, levels = myModels)
  p1 <- ggplot(df_roc, aes(x=FalsePositive, y=TruePositive, color=type)) + geom_line(lwd=0.5) + 
    labs(x="False positive rate", y="True positive rate") +
    scale_color_manual(values=myColors, name = "AUC", labels = paste0(aucs, " (", myModels, ")")) +
    theme(aspect.ratio=1, legend.position = c(0.7,0.3), legend.key = element_rect(fill=NA), legend.background = element_blank(), legend.title = element_text(size=10))
  return(p1)  
}
myColors=c("purple", "#317EC2", "#5AAA46", "#E7872B","red")
myModels=c("Methylation", "Occupancy", "Fuzziness", "WPS","Multimodal")
data=as.data.frame(read_xlsx(path = "finalModelRst/20230117_result_fromChaorong_v1.1.xlsx", sheet = "US1_proba"))
data=data[,c(1:3,10,11,8,9)]
p1=plotROCwtWPS(data)
data=as.data.frame(read_xlsx(path = "finalModelRst/20230117_result_fromChaorong_v1.1.xlsx", sheet = "CN_proba"))
data=data[,c(1:3,10,11,8,9)]
p2=plotROCwtWPS(data)
data=as.data.frame(read_xlsx(path = "finalModelRst/20230117_result_fromChaorong_v1.1.xlsx", sheet = "CrossCohort_TrainOnUS2.1_proba"))
p3=plotROCwtWPS(data)
pdf(file="newFigures and tables/Fig5A-C.pdf", width = 6.5, height = 3.5)
ggarrange(p1, p2, p3, ncol=3, nrow=1, labels = c("A", "B", "C"))
dev.off()

plotProb <- function(df, outFile1, outFile2){
  colnames(df)=c("label", "Methylation", "Occupancy", "Fuzziness", "WPS")
  correlations <- cor(df[,-1], method = "spearman")
  df_p <- t(df[,-1])
  colnames(df_p)=c(1:ncol(df_p))
  df$label[df$label==1]="Cancer"
  df$label[df$label==0]="NonCancer"
  pheatmap(df_p, cluster_cols = F, cluster_rows = F, show_colnames = F, show_rownames = T, 
           breaks = seq(0,1,length.out = 101), legend_labels = seq(0,1,0.2), border_color = NA,
           color = colorRampPalette(c("navy", "white", "red"), space="Lab")(100), filename = outFile1,
           annotation_col = data.frame(conditions=as.factor(df$label)), annotation_colors = list(conditions=c(NonCancer="gray", Cancer="darkred")))
  pdf(file=outFile2, width = 3.25, height = 3.25)
  corrplot(correlations, method = "color", type="lower", addCoef.col="yellow", tl.col="black", is.corr = F, addgrid.col="gray", 
           cl.lim = c(0,1.0), cl.length = 5) 
  dev.off()
}
data=as.data.frame(read_xlsx(path = "finalModelRst/20230117_result_fromChaorong_v1.1.xlsx", sheet = "US1_proba"))
data=data[,c(2,3,10,11,8)]
plotProb(data,"newFigures and tables/Fig5-US1hp.pdf", "newFigures and tables/Fig5-US1cor.pdf")

data=as.data.frame(read_xlsx(path = "finalModelRst/20230117_result_fromChaorong_v1.1.xlsx", sheet = "CN_proba"))
data=data[,c(2,3,10,11,8)]
plotProb(data,"newFigures and tables/Fig5-CNhp.pdf", "newFigures and tables/Fig5-CNcor.pdf")

data=as.data.frame(read_xlsx(path = "finalModelRst/20230117_result_fromChaorong_v1.1.xlsx", sheet = "CrossCohort_TrainOnUS2.1_proba"))
data=data[,-c(1,7)]
plotProb(data,"newFigures and tables/Fig5-US2hp.pdf", "newFigures and tables/Fig5-US2cor.pdf")

########## Fig. 6 ##########
myColors=c("red", "purple", "#317EC2", "#E7872B")
myModels=c("Multimodal", "Methylation", "Occupancy", "WPS")
plotROCFig6 <- function(data){
  aucs=c()
  df_roc <- data.frame()
  for(i in 1:4){
    pred <- prediction(data[,i+2], data$Label)
    perf <- performance(pred, "tpr", "fpr")
    aucs[i] <- round(performance(pred, measure = "auc")@y.values[[1]], digits = 4)
    df_roc <- rbind(df_roc,data.frame(type=rep(myModels[i], length(perf@x.values[[1]])),FalsePositive=perf@x.values[[1]], TruePositive=perf@y.values[[1]]))
  }
  df_roc$type=factor(df_roc$type, levels = myModels)
  p <- ggplot(df_roc, aes(x=FalsePositive, y=TruePositive, color=type)) + geom_line(lwd=0.5) + labs(x="False positive rate", y="True positive rate") +
    scale_color_manual(values=myColors, name = "AUC", labels = paste0(aucs, " (", myModels, ")")) +
    theme(aspect.ratio=1, legend.position = c(0.7,0.3), legend.key = element_rect(fill=NA), legend.background = element_blank(), legend.title = element_text(size=10))
  return(p)
}
data <- as.data.frame(read_excel(path="finalModelRst/20230131_cfTAPS_corrected.xlsx", sheet = "ctrl-HCC_Proba"))
data <- data[,c(1,2,8,3,4,6)]
p1 <- plotROCFig6(data)
data <- as.data.frame(read_excel(path="finalModelRst/20230131_cfTAPS_corrected.xlsx", sheet = "ctrl-PDAC_Proba"))
data <- data[,c(1,2,8,3,4,6)]
p2 <- plotROCFig6(data)

data <- as.data.frame(read_excel(path="finalModelRst/20230201_cfTAPS_3class.xlsx", sheet = "3class_prediction"))
data <- data[,c(1,2,8,3,4,6)]
accuracies = c()
for(i in 1:4){
  accuracies[i] = nrow(data[data[,i+2] == data$Label,])/nrow(data)
}
df <- data.frame(model=myModels, accuracy=accuracies)
df$model=factor(df$model, levels = myModels)
p3 <- ggplot(df, aes(x=model, y=accuracy)) + geom_bar(stat="identity",width = 0.8, color=myColors, fill=NA) + ylim(0,0.8) + labs(x="", y="Accuracy")+
  geom_text(aes(label = round(accuracy,digits = 4)), vjust=-0.3) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))
pdf(file="newFigures and tables/Fig. 6-CDE.pdf", width = 3, height = 9)
ggarrange(p1, p2, p3, ncol=1, nrow=3, labels = c("C", "D", "E"))
dev.off()