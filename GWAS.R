## GAPIT
library(multtest)
library(gplots)
#library(LDheatmap)
library(genetics)
library(ape)
library(EMMREML)
library(compiler) #this library is already installed in R
library("scatterplot3d")
source("http://www.zzlab.net/GAPIT/emma.txt")
source("/home/pq26/gwas-bc/blue-blup/gapit_functions.R")


# read in trait, SNP, kinship data
myY = read.table('blue_blup.txt',header=TRUE,stringsAsFactors=FALSE)
myG = read.table('GBS_454_258K_CR06_AGPv4.hmp.txt',sep='\t',header=FALSE,stringsAsFactors=FALSE)
myCV = read.table('../CE_FT_alllines_01202019_wrapper.txt',header=TRUE,stringsAsFactors=FALSE, check.names=FALSE)[,c(1,10)]
myCV = myCV[!is.na(myCV[,-1]),]
colnames(myCV) = c('Taxa','17ft')

myGAPIT <- GAPIT(
  Y=myY,
  G=myG,
  CV=myCV,
  group.from=nrow(myY),
  group.to=nrow(myY),
  group.by=1,
  Major.allele.zero=T,
  Model.selection=TRUE)
