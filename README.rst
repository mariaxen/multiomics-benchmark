====================
multiomics-benchmark
====================

This is the code that accompanies our paper BenMO: Benchmarking predictive machine learning algorithms using MultiOmics data. In this repository, we provide R code for data pre-processing and a combination of Python and R code for modeling, results, and
plots. We also provide Python wrappers for the models that are run using R functions. In addition, we provide model wrappers for optimizing the hyperparameters of the models that do not have the corresponding function currently implemented in Python or R.

Data
===========

The healthy pregnancy dataset contains only samples from women who delivered at term with no complications and is available from http://nalab.stanford.edu/multiomics-pregnancy/. 
The preeclampsia dataset contains both healthy and preeclamptic pregnancies, and is available from https://github.com/ivanam5/Multiomics_Preeclampsia/tree/main/Multiomics_Datasets.
The TCGA data we use here are related to 10 types of cancer, Breast invasive carcinoma (TCGA-BRCA), Colorectal adenocarcinoma (TCGA-COADREAD), Glioma (TCGA-GBMLGG), Head and Neck squamous cell carcinoma (TCGA-HNSC), Pan-kidney cohort (TCGA-KIPAN), Lung squamous cell carcinoma (TCGA-LUSC), Ovarian serous cystadenocarcinoma (TCGA-OV), Stomach and Esophageal carcinoma (TCGA-STES), Thyroid carcinoma (TCGA-THCA), and Uterine corpus endometrial carcinoma (TCGA-UCEC). 
Datasets are available from http://linkedomics.org/.
The majority of TCGA studies contain 5 modalities - miRNA (Hi Seq, Gene level), RNAseq (Hi Seq, Gene level), RPPA (Reverse Phase Protein Array) (Gene level), SCNV (Somatic Copy Number Variation) (Gene level, log-ratio), and DNA methylation (Gene level, HM450K), with the exception of uterine cancer that excludes the RNA modality. 
