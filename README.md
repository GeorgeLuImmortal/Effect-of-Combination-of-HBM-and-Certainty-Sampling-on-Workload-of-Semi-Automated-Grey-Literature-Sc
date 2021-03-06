# Effect of Combination of HBM and Certainty Sampling on Workload of Semi-Automated Grey Literature Screening 

This repository is temporarily associated with paper [Lu, J., Henchion, M. and Mac Namee, B., 2021. Effect of Combination of HBM and Certainty Sampling on Workload of Semi-Automated Grey Literature Screening](https://icml.cc/Conferences/2021/ScheduleMultitrack?event=8362) (to be published at the 3rd Workshop of Human in the Loop, 38<sup>th</sup> ICML, 2021)

## Grey Literature Datasets (to be released)
Csv files in the directory is just for providing a toy experiment samples. The results of this dataset is invalid since the dataset here is used for pretraining the language model (therefore performance will be super high even if only 10 example are used for training).

## Usage

### Dependencies
Tested Python 3.6, and requiring the following packages, which are available via PIP:

* Required: [numpy >= 1.19.5](http://www.numpy.org/)
* Required: [scikit-learn >= 0.21.1](http://scikit-learn.org/stable/)
* Required: [pandas >= 1.1.5](https://pandas.pydata.org/)
* Required: [gensim >= 3.7.3](https://radimrehurek.com/gensim/)
* Required: [matplotlib >= 3.3.3](https://matplotlib.org/)
* Required: [torch >= 1.9.0](https://pytorch.org/)
* Required: [transformers >= 4.8.2](https://huggingface.co/transformers/)
* Required: [Keras >= 2.0.8](https://keras.io/)
* Required: [Tensorflow >= 1.14.0](https://www.tensorflow.org/)
* Required: [FastText model trained with Wikipedia 300-dimension](https://fasttext.cc/docs/en/pretrained-vectors.html)
* Required: [GloVe model trained with Gigaword and Wikipedia 200-dimension](https://nlp.stanford.edu/projects/glove/)
* Required: packaging >= 20.0


### Step 1. Data Processing

The first step is encoding raw text data into different high-dimensional vectorised representations. The raw text data should be stored in directory "corpus_data/", each dataset should have its individual directory, for example, the "animal_by_product/" directory under folder "corpus_data/". The input corpus of documents should consist of plain text files stored in csv format (two files for one corpus, one for documents belong to class A and one for documents for class B) with a columan named as __text__. It should be noted that the csv file must be named in the format _#datasetname_neg_text.csv_ or _#datasetname_pos_text.csv_. Each row corresponding to one document in that corpus, the format can be refered to the csv file in the sample directory "corpus_data/animal_by_product/". Then we can start preprocessing text data and converting them into vectors by:

    python encode_text.py -d dataset_name -t encoding_methods
    
The options of -t are `hbm` (corresponding to the sentence representation generated by the token-level RoBERTa encoder in the paper), `roberta-base`, `fasttext`, and `PV-TD` for example `-t roberta-base,fasttext` means encoding documents by RoBERTa and FastText respectively. The encoded documents are stored in directory "dataset/", e.g. the FastText document representations are stored in "dataset/fasttext/" and HBM and RoBERTa embeddings are stored in "dataset/roberta-base/". It should be noted that the sentence representations for hbm is suffixed by ".pt" and the document representations generated by RoBERTa are suffixed by ".csv"(average all tokens to represent a document) or "\_cls.csv" (using classifier token "\<s\>" to represent a document) For encoding by FastText and PV-TD, you need to download the pretrained FastText model in advance (see __Dependencies__).

### Step 2. Run HBM + Certainty/Uncertainty/FastRead

We can evaluate the HBM + Certainty/Uncertainty:

    python run_active_learning_hbm.py -d datasetname -l num_active_learning_iterations -r random_seeds -s selection_strategy -m max_len

The `num_active_learning_iterations` can be random int numbers up to the size of the datasets, for example  `-l 10` means active learning process iterate 10 loops and 100 documents will be shown to the user (we set the number of labelled examples each step 10 documents).  The `random_seeds` are random state for subsampling initial training set. For example `-r 1988,1999` will training HBM with 2 different training sets, i.e. 10 labelled instances sampled by seed 1988, 10 labelled instances sampled by seed 1999. `max_len` is the max length of sentences for document set for HBM and `selection_strategy` means the selection method used in active learning, the options are `mostConfident` for certainty sampling and `uncertainty` for uncertainty sampling. Additionally, if you want to output attention scores as well, you need `-a` otherwise `-n` for disabling output attention scores.
For HBM + FastRead:

    python run_active_learning_hbm_fastread.py -d datasetname -l num_active_learning_iterations -r random_seeds -m max_len
    
The disable/enable output attention scores is the same as the previous one. For all usages please input `-h`.

The evaluation results are stored in directory "outputs/". Furthermore, the concrete results of each step are stored in "outputs/#datasetname/". The results files starting with "metric_" store various evaluation metric results (AUC, coverage, ACC, etc.) while files starting with "raw_" store the confusion matrix (tp, tn, fp, fn), and files starting with "permutation_" store the index of documents selected in each step.

### Step 3. Run SVM-based method (RoBERTa, PV-TD) + Certainty/Uncertainty/FastRead

We can evaluate SVM-based method (pretrained RoBERTa + SVM or PV-TD + SVM) by: 

    python run_active_learning_svm-based.py -d datasetname -l num_active_learning_iterations -r random_seeds -s selection_strategy -e encoding_method
    
All arguments have the same meaining as those of previous. `encoding_method` means the text representation technique used, the options are `roberta-base` and `PV-TD`. 
Similarly, for RoBERTa/PV-TD + FastRead:

    python run_active_learning_svm-based_fastread.py -d datasetname -l num_active_learning_iterations -r random_seeds -e encoding_method
   
For all configurations, please input `-h`. Similarly the results are stored in "outputs/#datasetname/".


### Step 4. Run ATAL + Certainty/Uncertainty/FastRead

We can evaluate ATAL by: 

    python run_ATAL.py -d datasetname -l num_active_learning_iterations -r random_seeds -s selection_method -a fine_tune_learning_rate
    
All arguments have the same meaning as those of previous, except the `-a fine_tune_learning_rate` which is the learning rate for fine tuning.
Similarly, for ATAL + FastRead:
    
    python run_ATAL_fastread.py -d datasetname -l num_active_learning_iterations -r random_seeds -a fine_tune_learning_rate
    
For all configurations, please input `-h`. Similarly the results are stored in "outputs/#datasetname/".

### Step 5 Visualise attention score

When we run the script in __Step 2__, besides the AUC scores on testing set, we can also get the attention scores of each sentences that measure whether sentences contribute a lot in forming the document representation. Hence, these attention scores can serve as clue of whether the sentences are important or not. The attention scores are stored in "outputs/#dataset_name/attention/". You can visualise this attention scores by playing with the notebook __Visualization_of_informative_sentences.ipynb__.
