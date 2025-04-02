# textBench: A thorough benchmark of automatic text classification 

## From traditional approaches to large language models

This repository contains a Python 3 implementation for textBench.

## Setup environment

Clone this repository in your machine. Execute the installation under the settings directory.

```
git clone https://github.com/waashk/textBench.git
```

You have two options for installing: virtualenv or Docker.

### Installing and Activating through virtualenv

```
cd textBench/settings/
bash setupENV.sh
```

For activating the environment:

```
source env/bin/activate
```

### Installing and Activating through Dockerfile

@TODO

```
cd textbench/settings/
docker build -t textbench:1.0 .
```

For activating the Docker:

```
cd textbench/
docker run --gpus all --ipc=host --ulimit memlock=-1 --rm --name textbench -p 8888:8888 -v `pwd`:/home/textbench -i -t textbench:1.0 /bin/bash
```

## Requirements

This project is based on ```python==3.9```, and it requires ```Docker version >= 24.0.1``` (if installed through Docker).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Examples

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TB_CONFIG_NAME="config_default.yaml"
python main.py 
```
@TODO: re-send them in the .parquet format.

## Automatic Text Classification Datasets

| **Dataset**  | **Size** | **Dim.** | **# Classes** | **Density** | **Skewness**         | **Link**                                       |
|--------------|----------|----------|---------------|-------------|----------------------|------------------------------------------------|
| DBLP         | 38,128   | 28,131   | 10            | 141         | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555264) |
| Books        | 33,594   | 46,382   | 8             | 269         | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555256) |
| ACM          | 24,897   | 48,867   | 11            | 65          | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555249) |
| 20NG         | 18,846   | 97,401   | 20            | 96          | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555237) |
| OHSUMED      | 18,302   | 31,951   | 23            | 154         | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555276) |
| Reuters90    | 13,327   | 27,302   | 90            | 171         | Extremely Imbalanced | [LINK](https://doi.org/10.5281/zenodo.7555298) |
| WOS-11967    | 11,967   | 25,567   | 33            | 195         | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555385) |
| WebKB        | 8,199    | 23,047   | 7             | 209         | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555368) |
| Twitter      | 6,997    | 8,135    | 6             | 28          | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7554707) |
| TREC         | 5,952    | 3,032    | 6             | 10          | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555342) |
| WOS-5736     | 5,736    | 18,031   | 11            | 201         | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555379) |
| SST1         | 11,855   | 9,015    | 5             | 19          | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555319) |
| pang_movie   | 10,662   | 17,290   | 2             | 21          | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555283) |
| Movie Review | 10,662   | 9,070    | 2             | 21          | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555273) |
| vader_movie  | 10,568   | 16,827   | 2             | 19          | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555354) |
| MPQA         | 10,606   | 2,643    | 2             | 3           | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555268) |
| Subj         | 10,000   | 10,151   | 2             | 24          | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555339) |
| SST2         | 9,613    | 7,866    | 2             | 19          | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555310) |
| yelp_reviews | 5,000    | 23,631   | 2             | 132         | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555396) |
| vader_nyt    | 4,946    | 12,004   | 2             | 18          | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555361) |
| agnews       | 127,600  | 39,837   | 4             | 37          | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555424) |
| yelp_2013    | 335,018  | 62,964   | 6             | 152         | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555898) |
| imdb_reviews | 348,415  | 115,831  | 10            | 326         | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555547) |
| sogou        | 510,000  | 98,974   | 5             | 588         | Balanced             | [LINK](https://doi.org/10.5281/zenodo.5259056) |
| medline      | 860,424  | 125981   | 7             | 77          | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555820) |

To guarantee the reproducibility of the obtained results, all datasets and their respective CV train-test partitions are available on in the respective LINK column.

Each dataset contains the following files:
- data.parquet: pandas DataFrame with texts and associated labels for each document.
- splits/split_\<k\>.pkl:  pandas DataFrame with k-cross validation partition.
- tfidf/: the TFIDF representation for each fold in the CSR matrix format. (.gz)

## Output

The outputs of the main.py script are:

- saida_\<method\>.json: A JSON file containing general information about the execution, including i. reduction, ii. time, iii. hyperparameters, among others.
- measures.fold_<k>.json: Includes micro-f1, macro-f1, the confusion matrix, time_train, time_predict measures, and the machine name you used to generate the results for this specific fold k.
- pred.fold_<k>.parquet: A pandas dataframe containing y_pred and y_test for this specific fold k.
- proba.fold_<k>.gz: A numpy matrix generated by dump_svmlight_file(zero_based=False) and compressed with gzip, containing the posterior probabilities provided by the classifier applied to specific fold k.

## Citation

```
@article{cunha2025sigir,
  title={textBench: A thorough benchmark of automatic text classification. From traditional approaches to large language models},
  author={Cunha, Washington},
  booktitle={Proceedings of the 48th International ACM SIGIR conference on Research and Development in Information Retrieval},
  year={2025},
  doi = {https://dl.acm.org/doi/10.1145/XXXXXXX.XXXXXXX},
}
```

