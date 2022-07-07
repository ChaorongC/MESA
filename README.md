# MESA

<ins>M</ins>ultimodal <ins>E</ins>pigenetic <ins>S</ins>equencing <ins>A</ins>nalysis (MESA) of Cell-free DNA for Non-invasive Cancer Detection

## Dependencies
- Python >=3.6
- deepTools
- bedtools
- DANPOS2
- UCSC tools
- Python Package
  -  pandas
  -  numpy
  -  scikit-learn = 0.24.2
  -  joblib
  -  itertools
  -  boruta_py

## Usage
Clone the repository with `git`:
```shell
git clone https://github.com/ChaorongC/MESA
cd MESA
```

Or download the repository with `wget`:
```shell
wget https://github.com/ChaorongC/MESA/archive/refs/heads/main.zip
unzip MESA-main.zip
cd MESA-main
```

There are two python document in the root directory: `MESA.py`, `demo.py`
In 'MESA.py', function `SBS_LOO()` is for sequential backward selection (SBS) in a single type of feature, and `calculate_combine()` is for combining SBS results on different types of features then return the multimodal prediction result.
In 'demo.py', we show an example of how **MESA** perform sequential featurre selection on types of features and combines them into a multimodal cancer detection model.


## Authors
- Yumei Li (yumei.li@uci.edu)
- JianFeng Xu (Jianfeng@heliohealth.com)
- Chaorong Chen (chaoronc@uci.edu)
- Wei Li (wei.li@uci.edu)
