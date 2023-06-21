This repository provides code for training and testing state-of-the-art models for grammatical error correction with the official PyTorch implementation.
It is mainly based on `AllenNLP` and `transformers`.

## Requirements
1. We are using amp with deepspeed from **NVIDIA-Apex**
    ```bash
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation ./
    ```
    *Note that the above installation work for window. You can check https://github.com/NVIDIA/apex for more information.

2. Install following packages by conda/pip
    ```bash
    transformers==4.26.1 \
    scikit-learn==1.0.2 \
    numpy==1.24.2 \
    deepspeed==0.8.2 \
    python-Levenshtein \
    packaging \
    wandb \
    ninja
    ```

## Datasets
All the public GEC datasets used in the project can be downloaded from [here](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data).<br>
Synthetically created datasets can be generated/downloaded from [here](https://github.com/awasthiabhijeet/PIE/tree/master/errorify).<br>
Dataset in our case all have been preprocess

## Preprocess Data
1. Transform your data from m2 format to txt format using error.py from stage1
    ```bash
    Using example "python error.py ../../fce.train.gold.bea19.m2 ../../output/train_texts"
    ```
2. To train the model data has to be preprocessed and converted to special format with the command:
    ```bash
    python utils/preprocess_data.py -s SOURCE -t TARGET -o OUTPUT_FILE
    ```
*Generate edits from shell (edit shell file according to your data path):
    ```
    bash scripts/prepare_data.sh
    ```
    
## Pretrained models
<table>
  <tr>
    <th>Pretrained encoder</th>
    <th>Confidence bias</th>
    <th>Min error prob</th>
    <th>CoNNL-2014 (test)</th>
    <th>BEA-2019 (test)</th>
  </tr>
  <tr>
    <td>BERT <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/gector/bert_0_gectorv2.th">[link]</a></td>
    <td>0.1</td>
    <td>0.41</td>
    <td>61.0</td>
    <td>68.0</td>
  </tr>
  <tr>
    <td>RoBERTa <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th">[link]</a></td>
    <td>0.2</td>
    <td>0.5</td>
    <td>64.0</td>
    <td>71.8</td>
  </tr>
  <tr>
    <td>XLNet <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/gector/xlnet_0_gectorv2.th">[link]</a></td>
    <td>0.2</td>
    <td>0.5</td>
    <td>63.2</td>
    <td>71.2</td>
  </tr>
</table>

## Train Model
- Edit deepspeed_config.json according to your config params. Note that lr and batch_size options will be overrided by args. And args.lr indicates batch_size (regardless how many gpus are used, which equals effective_batch_size_per_gpu * num_gpus) * num accumulation steps. See more details at src/trainer.py.

   ```bash
   bash scripts/train.sh
   ```

## Inference
- Edit deepspeed_config.json according to your config params
    ```bash
    bash scripts/predict.sh
    ```

## Known Issues
- In distributed training (num gpu > 1), enable AMP with O1 state may raise ZeroDivision Error, which may be caused by apex, see APEX's github issues for help. Or, you can try a smaller lr to see if the error disappears.
