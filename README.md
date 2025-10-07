# CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction, CVPR 2025

<table align="center">
  <tr>
    <td align="center">
      <img src="image/README/1759793958770.png" alt="CARETrans overview" width="370" />
    </td>
    <td align="center">
      <img src="image/README/1759794067111.png" alt="CARETrans details" width="350" />
    </td>
  </tr>
</table>

## Data Preparation

1. Download the ImageNet-1K dataset files (`ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar`) and place them in the folder `./data`.
2. Run the following command to preprocess the download files:

   ```bash
   cd ./data && bash extract_ILSVRC.sh
   ```
3. Verify that the extracted dataset follows the following Image folder layout:

   ```
   DATA_PATH/
   ├── train/
   │   ├── n01440764/
   │   │   ├── n01440764_10026.JPEG
   │   │   └── ...
   │   └── ...
   └── val/
       ├── n01440764/
       │   ├── ILSVRC2012_val_00000293.JPEG
       │   └── ...
       └── ...
   ```

## Environment Preparation

Before runing the code, please install some necessary packages required by this repository by using the following commands.

```
conda create -n care python=3.8 && conda activate care
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
cd CARETrans && pip install -r requirement.txt
```

## Training

* Use the following command to train your CARE Transformer models. We train CARE Transformers on 12 NVIDIA RTX 4090 GPUs; you can adjust the number of GPUs and other hyperparameters in the script `cmd/train.sh` according to your hardware and computational resources.

```
bash cmd/train.sh
```

## Evaluation

* To evaluate the accuracy of models on the ImageNet-1K dataset, please use the following command:

```
bash cmd/eval.sh 
```

* To evaluate the GMACs, parameters, and GPU latency of models, please use the following command:

```
bash benchmark.sh
```

* To evaluate the latency of models on mobile device, please first convert the model from PyTorch to the .mlmodel format:

Then, following [EfficientFormer](https://github.com/snap-research/EfficientFormer) and [coreml-performance](https://github.com/vladimir-chernykh/coreml-performance), use `coreml/coreml-performance` to evaluate the model on xcode.

Note that our well-trained checkpoints are provided in [Google Drive](https://drive.google.com/drive/folders/1UGRHQKjBGY8EjUtsHLsy9VBXp7rfUqkf?usp=sharing). Download the checkpoints and place the folder `./ckpt` under `./CARETrans`.

The results will be around the following.

|        Method        |   Type   | GMACs | Params (M) | iPhone13 (ms) | Intel i9 (ms) | RTX 4090 (ms) | Top-1 Acc (%) |
| :-------------------: | :------: | :---: | :--------: | :-----------: | :-----------: | :-----------: | :------------: |
|    MobileNetV2-1.0    |   CONV   |  0.3  |    3.5    |      1.0      |      1.1      |      9.3      |      71.8      |
|        MLLA-T        | LA+CONV |  4.2  |    25.0    |      5.1      |     21.3     |     51.5     |      83.5      |
| **Our CARE-S0** | LA+CONV |  0.7  |    7.3    |      1.1      |      4.3      |      9.8      | **78.4** |
| **Our CARE-S1** | LA+CONV |  1.0  |    9.6    |      1.4      |      6.6      |     14.2     | **80.1** |
| **Our CARE-S2** | LA+CONV |  1.9  |    19.5    |      2.0      |      9.4      |     20.4     | **82.1** |

## Bibtex

```
@inproceedings{zhou2025care,
  title={CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction},
  author={Zhou, Yuan and Xu, Qingshan and Cui, Jiequan and Zhou, Junbao and Zhang, Jing and Hong, Richang and Zhang, Hanwang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={20135--20145},
  year={2025}
}
```

## Acknowledgment

Our code is based on [pytorch-image-models](https://github.com/huggingface/pytorch-image-models), [poolformer](https://github.com/sail-sg/poolformer), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), [inceptionnext](https://github.com/sail-sg/inceptionnext), and [metaformer](https://github.com/sail-sg/metaformer).
