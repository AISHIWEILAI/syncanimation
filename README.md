# SyncAnimation: A Real-Time End-to-End Framework for Audio-Driven Human Pose and Talking Head Animation [IJCAI 2025]


<p align='center'>
  <b>
    <a href="https://aishiweilai.github.io/syncanimation.github.io/">Project Page</a>
    | 
    <a href="https://arxiv.org/abs/2501.14646">Paper (arXiv)</a>
  </b>
</p> 

 ![SyncAnimation Demo](assets/image/pipline.png)
 
Generating talking avatar driven by audio remains a significant challenge. Existing methods typically require high computational costs and often lack sufficient facial detail and realism, making them unsuitable for applications that demand high real-time performance and visual quality. Additionally, while some methods can synchronize lip movement, they still face issues with consistency between facial expressions and upper body movement, particularly during silent periods. In this paper, we introduce SyncAnimation, the first NeRF-based method that achieves audio-driven, stable, and real-time generation of speaking avatar by combining generalized audio-to-pose matching and audio-to-expression synchronization. By integrating AudioPose Syncer and AudioEmotion Syncer, SyncAnimation achieves high-precision poses and expression generation, progressively producing audio-synchronized upper body, head, and lip shapes. Furthermore, the High-Synchronization Human Renderer ensures seamless integration of the head and upper body, and achieves audio-sync lip.

---



##  Introduction

Most existing audio-driven talking head synthesis methods focus only on the facial region, pasting other parts like the torso from the original image, which leads to audio inconsistency between facial movements, lips, and body motion. **SyncAnimation** addresses this issue by ensuring:

- Audio-Body Consistency
- Audio-Face Consistency
- Audio-Lips Consistency

![SyncAnimation Demo](assets/image/objectives.png)

---



##  Installation & Dependencies



### Linux / Ubuntu

The environment setup of this project follows the installation process of [SyncTalk](https://github.com/ZiqiaoPeng/SyncTalk). Below is the recommended installation process on Ubuntu (tested on Ubuntu 20.04 with PyTorch 1.12.1 + CUDA 11.3):

```bash
git clone https://github.com/AISHIWEILAI/syncanimation.git
cd syncanimation

# It is recommended to use a conda environment
conda create -n syncanimation python==3.8.8
conda activate syncanimation

# Install PyTorch and torchvision (choose versions according to your CUDA version)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

sudo apt-get install portaudio19-dev
pip install -r requirements.txt

# Install required modules (freqencoder / gridencoder / shencoder / raymarching)
pip install ./freqencoder
pip install ./shencoder
pip install ./gridencoder
pip install ./raymarching

# Install PyTorch3D (if issues occur, use the fallback script)
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1121/download.html
# Or:
python ./scripts/install_pytorch3d.py

# Install TensorFlow GPU version
pip install tensorflow-gpu==2.8.1

# Download HuBERT weights (required for preprocessing / training)
bash scripts/setup_hubert.sh
```

> **Note**: You may encounter compatibility issues when installing PyTorch3D. It is recommended to use the scripts/install_pytorch3d.py script as a fallback.

---



##  Usage

All helper scripts live under `scripts/` and should be run from the project root. Below we use `May` as an example subject ID; replace it with your own ID for other videos.

**Data layout:** place the input video at `data/<ID>/<ID>.mp4` (e.g. `data/May/May.mp4`). Scripts read from `data/<ID>/` and write checkpoints to `model/<ID>/`.


| Script                         | Usage                                                                                      |
| ------------------------------ | ------------------------------------------------------------------------------------------ |
| `scripts/setup_hubert.sh`      | Download HuBERT weights to `data_utils/facebook/` (once during setup)                      |
| `scripts/preprocess.sh`        | Preprocess one subject in foreground: `[ID]`                                               |
| `scripts/preprocess_screen.sh` | Preprocess in screen: `[ID]` |
| `scripts/train.sh`             | Three-stage training + inference: `[ID]` + stage (`all`, `torso`, `face`, `lips`, `infer`) |
| `scripts/train_screen.sh`      | Run full training in screen: `[ID]`                                                      |




### 1. Preprocessing

```bash
# Example: subject ID = May, video at data/May/May.mp4
bash scripts/preprocess_screen.sh May    # screen (recommended)

# or foreground:
bash scripts/preprocess.sh May
```

```bash
python data_utils/process.py --path data/May/May.mp4 --task -1 --asr hubert
```


| Task              | Output                    |
| ----------------- | ------------------------- |
| Audio + HuBERT    | `aud.wav`, `aud_hu.npy`   |
| Frame extract     | `ori_imgs/`               |
| Semantic parsing  | `parsing/`                |
| Background        | `bc.jpg`                  |
| GT / torso        | `gt_imgs/`, `torso_imgs/` |
| Landmarks         | `*.lms`                   |
| 3DMM tracking     | `track_params.pt`         |
| Optical flow + BA | `bundle_adjustment.pt`    |
| Blendshape        | `bs.npy`                  |
| Transforms        | `transforms_*.json`       |


For `aud_Xhu.npy` (256-d), please refer to the [official FaceXHuBERT repository](https://github.com/galib360/FaceXHuBERT).

### 2. Three-stage Training + Inference

After preprocessing, training reads from `data/May/` and saves to `model/May/`:

```bash
bash scripts/train.sh May              # torso, face, infer, lips and infer
bash scripts/train.sh May torso        # torso only, 150k
bash scripts/train.sh May face         # face only, 120k
bash scripts/train.sh May lips         # lips only, 160k
bash scripts/train.sh May infer        # inference only

bash scripts/train_screen.sh May    # screen background
```


| Stage               | Workspace (May example)           |
| ------------------- | --------------------------------- |
| Torso               | `model/May/May_trial_torso_audio` |
| Face / lips / infer | `model/May/May_trial_audio`       |


Example output: `model/May/May_trial_audio/results/ngp_ep0029_infer.mp4`

```bash
python audio_main.py \
    --path data/May --fps 25 --asr_model hubert --test \
    --workspace model/May/May_trial_audio \
    --infer data/inference/c-eng-chi-chi_Xhu.npy \
    --aud data/inference/c-eng-chi-chi_hu.npy \
    --torso --special --bs_au45
```

We provide the preprocessed **May** data and trained results on [Baidu Netdisk](https://pan.baidu.com/s/1a-OqqPpc0_WBZ0zm-t-bgg) (code: `0408`).

---



##  Citation

Please cite the following paper if you use this method, model, or conduct derivative research based on this project:

```bibtex
@inproceedings{ijcai2025p185,
  title     = {SyncAnimation: A Real-Time End-to-End Framework for Audio-Driven Human Pose and Talking Head Animation},
  author    = {Liu, Yujian and Xu, Shidang and Guo, Jing and Wang, Dingbin and Wang, Zairan and Tan, Xianfeng and Liu, Xiaoli},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {James Kwok},
  pages     = {1657--1665},
  year      = {2025},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2025/185},
  url       = {https://doi.org/10.24963/ijcai.2025/185},
}
```

---



##  Acknowledgements

This project is built upon or inspired by the following open-source projects:

- [SyncTalk](https://github.com/ZiqiaoPeng/SyncTalk)
- [ER-NeRF](https://github.com/Fictionarry/ER-NeRF)
- [GeneFace](https://github.com/yerfor/GeneFace)
- [AD-NeRF](https://github.com/YudongGuo/AD-NeRF)
- [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch)

We sincerely thank the authors of these projects for their contributions to the open-source community.

---



##  Disclaimer

By using this project, you agree to comply with all applicable laws and regulations.
You must not use it to generate or disseminate harmful content.
The developers assume no responsibility for any direct, indirect, or consequential damages arising from the use or misuse of this software.
