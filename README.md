# MyVideoGeneration [WIP]
Conditional video generation with style

## Changing emotion mid sequence

![alt text](./anim/emotion_chain.gif)

## Gan inversion of motion styles

![alt text](./anim/fake_projected.gif)

## Requirements
- Ubuntu 20.04.3 LTS (GNU/Linux 5.4.0-91-generic x86_64)
- CUDA Version: 11.5
- Python>=3.8

## Environment Setup

```
conda create -n CTSVG python=3.8
conda activate CTSVG

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python tqdm pandas scipy scikit-learn
```


