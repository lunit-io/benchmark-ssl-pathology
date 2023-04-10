# Benchmarking Self-Supervised Learning on Diverse Pathology Datasets

[[`Project page`]](https://lunit-io.github.io/research/publications/pathology_ssl/) [[`arxiv`]](https://arxiv.org/abs/2212.04690)

Official PyTorch Implementation and pre-trained models for `Benchmarking Self-Supervised Learning on Diverse Pathology Datasets`.




# Pre-trained weights
We execute the largest-scale study of SSL pre-training on pathology image data. Our study is conducted using 4 representative SSL methods below on diverse downstream tasks. We establish that large-scale domain-aligned pre-training in pathology consistently out-performs ImageNet pre-training.

1. `bt_rn50_ep200.torch`: ResNet50 pre-trained using [Barlow Twins ](https://arxiv.org/abs/2103.03230)
2. `mocov2_rn50_ep200.torch`: ResNet50 pre-trained using [MoCoV2](https://arxiv.org/abs/2003.04297)
3. `swav_rn50_ep200.torch`: ResNet50 pre-trained using [SwAV](https://arxiv.org/abs/2006.09882)
4. `dino_small_patch_${patch_size}_ep200.torch`: ViT-Small/`${patch_size}` pre-trained using [DINO](https://arxiv.org/abs/2104.14294)

We provide SSL weights of ResNet50 and ViT-S backbone pre-trained on 19M patches from TCGA. Note that, all weights are pre-trained for 200 ImageNet epochs. Please, see below example for using pre-trained weights.

## ResNet50-based weights
```python
import torch
from torchvision.models.resnet import Bottleneck, ResNet


class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-pathology-ssl/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def resnet50(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model


if __name__ == "__main__":
    # initialize resnet50 trunk using BT pre-trained weight
    model = resnet50(pretrained=True, progress=False, key="BT")
```

## ViT/S-based weights
```python
import torch
from timm.models.vision_transformer import VisionTransformer


def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-pathology-ssl/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def vit_small(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model


if __name__ == "__main__":
    # initialize ViT-S/16 trunk using DINO pre-trained weight
    model = vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
```

# License
Pre-trained weights in this repository are bound by ''Public License'' issued from Lunit Inc.
Note that, the weights might be used non-commercially, meaning that the weights are supposed to be used for research-only purpose.
Please, see the detail [here](https://github.com/lunit-io/benchmark-pathology-ssl/LICENSE).


# Citation
```
@inproceedings{kang2022benchmarking,
  author={Kang, Mingu and Song, Heon and Park, Seonwook and Yoo, Donggeun and Pereira, Sérgio},
  title={Benchmarking Self-Supervised Learning on Diverse Pathology Datasets},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month={June},
  year={2023},
  pages={TBU}
}
```
