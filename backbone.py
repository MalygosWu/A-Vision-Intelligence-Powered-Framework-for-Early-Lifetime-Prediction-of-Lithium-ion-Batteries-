# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 17:02:47 2025

@author: wcfda
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import warnings
import numpy as np
from PIL import Image
from skimage.feature import hog
warnings.filterwarnings("ignore")

def HOG(images):
    hog_features = []
    for image in images:
        #image = images[0]
        image_feature = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        hog_features.append(image_feature)
    return np.array(hog_features)


def _get_weights_enum_if_exists(model_name):
    """
    Try to return the corresponding weights enum class from torchvision.models (if available).
    Returns (weights_obj, weights_name) or (None, None) if not found.
    """
    # Map common model function -> weights enum name convention in torchvision
    # Examples: efficientnet_b0 -> EfficientNet_B0_Weights
    #           mobilenet_v2 -> MobileNet_V2_Weights
    #           vgg16 -> VGG16_Weights
    parts = model_name.split('_')
    # Build name by capitalizing parts and appending '_Weights'
    name = ''.join([p.capitalize() for p in parts]) + "_Weights"
    weights_enum = getattr(torchvision.models, name, None)
    return (weights_enum, name) if weights_enum is not None else (None, None)


def get_input_size_from_weights_or_model(model, weights_obj=None, default_spatial=224):
    """
    Try to retrieve (C,H,W) expected input size from weights.meta or model.default_cfg; otherwise fallback.
    """
    # 1) weights meta
    if weights_obj is not None:
        try:
            meta = getattr(weights_obj, "meta", None)
            if isinstance(meta, dict):
                size = meta.get("size") or meta.get("input_size")
                if size is not None:
                    if isinstance(size, (tuple, list)) and len(size) == 3:
                        return (int(size[0]), int(size[1]), int(size[2]))
                    elif isinstance(size, int):
                        return (3, int(size), int(size))
        except Exception:
            pass
    # 2) model.default_cfg
    try:
        cfg = getattr(model, "default_cfg", None)
        if isinstance(cfg, dict):
            inp = cfg.get("input_size") or cfg.get("size") or cfg.get("image_size")
            if inp is not None:
                if isinstance(inp, (tuple, list)) and len(inp) == 3:
                    return (int(inp[0]), int(inp[1]), int(inp[2]))
                elif isinstance(inp, int):
                    return (3, int(inp), int(inp))
    except Exception:
        pass
    # 3) fallback: infer channels from first Conv2d, spatial -> default_spatial
    in_ch = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            in_ch = m.in_channels
            break
    if in_ch is None:
        in_ch = 3
    return (int(in_ch), int(default_spatial), int(default_spatial))


# ---------- Dataset ----------
class NumpyGrayDataset(Dataset):
    """
    Wrap numpy grayscale images (N, H, W) and apply torchvision transform.
    Converts each grayscale image to RGB by duplicating the single channel.
    """
    def __init__(self, images_np, transform):
        # images_np: np.ndarray (N,H,W) or (H,W)
        arr = np.asarray(images_np)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        if arr.ndim != 3:
            raise ValueError("images_np must be shape (N,H,W) or (H,W)")
        self.images = arr.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        im = self.images[idx]
        # Binary images might be 0/1; convert to 0..255 uint8
        if im.dtype == np.bool_ or im.max() <= 1.0:
            im_uint8 = (im.astype(np.uint8) * 255)
        else:
            im_uint8 = np.clip(im, 0, 255).astype(np.uint8)
        pil = Image.fromarray(im_uint8, mode='L').convert('RGB')  # convert to 3-channel
        return self.transform(pil)


# ---------- Feature extractor builder ----------
def build_feature_extractor_from_torchvision_model(model, model_name):
    """
    Build a feature-extraction nn.Module that returns 1D pooled feature vectors.
    The wrapper runs model.features (when present) + avgpool (or AdaptiveAvgPool2d(1)) + flatten.
    Works for EfficientNet, MobileNet, VGG variants in torchvision.
    """
    # try to find "features" submodule
    features_module = None
    if hasattr(model, "features"):
        features_module = model.features
    else:
        # Some models keep feature layers under other names - fallback to full model
        features_module = model

    # try to use existing avgpool, else use AdaptiveAvgPool2d(1)
    avgpool = getattr(model, "avgpool", None)
    if avgpool is None:
        avgpool = nn.AdaptiveAvgPool2d(1)

    # Build wrapper
    class _Extractor(nn.Module):
        def __init__(self, features, avgpool_layer):
            super().__init__()
            self.features = features
            self.avgpool = avgpool_layer
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return x

    return _Extractor(features_module, avgpool)


# ---------- Main extraction function ----------
def extract_image_features(images_np, model_names, batch_size=16, num_workers=0):
    """
    images_np: numpy array shape (N, H, W) with grayscale images (0/1 or 0..255)
    model_names: tuple/list of torchvision model function names (strings)
    Returns: dict: {model_name: feature_array (N, feat_dim)}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for model_name in model_names:
        print(f"\nLoading model: {model_name} ...")
        # try to get weights enum
        weights_enum, weights_name = _get_weights_enum_if_exists(model_name)
        model = None
        weights_arg = None
        if weights_enum is not None:
            try:
                weights_arg = getattr(weights_enum, "IMAGENET1K_V1", None) or list(weights_enum)[0]
                # call model builder with weights=weights_arg if signature supports it
                model_fn = getattr(torchvision.models, model_name)
                try:
                    model = model_fn(weights=weights_arg)
                except TypeError:
                    # older torchvision: fallback to pretrained=True
                    model = model_fn(pretrained=True)
            except Exception:
                # fallback attempt
                try:
                    model = getattr(torchvision.models, model_name)(pretrained=True)
                except Exception as e:
                    raise RuntimeError(f"Could not load pretrained {model_name}: {e}")
        else:
            # fallback
            try:
                model = getattr(torchvision.models, model_name)(pretrained=True)
            except Exception as e:
                raise RuntimeError(f"Could not load pretrained {model_name}: {e}")

        model.eval()
        # get expected input size
        C, H, W = get_input_size_from_weights_or_model(model, weights_arg, default_spatial=224)
        input_size = (H, W)
        print(f"  -> expected input size (C,H,W) = {(C,H,W)}; using spatial {input_size}")

        # preprocessing transform
        transform = T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # dataset + loader
        dataset = NumpyGrayDataset(images_np, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # build feature extractor and move to device
        extractor = build_feature_extractor_from_torchvision_model(model, model_name)
        extractor.to(device)
        extractor.eval()

        # run inference
        feats_list = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = extractor(batch)  # shape (B, feat_dim)
                feats_list.append(out.cpu().numpy())

        feats = np.vstack(feats_list)
        results[model_name] = feats
        print(f"  -> extracted features shape: {feats.shape}")

    return results

def extract_features(images, backbone):
    if backbone == 'HOG':
        image_features = HOG(images)
    elif backbone == 'EfficientNet':
        feature_dict = extract_image_features(images, model_names=['efficientnet_b0'], batch_size=16)
        image_features = feature_dict['efficientnet_b0']
    elif backbone == 'MobileNet':
        feature_dict = extract_image_features(images, model_names=['mobilenet_v2'], batch_size=16)
        image_features = feature_dict['mobilenet_v2']
    elif backbone == 'VGG':
        feature_dict = extract_image_features(images, model_names=['vgg16'], batch_size=16)
        image_features = feature_dict['vgg16']
    return image_features


# ----------------------------------------------------------------------------- Main operation
if False:
    N = 100
    H = 128
    W = 128
    # Example: random binary images (0/1)
    images = (np.random.rand(N, H, W) > 0.7).astype(np.uint8)

    # Extract features
    features_dict = extract_image_features(images, model_names=["efficientnet_b0", "mobilenet_v2", "vgg16"], batch_size=16)
    
    # Example: access EfficientNet features
    hog_feats = HOG(images)
    eff_feats = features_dict["efficientnet_b0"]
    mob_feats = features_dict["mobilenet_v2"]
    vgg_feats = features_dict["vgg16"]
    print("\nFinal shapes:")
    print(" EfficientNet-B0:", eff_feats.shape)
    print(" MobileNet-V2  :", mob_feats.shape)
    print(" VGG16         :", vgg_feats.shape)

