import os
import numpy as np
from skimage.color import rgb2gray

import torch
import monai
from monai.transforms import (
    AsChannelFirst,
    AsChannelLast,
    CastToType,
    Compose,
    Lambda,
    Resize,
    ResizeWithPadOrCrop,
    ScaleIntensity,
    ToTensor,
    ToNumpy,
)
from monai.transforms.compose import Transform

class Gray2Rgb(Transform):
    """
    Converts gray image (a single color channel) to RGB (three color channels, identical to channel 0)

    Args:
        None
    """

    def __init__(self) -> None:
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`.
        """
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        if img.shape[-1]==1:
            img = np.concatenate((img,img,img),axis=2)
        elif img.shape[-1]==3:
            pass
        else:
            raise ValueError('Input img to Gray2Rgb needs to have 1 or three channels, but not %d.'%(img.shape[-1]))
        return img

class Rgb2Gray(Transform):
    """
    Converts gray image (a single color channel) to RGB (three color channels, identical to channel 0)

    Args:
        None
    """

    def __init__(self) -> None:
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`.
        """
        if img.ndim == 2:
            pass
        elif img.ndim == 3:
            assert img.shape[2]==3
            img = rgb2gray(img)
        else:
            raise ValueError('Input img to Rgb2Gray needs to have three channels, but not %d.'%(img.shape[-1]))
        return img

class DeepVOG3DModel:
    def __init__(self, device="cuda", fn_model_weights=None, 
                 video_width=320, video_height=240):
        if fn_model_weights is None:
            fn_model_weights = 'best_metric_model_dv3d_segmentation2d_dict_withdropout.pth'
        assert device=="cuda" or device=="cpu"
        if device=="cuda" and not torch.cuda.is_available():
            device=="cpu"
            print('Warning: "cuda" device not available for DeepVOG3DModel. Defaulting to "cpu".')
        # set up model
        device = torch.device(device)
        model = monai.networks.nets.UNet(
            dimensions=2,
            in_channels=3,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            dropout=0.5,
            num_res_units=2,
            ).to(device)
        # load model weights
        base_dir = os.path.dirname(__file__)
        ff_model_weights = os.path.join(base_dir, fn_model_weights)
        model.load_state_dict(torch.load(ff_model_weights))
        model.eval()
        # set up transformation pipeline: forward and inverse
        sigmoid = torch.nn.Sigmoid()
        video_shape = (video_height, video_width)
        if video_shape == (240,320):
            transforms = Compose(
                    [
                        Gray2Rgb(),
                        AsChannelFirst(),
                        ScaleIntensity(),
                        CastToType(np.float32),
                        ToTensor(),
                    ]
                )
        else:
            transforms = Compose(
                    [
                        Gray2Rgb(),
                        AsChannelFirst(),
                        ScaleIntensity(),
                        Resize(spatial_size=(240,320)), # ResizeWithPadOrCrop
                        CastToType(np.float32),
                        ToTensor(),
                    ]
                )
        transforms_inv_seg = Compose(
                [
                    Lambda(lambda im: sigmoid(im)),
                    ToNumpy(),
                    AsChannelLast(),
                ]
            )
        # if resizing to original shape should be necessary:
        #    Resize(spatial_size=video_shape), #without resize might be better...
        # store
        self.video_shape = video_shape
        self.fn_model_weights = fn_model_weights
        self.model = model
        self.device = device
        self.transforms = transforms
        self.transforms_inv_seg = transforms_inv_seg

    def get_model(self):
        '''
        Returns
        -------
        model: pytorch/monai 3D UNet/VNet model
            model can handled as any other torch.nn.Module

        '''
        return self.model
    
    def empty_gpu_cache(self):
        torch.cuda.empty_cache()
    
    def predict(self,x):
        '''
        Parameters
        ----------
        x : np.array of shape (batch_size, height, width, nr_channels)
            nr_channels can be 

        Returns
        -------
        y : np.array of shape (batch_size, height, width, 4)
            The four channels are (from 0 to 3): pupil, iris, glints, visible_map

        '''
        # if only a single image, add a mini-batch dimension
        if x.ndim==3:
            x = x[None, ...]
        # for mini-batches, transform, process, and inverse transform
        x_trans = []
        for img in x:
            x_trans.append(self.transforms(img))
        x_trans = torch.stack(x_trans).to(self.device)
        # run inference
        y_tensor = self.model(x_trans)
        # inverse transform
        y = self.transforms_inv_seg(y_tensor)
        # for some reason, the order of axes now is [4, height, width, batch_size] -> swap axes 0 and 3
        y = np.swapaxes(y,0,3) 
        return y

def load_DeepVOG3D(video_width=320, video_height=240):
    return DeepVOG3DModel(video_width=video_width, video_height=video_height)