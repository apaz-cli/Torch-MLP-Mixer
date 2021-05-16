import torch
import einops

# Image input shape:  [batch_size, channels, width, height]
# Patch output shape: [batch_size, channels, patch_pixels]
# Where the image width and height are divisible by the patch width and height respectively.

# Note that in the output, patches are stacked with/as channels. The original
# MLP-Mixer paper implements subimage segmentation as convolutions, which
# naturally produces the same result.


class ImgToPatches(torch.nn.Module):
    def __init__(self, num_channels, width, height, patch_width, patch_height):
        super(ImgToPatches, self).__init__()

        if (width % patch_width) != 0:
            raise ValueError('Image width must be divisible by patch width.')
        if (height % patch_height) != 0:
            raise ValueError('Image height must be divisible by patch height.')

        self._num_patch_channels = (
            width//patch_width)*(height//patch_height) * num_channels

        self.patch_width = patch_width
        self.patch_height = patch_height

    def forward(self, img):
        # Extract image patches across width and height
        img = img.unfold(2, self.patch_width, self.patch_width).unfold(
            3, self.patch_height, self.patch_height)
        return einops.rearrange(img, 'b c x y w h -> b (c x y) (w h)')
