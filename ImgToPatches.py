import torch
import einops

# Image input shape:  [batch_size, channels, width, height]
# Patch output shape: [batch_size, num_patches, channels, patch_width, patch_height]
# Where the image width and height are divisible by the patch width and height respectively.
class ImgToPatches(torch.nn.Module):
    def __init__(self, width, height, patch_width, patch_height):
        super(ImgToPatches, self).__init__()

        if (width % patch_width) != 0:
            raise ValueError('Image width must be divisible by patch width.')
        if (height % patch_height) != 0:
            raise ValueError('Image height must be divisible by patch height.')

        self.patch_width = patch_width
        self.patch_height = patch_height

    def forward(self, img):
        img = img.unfold(2, self.patch_width, self.patch_width).unfold(3, self.patch_height, self.patch_height)
        return einops.rearrange(img, 'b c xp yp x y -> b (xp yp) c x y')
        

# Create an example image batch tensor of the correct shape.
# Let's say a batch of 64 256x320 RGB images.
img = torch.arange(64*3*256*320).reshape(64, 3, 256, 320)
print(img.shape)

# Extract 20 64x64 patches
patchsplit = ImgToPatches(256, 320, 64, 64)
#patchsplit = torch.jit.trace(patchsplit, img)
patches = patchsplit(img)
print(patches.shape)
