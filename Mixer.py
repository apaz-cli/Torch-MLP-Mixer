from ImgToPatches import ImgToPatches
import torch


class MLP(torch.nn.Module):
    def __init__(self, dim, lin_init_fn=None):
        super(MLP, self).__init__()

        # Declare and initialize
        self.lin1 = torch.nn.Linear(dim, dim)
        self.gelu = torch.nn.GELU()
        self.lin2 = torch.nn.Linear(dim, dim)

        if lin_init_fn:
            lin_init_fn(self.lin1.weight.data)
            lin_init_fn(self.lin2.weight.data)

    def forward(self, x):
        x = self.lin1(x)
        x = self.gelu(x)
        x = self.lin2(x)
        return x


class Mixer(torch.nn.Module):
    def __init__(self, channels, patch_width, patch_height, lin_init_fn=None):
        super(Mixer, self).__init__()

        pixels = patch_width*patch_height

        #print(f'pixels: {pixels}, channels: {channels}')

        self.ln1 = torch.nn.LayerNorm(pixels, elementwise_affine=False)
        self.ln2 = torch.nn.LayerNorm(pixels, elementwise_affine=False)

        self.mlp1 = MLP(channels, lin_init_fn=lin_init_fn)
        self.mlp2 = MLP(pixels, lin_init_fn=lin_init_fn)

    # Input of size:
    # [batch_size, channels, pixels]
    def forward(self, x):
        x = self.token_mixing(x)
        x = self.channel_mixing(x)
        return x

    def token_mixing(self, x):
        prev_x = x  # skip connection
        x = self.ln1(x)
        x = torch.transpose(x, 1, 2)  # channels and pixels
        x = self.mlp1(x)
        x = torch.transpose(x, 1, 2)
        return x + prev_x

    def channel_mixing(self, x):
        prev_x = x  # skip connection
        x = self.ln2(x)
        x = self.mlp2(x)
        return x + prev_x


def main():
    print("Mixer Example")

    # Create an example image batch tensor of the correct shape.
    # Let's say a batch of 64 256x320 RGB (3-channel) images.
    batch_size = 64
    channels = 3
    img_width = 256
    img_height = 320

    patch_width = patch_height = 64

    img = torch.arange(batch_size*channels*img_width*img_height,
                       dtype=torch.float32).reshape(batch_size, channels, img_width, img_height)
    print(f"Image batch shape:                 {img.shape}")

    # Extract 20 64x64 patches
    patchsplit = ImgToPatches(
        channels, img_width, img_height, patch_width, patch_height)
    patches = patchsplit(img)
    print(f"Batch of image patch stacks shape: {patches.shape}")

    # Feed forward into Mixer
    mixer = Mixer(patchsplit._num_patch_channels, 64, 64)
    mixer_out = mixer(patches)
    print(f"Mixer output shape:                {mixer_out.shape}")


if __name__ == "__main__":
    main()
