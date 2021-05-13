import torch
from ImgToPatches import ImgToPatches
from Mixer import Mixer


class MLPMixer(torch.nn.Module):
    def __init__(self, num_mixers, channels, img_width, img_height, patch_width, patch_height, lin_init_fn=None):
        super(MLPMixer, self).__init__()

        if (int(num_mixers) < 0):
            raise TypeError(
                "MLP-Mixer architecture cannot contain a negative number of mixers.")

        self.num_mixers = int(num_mixers)

        if (num_mixers == 0):
            raise TypeError("")

        # Does its own validation
        self.to_patches = ImgToPatches(channels,
                                       img_width,
                                       img_height,
                                       patch_width,
                                       patch_height)

        flat_patch_dim = patch_width*patch_height
        self.ppfc = torch.nn.Linear(flat_patch_dim, flat_patch_dim)

        self.mixers = [Mixer(self.to_patches._num_patch_channels,
                             patch_width,
                             patch_height,
                             lin_init_fn=lin_init_fn)
                       for _ in range(self.num_mixers)]

        self.phln = torch.nn.LayerNorm(60)

    def forward(self, x):
        print(f'Shape of initial image stack: {x.shape}')
        x = self.to_patches(x)
        print(f'Shape after patching:         {x.shape}')
        x = self.ppfc(x)
        print(f'Shape after per-patch FC:     {x.shape}')
        for m in self.mixers:
            x = m(x)
        print(f'Shape after mixers:           {x.shape}')
        x = torch.mean(x, dim=2)
        print(f'Shape after average pooling:  {x.shape}')
        #x = torch.flatten(x, start_dim=1, end_dim=2)
        #print(f'Shape after flatten:          {x.shape}')
        x = self.phln(x)
        return x


def main():
    print("MLP-Mixer Example")

    torch.set_num_threads(8)

    # Create an example image batch tensor of the correct shape.
    # Let's say a batch of 64 256x320 RGB (3-channel) images.
    batch_size = 64
    channels = 3
    img_width = 256
    img_height = 320
    patch_width = patch_height = 64
    img = torch.arange(batch_size*channels*img_width*img_height,
                       dtype=torch.float32).reshape(batch_size, channels, img_width, img_height)

    # Declare the model
    model = MLPMixer(num_mixers=3,
                     channels=channels,
                     img_width=img_width,
                     img_height=img_height,
                     patch_width=patch_width,
                     patch_height=patch_height)

    # Feed forward into model
    model(img)


if __name__ == "__main__":
    main()
