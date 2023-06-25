import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d=64):
        """
        channels_img: number of channels in the images
        features_d: number of features in the first layer of the discriminator
        """
        super(Discriminator, self).__init__()
        self.D = nn.Sequential(
            # Input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # Output: N x features_d x 32 x 32
            self._block(features_d, features_d*2, kernel_size=4, stride=2, padding=1),
            # Output: N x features_d*2 x 16 x 16
            self._block(features_d*2, features_d*4, kernel_size=4, stride=2, padding=1),
            # Output: N x features_d*4 x 8 x 8
            self._block(features_d*4, features_d*8, kernel_size=4, stride=2, padding=1),
            # Output: N x features_d*8 x 4 x 4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
            # Output: N x 1 x 1 x 1
            nn.Sigmoid(),
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
        
    
    def forward(self, x):
        return self.D(x)
    

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g=64):
        """
        z_dim: dimension of the noise vector
        channels_img: number of channels in the images
        features_g: number of features in the first layer of the generator
        """
        super(Generator, self).__init__()
        self.G = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim, features_g*16, kernel_size=4, stride=1, padding=0), 
            # Output: N x features_g*16 x 4 x 4
            self._block(features_g*16, features_g*8, kernel_size=4, stride=2, padding=1),
            # Output: N x features_g*8 x 8 x 8
            self._block(features_g*8, features_g*4, kernel_size=4, stride=2, padding=1),
            # Output: N x features_g*4 x 16 x 16
            self._block(features_g*4, features_g*2, kernel_size=4, stride=2, padding=1),
            # Output: N x features_g*2 x 32 x 32
            nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(), 
            # Output: N x channels_img x 64 x 64, range [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.G(x)
    

def initialize_weights(model):
    """
    Initialize weights of the model with N(0, 0.02)
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    
def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    D = Discriminator(in_channels, 8)
    initialize_weights(D)
    assert D(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    G = Generator(z_dim, in_channels, 8)
    initialize_weights(G)
    z = torch.randn((N, z_dim, 1, 1))
    assert G(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("All tests passed")