import torch.nn as nn


# Input shape (b,c,t,h,w): torch.Size([512, 1, 12, 40, 130]) for example
class MaskAutoencoder (nn.Module):
    def __init__(self, sample_input, latent_dim = 256):
        super(MaskAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.sample_input = sample_input
        assert sample_input.shape[2] >= 12, "Less than 12-frames per data. \
        \nRequires different architecture (smaller kernel size and additional padding)"

        ### Mask video -> 3D encoded features
        self.encoder = nn.Sequential(
            self.conv_encoder_layer (in_channels=1, out_channels=16, kernel_size=5, padding=0),
            self.conv_encoder_layer (in_channels=16, out_channels=32, kernel_size=3, padding=1),
            self.conv_encoder_layer (in_channels=32, out_channels=64, kernel_size=3, padding=1),
            )

        ### 3D encoded features -> latent representations
        out = self.encoder (sample_input)
        feature_dim = out.view(out.size(0), -1).size(1)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fcn_encode = nn.Sequential(
            self.bottleNeck (in_features=feature_dim, out_features=2048),
            self.bottleNeck (in_features=2048, out_features=1024),
            nn.Linear (in_features=1024, out_features=latent_dim)
            )
        
        ### latent representations -> 3D encoded features
        self.fcn_decode = nn.Sequential(
            self.bottleNeck (in_features=latent_dim, out_features=1024),
            self.bottleNeck (in_features=1024, out_features=2048),
            nn.Linear (in_features=2048, out_features=feature_dim)
            )

        ### 3D encoded features -> reconstruction (symmetric)
        self.sizes = self.get_upsample_size(encoder=self.encoder)
        self.decoder = nn.Sequential(
            nn.Upsample(size=self.sizes[1], mode='trilinear', align_corners=True),
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d (32),
            nn.ReLU(),

            nn.Upsample(size=self.sizes[2], mode='trilinear', align_corners=True),
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d (16),
            nn.ReLU(),

            nn.Upsample(size=self.sizes[3], mode='trilinear', align_corners=True),
            nn.ConvTranspose3d(16, 1, kernel_size=5, padding=0),
            nn.Sigmoid(),
            )
        
        ### Weight Initialization
        self.Kaiming_init()

    def forward(self, x, encode=False):
        x = self.encoder (x)
        b,c,t,h,w = x.shape
        x = self.flatten (x)
        x = self.fcn_encode (x)
        if encode:
            return x
        else:
            x = self.fcn_decode (x)
            x = x.view (b,c,t,h,w)
            x = self.decoder (x)
            return x
        
    @staticmethod
    def conv_encoder_layer (in_channels, out_channels, kernel_size, padding, stride=1, maxpool_size=2, activation=nn.ReLU()):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride
                    ),
            nn.BatchNorm3d (out_channels),
            activation,
            nn.MaxPool3d(kernel_size=maxpool_size)
            )
        return conv_layer
   
    @staticmethod
    def bottleNeck (in_features, out_features, activation=nn.ReLU()):
        linear_layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm (out_features),
            activation
            )
        return linear_layer
        
    def Kaiming_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)

    def get_upsample_size (self, encoder):
        sample_input = self.sample_input
        b,c,t,h,w = sample_input.shape
        upsample_size = [(t-4, h-4, w-4)] # store sizes
        for i in range (len(encoder)):
            sample_input = encoder[i] (sample_input)
            size = tuple(sample_input.shape[2:])
            upsample_size.append (size)
        upsample_size = upsample_size[::-1]
        return upsample_size
