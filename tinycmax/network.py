import torch.nn as nn

from tinycmax.blocks import conv_encoder, upsample_decoder


class FlowNetwork(nn.Module):
    """
    Feedforward optical flow prediction network.
    Input: stacked event frames [B, 2*N, H, W]; output: flow map [B, 2, H, W].
    """

    mode = "flow"

    def __init__(
        self,
        encoder_channels,
        memory_channels,
        decoder_channels,
        activation_fn,
        final_bias,
        padding_mode,
        scaling,
    ):
        super().__init__()

        self.scaling = scaling

        self.encoder = conv_encoder(encoder_channels, activation_fn, padding_mode=padding_mode)
        self.memory = nn.Sequential(
            nn.Conv2d(memory_channels, memory_channels, kernel_size=3, padding=1, padding_mode=padding_mode),
            activation_fn(),
        )
        self.decoder = upsample_decoder(
            decoder_channels, activation_fn, final_bias, padding_mode=padding_mode, mode=self.mode
        )

    def forward(self, input):
        frame = input["events"]
        encoder = self.encoder(frame)
        memory = self.memory(encoder)
        flow_map = self.decoder(memory)
        flow_map *= self.scaling
        return dict(flow=flow_map)
