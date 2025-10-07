import argparse
import pdb
import torch
import torch.nn as nn
import torchvision
import coremltools as ct
from models import *
import timm
import os

def parse():
    parser = argparse.ArgumentParser(description='Toolbox')
    parser.add_argument('--model', metavar='ARCH', default='CARETrans_S0')
    parser.add_argument('--ckpt', default=None, type=str, metavar='PATH',
                        help='path to checkpoint')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='profiling GMACs')
    parser.add_argument("--resolution", default=224, type=int)
    parser.add_argument('--onnx', action='store_true', default=False,
                        help='export onnx')
    parser.add_argument('--coreml', action='store_true', default=False,
                        help='export coreml')
    args = parser.parse_args()
    return args


class ProfileConv(nn.Module):
    def __init__(self, model):
        super(ProfileConv, self).__init__()
        self.model = model
        self.hooks = []
        self.macs = []
        self.params = []

        def hook_conv(module, input, output):
            self.macs.append(output.size(1) * output.size(2) * output.size(3) *
                             module.weight.size(-1) * module.weight.size(-1) * input[0].size(1) / module.groups)
            self.params.append(module.weight.size(0) * module.weight.size(1) *
                               module.weight.size(2) * module.weight.size(3) + module.weight.size(1))

        def hook_linear(module, input, output):
            if len(input[0].size()) > 2: self.macs.append(module.weight.size(0) * module.weight.size(1) * input[0].size(-2))
            else: self.macs.append(module.weight.size(0) * module.weight.size(1))
            self.params.append(module.weight.size(0) * module.weight.size(1) + module.bias.size(0))

        def hook_gelu(module, input, output):
            if len(output[0].size()) > 3: self.macs.append(output.size(1) * output.size(2) * output.size(3))
            else: self.macs.append(output.size(1) * output.size(2))

        def hook_layernorm(module, input, output):
            self.macs.append(2 * input[0].size(1) * input[0].size(2))
            self.params.append(module.weight.size(0) + module.bias.size(0))

        def hook_avgpool(module, input, output): self.macs.append(output.size(1) * output.size(2) * output.size(3) * module.kernel_size * module.kernel_size)
        def hook_attention(module, input, output): self.macs.append(module.key_dim * module.N * module.N2 * module.num_heads + module.dh * module.N * module.N2)

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d): self.hooks.append(module.register_forward_hook(hook_conv))
            elif isinstance(module, nn.Linear): self.hooks.append(module.register_forward_hook(hook_linear))
            elif isinstance(module, nn.GELU): self.hooks.append(module.register_forward_hook(hook_gelu))
            elif isinstance(module, nn.LayerNorm): self.hooks.append(module.register_forward_hook(hook_layernorm))
            elif isinstance(module, nn.AvgPool2d): self.hooks.append(module.register_forward_hook(hook_avgpool))

    def forward(self, x):
        self.model.to(x.device)
        _ = self.model(x)
        for handle in self.hooks: handle.remove()
        return self.macs, self.params

if __name__ == '__main__':
    args = parse()
    model_name = eval(args.model)
    model = model_name(resolution=args.resolution)
    try:
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['model'])
        print('load success, model is initialized with pretrained checkpoint')
    except:
        print('model initialized without pretrained checkpoint')

    model.eval()
    dummy_input = torch.randn(1, 3, args.resolution, args.resolution)

    if args.profile:
        profile = ProfileConv(model)
        MACs, params = profile(dummy_input)
        print('number of tracked layers (conv, fc, gelu, ...):', len(MACs))
        print(sum(MACs) / 1e9, 'GMACs')
        print(sum(params) / 1e6, 'M parameters')

    if args.onnx:
        torch.onnx.export(model, dummy_input, args.model + ".onnx", verbose=False)
        print('successfully export onnx')

    if args.coreml:
        example_input = dummy_input
        traced_model = torch.jit.trace(model, example_input)
        out = traced_model(example_input)

        model = ct.convert(
                        traced_model,
                        inputs=[ct.ImageType(name="input_1", shape=example_input.shape, channel_first=True)]
        )
        model.save(os.path.join('./coreml', args.model + ".mlmodel"))
        print('successfully export coreML')