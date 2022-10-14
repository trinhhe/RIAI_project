import argparse
import torch
import torch.nn as nn
import cfg
import networks
import deeppoly
import util
import verify

DEVICE = cfg.DEVICE
INPUT_SIZE = cfg.INPUT_SIZE


def analyze(net: networks.FullyConnected, inputs, eps, true_label):
    inputs.requires_grad_(False)
    deep = verify.create_verifier(net, true_label, eps)
    return verify.train_and_verify(deep, inputs, iterations=10000000)


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net',
                        type=str,
                        required=False,
                        help='Neural network architecture which is supposed to be verified.',
                        default='net0_fc1')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Print debug output')
    parser.add_argument('--spec', type=str, required=False, help='Test case to verify.',
                        default="../test_cases/net0_fc1/example_img0_0.01800.txt")
    args = parser.parse_args()

    cfg.DEBUG = args.debug
    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    net = util.load_net(args.net)
    # inputs.shape [1,1,28,28]
    inputs = torch.Tensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    inputs.requires_grad_(False)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label
    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
