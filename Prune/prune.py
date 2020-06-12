import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    # Single Pruning a Module
    model = LeNet().to(device=device)
    module = model.conv1
    print(list(module.named_parameters()))
    print("=" * 50)
    print(list(module.named_buffers()))
    prune.random_unstructured(module, name="weight", amount=0.3)
    print("=" * 50)
    print(list(module.named_parameters()))
    print("=" * 50)
    print(list(module.named_buffers()))
    print("=" * 50)
    print(module.weight)
    print("=" * 50)
    print(module._forward_pre_hooks)

    prune.l1_unstructured(module, name="bias", amount=3)
    print("=" * 50)
    print(list(module.named_parameters()))
    print("=" * 50)
    print(list(module.named_buffers()))
    print("=" * 50)
    print(module.bias)
    print("=" * 50)
    print(module._forward_pre_hooks)

    # Iterative Pruning
    # prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)

    # As we can verify, this will zero out all the connections corresponding to
    # 50% (3 out of 6) of the channels, while preserving the action of the
    # previous mask.
    print("=" * 50)
    print(module.weight)
    for hook in module._forward_pre_hooks.values():
        if hook._tensor_name == "weight":  # select out the correct hook
            break
    print("=" * 50)
    print(list(hook))  # pruning history in the container
    print("=" * 50)
    print(model.state_dict().keys())

    # Remove pruningre parametrization
    print("=" * 50)
    print(list(module.named_parameters()))
    print("=" * 50)
    print(list(module.named_buffers()))
    print("=" * 50)
    print(module.weight)
    prune.remove(module, 'weight')
    print("=" * 50)
    print(list(module.named_parameters()))
    print("=" * 50)
    print(list(module.named_buffers()))

    # Pruning multiple parameters in a model
    new_model = LeNet()
    for name, module in new_model.named_modules():
        # prune 20% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.2)
        # prune 40% of connections in all linear layers
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.4)
    print("=" * 50)
    print(dict(new_model.named_buffers()).keys())

