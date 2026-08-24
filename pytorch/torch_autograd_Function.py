import torch
from torch.autograd import Function

class MyReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Use it via .apply, not by calling forward directly
x = torch.randn(5, requires_grad=True)
y = MyReLU.apply(x)
y.sum().backward()
print(x.grad)