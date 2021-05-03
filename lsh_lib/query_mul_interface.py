import torch
import query_mul


class QueryMulFn(torch.autograd.Function):
    """C = A @ B where A and B are dense matrices, but the output C is sparse, specified by CSR format
    """
    @staticmethod
    def forward(ctx, A, B, rowPtrC, colIdxC):
        # Ensure that A and B are contiguous in the column-major format, to avoid copying twice in backward
        A_cont = A.detach().t().contiguous().t()  # Have to detach otherwise torch.autograd complains
        B_cont = B.detach().t().contiguous().t()
        ctx.save_for_backward(A_cont, B_cont, rowPtrC, colIdxC)
        valC = query_mul.constrained_gemm(A_cont, B_cont, rowPtrC, colIdxC)
        return valC

    @staticmethod
    def backward(ctx, grad):
        A_cont, B_cont, rowPtrC, colIdxC = ctx.saved_tensors
        grad_A, grad_B = None, None
        if ctx.needs_input_grad[0]:
            grad_A = query_mul.csrmm(grad, rowPtrC, colIdxC, B_cont, A_cont.shape[0], False, True)
        if ctx.needs_input_grad[1]:
            grad_B = query_mul.csrmm(grad, rowPtrC, colIdxC, A_cont, B_cont.shape[1], True, False).t()
        return grad_A, grad_B, None, None

query_mul_fn = QueryMulFn.apply
