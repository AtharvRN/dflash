"""Chunked stable entropy reduction, no probability tensor or CPU transfer."""
import triton
import triton.language as tl


@triton.jit
def entropy_partials(X, Y, stride: tl.constexpr, vocab: tl.constexpr,
                     chunks: tl.constexpr, K: tl.constexpr):
    row, chunk = tl.program_id(0), tl.program_id(1)
    col = chunk*K + tl.arange(0, K)
    x = tl.load(X + row*stride + col, col < vocab, other=-float('inf')).to(tl.float32)
    peak = tl.max(x, 0)
    e = tl.exp(x - peak)
    total = tl.sum(e, 0)
    weighted = tl.sum(e * tl.where(col < vocab, x, 0.), 0)
    base = (row*chunks + chunk)*3
    tl.store(Y + base, peak)
    tl.store(Y + base + 1, total)
    tl.store(Y + base + 2, weighted)
