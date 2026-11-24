"""GPU-native version of mast3r.fast_nn.fast_reciprocal_NNs (dot-product descriptors, integer subsampling).

Same algorithm and same output as the official function (iterated nearest-neighbour hopping between the two descriptor maps until
reciprocal fix-points, max 10 iterations, unique correspondences sorted on the first image), but the whole loop stays on the GPU and
synchronises once at the end. The official implementation moves indices to the CPU in every block of every iteration (~700 small
synchronisations per call), which costs ~10 s per call when the GPU is time-sliced between many processes.
Checked against the official function with check_fastnn.py."""
import numpy as np, torch


@torch.inference_mode()
def fast_reciprocal_gpu(d1, d2, S=4, max_iter=10, blk=1024):
    """d1: (H1, W1, D), d2: (H2, W2, D) float tensors on the GPU -> xy1, xy2 integer arrays (N, 2) as (x, y)."""
    H1, W1, D = d1.shape; H2, W2, _ = d2.shape; p1 = d1.reshape(-1, D); p2 = d2.reshape(-1, D); dev = p1.device
    y1, x1 = torch.meshgrid(torch.arange(S // 2, H1, S, device=dev), torch.arange(S // 2, W1, S, device=dev), indexing='ij')
    xy1 = (x1 + W1 * y1).reshape(-1).sort().values; xy2 = torch.full_like(xy1, -1); old1 = xy1.clone(); old2 = xy2.clone(); notyet = torch.ones_like(xy1, dtype=torch.bool)
    nn = lambda q, db: torch.cat([(q[i:i + blk] @ db.T).argmax(1) for i in range(0, len(q), blk)])
    for it in range(max_iter):
        xy2 = torch.where(notyet, nn(p1[xy1], p2), xy2); notyet = notyet & (old2 != xy2)       # points whose match did not change have converged
        xy1 = torch.where(notyet, nn(p2[xy2], p1), xy1); notyet = notyet & (old1 != xy1)
        if it == max_iter - 1: break
        old2 = xy2.clone(); old1 = xy1.clone()
    conv = ~notyet; a = xy1[conv].cpu().numpy().astype(np.int64); b = xy2[conv].cpu().numpy().astype(np.int64)                # the only synchronisation
    key = np.unique(a * (H2 * W2) + b); a, b = key // (H2 * W2), key % (H2 * W2)                                              # unique pairs, sorted on xy1 then xy2
    return np.stack([a % W1, a // W1], 1), np.stack([b % W2, b // W2], 1)
