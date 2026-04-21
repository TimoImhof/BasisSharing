import torch
import pytest

"""
SVD-based weight sharing across layers (math convention).

Paper: W(i) in R^(d1 x d2), horizontally concatenated -> W in R^(d1 x nd2)
       W ≈ Wk = Uk @ Sk @ Vhk
       Wk = B @ Vhk  where B = Uk @ Sk  (basis matrix)
       W(i)[:,j] ≈ sum_m( B[:,m] * C(i)[m,j] )

PyTorch convention: layer.weight stored as (H_out, H_in) = (d2, d1) in math terms.
                    Must transpose before applying paper's math.

Shapes summary:
    W(i) math:          (d1, d2)    = (d_in, d_out)
    W(i) pytorch:       (d_out, d_in) -> .T gives (d_in, d_out) = math convention
    W_concat:           (d1, n*d2)  = (d_in, n*d_out)
    U:                  (d1, d1)    = (d_in, d_in)
    S:                  (d1,)       = (d_in,)           <- 1D vector, min(d1, n*d2)
    Vh:                 (d1, n*d2)  = (d_in, n*d_out)
    Uk:                 (d1, k)     = (d_in, k)
    Sk:                 (k, k)
    Vhk = C:            (k, n*d2)   = (k, n*d_out)
    B = Uk @ Sk:        (d1, k)     = (d_in, k)
    C(i):               (k, d2)     = (k, d_out)        <- per-layer slice of C
    B[:,m]:             (d1,)       = (d_in,)            <- single basis vector
    C(i)[m,j]:          scalar                           <- coefficient
    W(i) reconstructed: (d1, d2)    = (d_in, d_out)
"""

d_out = 10  # H_out — PyTorch's first dim, paper's d2
d_in = 8  # H_in  — PyTorch's second dim, paper's d1
n = 2  # number of layers
k = 4  # rank for truncation


@pytest.fixture
def pytorch_weights():
    """Simulate real PyTorch layer weights: shape (d_out, d_in)."""
    torch.manual_seed(42)
    return [torch.randn(d_out, d_in) for _ in range(n)]


def test_svd_decomposition(pytorch_weights):
    """
    Test the full SVD decomposition pipeline from PyTorch weights to
    shared basis B and per-layer coefficients C(i).
    """
    W1, W2 = pytorch_weights

    # --- Step 1: Convert to math convention ---
    # PyTorch stores (d_out, d_in), paper wants (d1, d2) = (d_in, d_out)
    W1_math = W1.T  # (d_in, d_out)
    W2_math = W2.T  # (d_in, d_out)
    assert W1_math.shape == (d_in, d_out)

    # --- Step 2: Horizontal concatenation -> W_concat ---
    # Paper: W in R^(d1 x nd2)
    W_concat = torch.cat([W1_math, W2_math], dim=1)
    assert W_concat.shape == (d_in, n * d_out)  # (8, 20)

    # --- Step 3: Full SVD ---
    # S returned as 1D vector of length min(d1, n*d2), NOT a 2D diagonal matrix
    U, S, Vh = torch.linalg.svd(W_concat, full_matrices=False)
    assert U.shape == (d_in, min(d_in, n * d_out))  # (8, 8)
    assert S.shape == (min(d_in, n * d_out),)  # (8,)  <- 1D!
    assert Vh.shape == (min(d_in, n * d_out), n * d_out)  # (8, 20)

    # --- Step 4: Truncate to rank k ---
    Uk = U[:, :k]  # (d_in, k)  = (8, 4)
    Sk = torch.diag(S[:k])  # (k, k)     = (4, 4)  <- explicit diagonal matrix
    Vhk = Vh[:k, :]  # (k, n*d2)  = (4, 20)
    assert Uk.shape == (d_in, k)  # (8, 4)
    assert Sk.shape == (k, k)  # (4, 4)
    assert Vhk.shape == (k, n * d_out)  # (4, 20)

    # --- Step 5: Basis matrix B and coefficient matrix C ---
    # Paper: Wk = B @ Vhk, where B = Uk @ Sk
    # B columns are the shared basis vectors across all layers
    B = Uk @ Sk
    C = Vhk  # full coefficient matrix, all layers stacked
    assert B.shape == (d_in, k)  # (8, 4)  — shared across layers
    assert C.shape == (k, n * d_out)  # (4, 20) — all layers

    # --- Step 6: Per-layer coefficient matrices ---
    # C was built from horizontally concatenated W(i), so split column-wise
    C1 = C[:, 0 * d_out : 1 * d_out]  # (k, d_out) = (4, 10)
    C2 = C[:, 1 * d_out : 2 * d_out]  # (k, d_out) = (4, 10)
    assert C1.shape == (k, d_out)
    assert C2.shape == (k, d_out)

    # --- Step 7: Reconstruct full layers ---
    # W(i) ≈ B @ C(i):  (d_in, k) @ (k, d_out) = (d_in, d_out)
    W1_approx = B @ C1
    W2_approx = B @ C2
    assert W1_approx.shape == (d_in, d_out)  # (8, 10)
    assert W2_approx.shape == (d_in, d_out)  # (8, 10)

    # --- Step 8: Single column reconstruction (equation 1 from paper) ---
    # W(i)[:,j] ≈ sum_m( B[:,m] * C(i)[m,j] )
    j = 3  # arbitrary column index
    col_from_matmul = W1_approx[:, j]  # (d_in,) = (8,)
    col_from_sum = sum(
        B[:, m] * C1[m, j]  # basis vector * scalar coefficient
        for m in range(k)
    )
    col_from_mat_vec_mul = B @ C1[:, j]  # Should be the same as col_from_matmul
    assert col_from_matmul.shape == (d_in,)
    assert torch.allclose(col_from_matmul, col_from_sum, atol=1e-5)
    assert torch.allclose(col_from_matmul, col_from_mat_vec_mul, atol=1e-5)

    # --- Step 9: Full rank sanity check (k == d_in -> lossless) ---
    Uk_full = U
    Sk_full = torch.diag(S)
    Vhk_full = Vh
    W_reconstructed = Uk_full @ Sk_full @ Vhk_full
    assert torch.allclose(W_reconstructed, W_concat, atol=1e-5)


@pytest.fixture
def XtX():
    """Simulate a well-conditioned X.T @ X (symmetric positive semi-definite)."""
    torch.manual_seed(0)
    X = torch.randn(256, d_in)
    return X.T @ X  # (d_in, d_in)


def test_spectral_decomposition_equivalent_to_cholesky(XtX):
    """
    Paper computes S via Cholesky:   S @ S.T = XtX
    Code computes S via eigendecomp: XtX = Q @ Λ @ Q.T  ->  S = Q @ Λ^(1/2)

    S @ S.T = (Q @ Λ^(1/2)) @ (Q @ Λ^(1/2)).T
        = Q @ Λ^(1/2) @ Λ^(1/2) @ Q.T
        = Q @ Λ @ Q.T
        = XtX  ✓

    Both produce a valid square root of XtX, i.e. S @ S.T = XtX.
    We verify both satisfy this property, even though S itself differs.

    Why eigendecomp: Cholesky requires strict positive definiteness. With small
    sample sizes XtX may have near-zero eigenvalues, causing Cholesky to fail.
    eigh + clamp handles this gracefully.
    """
    # Cholesky-based S (paper)
    L = torch.linalg.cholesky(XtX)  # L @ L.T = XtX, shape: (d_in, d_in)
    S_chol = L
    assert torch.allclose(S_chol @ S_chol.T, XtX, atol=1e-4), (
        "Cholesky S must satisfy S @ S.T == XtX"
    )

    # Spectral-based S (code)
    eivals, eivecs = torch.linalg.eigh(XtX)  # XtX = Q @ Λ @ Q.T
    eivals = eivals.clamp(min=1e-8)  # numerical safety
    S_spec = eivecs @ torch.diag(torch.sqrt(eivals))  # S = Q @ Λ^(1/2)
    assert S_spec.shape == (d_in, d_in)
    assert torch.allclose(S_spec @ S_spec.T, XtX, atol=1e-4), (
        "Spectral S must satisfy S @ S.T == XtX"
    )


def test_spectral_decomposition_handles_near_singular(XtX):
    """
    Cholesky fails on near-singular XtX (rank-deficient, as happens with
    small sample sizes or correlated features). Spectral decomposition survives
    because we clamp eigenvalues before taking sqrt.
    """
    # Make XtX near-singular by zeroing out small eigenvalues
    eivals, eivecs = torch.linalg.eigh(XtX)
    eivals_singular = eivals.clone()
    eivals_singular[:3] = 0.0  # force rank deficiency
    XtX_singular = eivecs @ torch.diag(eivals_singular) @ eivecs.T

    # Cholesky fails
    with pytest.raises(Exception):
        torch.linalg.cholesky(XtX_singular)

    # Spectral succeeds with clamping
    eivals_c, eivecs_c = torch.linalg.eigh(XtX_singular)
    eivals_c = eivals_c.clamp(min=1e-8)
    S = eivecs_c @ torch.diag(torch.sqrt(eivals_c))
    assert not torch.any(torch.isnan(S)), "S must not contain NaN after clamping"
    assert S.shape == (d_in, d_in)
