import math
import random
import unittest

import bigvsan_pq as bpq


class BigVSanPqTest(unittest.TestCase):
    def test_snake_pq_matches_float_snake(self) -> None:
        alphas = [0.75, 1.25, 2.5, 4.0]
        snake_pq = bpq.SnakePq(in_features=4, alphas=alphas, alpha_logscale=False)
        x_seq = [-2.0 + 0.05 * i for i in range(80)]

        for ch, a in enumerate(alphas):
            out_pq = snake_pq.forward_channel(x_seq, channel_idx=ch)
            out_ref = [x + (1.0 / (a + 1e-9)) * (math.sin(a * x) ** 2) for x in x_seq]
            max_diff = max(abs(u - v) for u, v in zip(out_pq, out_ref))
            self.assertLess(max_diff, 2e-3, f"SnakePq max_diff={max_diff} for alpha={a}")

    def test_snake_beta_pq_matches_float_snake_beta(self) -> None:
        alphas = [1.0, 2.0]
        betas = [0.5, 1.5]
        sb_pq = bpq.SnakeBetaPq(in_features=2, alphas=alphas, betas=betas)
        x_seq = [-1.5 + 0.1 * i for i in range(30)]
        for ch, (a, b) in enumerate(zip(alphas, betas)):
            out_pq = sb_pq.forward_channel(x_seq, channel_idx=ch)
            out_ref = [x + (1.0 / (b + 1e-9)) * (math.sin(a * x) ** 2) for x in x_seq]
            max_diff = max(abs(u - v) for u, v in zip(out_pq, out_ref))
            self.assertLess(max_diff, 3e-3)

    def test_quantum_san_popcount_spherical_norm_and_bfp64_projection(self) -> None:
        rng = random.Random(2024)
        D = 128
        out_dim = 16
        signs = [[rng.choice((-1, 0, 1)) for _ in range(D)] for _ in range(out_dim)]
        scales = [0.5 + 0.1 * j for j in range(out_dim)]

        san_pq = bpq.QuantumSANLinearPq(signs, scales)
        x = [rng.uniform(-1.0, 1.0) for _ in range(D)]
        out = san_pq.forward(x, flg_train=False)
        assert isinstance(out, list)

        # Reference float SANLinear: y_j = scale_j * <x, w_j / ||w_j||_2>
        ref_out = []
        for row_idx, (w_row, s) in enumerate(zip(signs, scales)):
            l2_sq = sum(w * w for w in w_row)
            self.assertEqual(san_pq.rows[row_idx].nnz_popcount, l2_sq)
            norm = math.sqrt(max(1, l2_sq))
            dot = sum(xi * (wi / norm) for xi, wi in zip(x, w_row))
            ref_out.append(dot * s)

        norm_ref = math.sqrt(sum(v * v for v in ref_out))
        diff_norm = math.sqrt(sum((a - b) ** 2 for a, b in zip(out, ref_out)))
        rel_err = diff_norm / max(1e-12, norm_ref)
        self.assertLess(rel_err, 0.01)


if __name__ == "__main__":
    unittest.main()
