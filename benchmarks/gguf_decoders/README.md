# GGUF decoder benchmarks

The serial decode step of the shipped `UD-Q3_K_XL` model is ~90% GGUF GEMVs. The file's
bytes are IQ4_XS 38%, IQ3_S 27%, Q5_K 9% (the lm_head), IQ3_XXS 7%, Q3_K 7% (mostly the
embedding), Q4_K 5%. `gemv_bandwidth.py` measures what each format's M=1 GEMV streams on
the 27B shapes; the Q8_0 and affine rows are the ceiling of the kernel body and the chip,
so the gap under them is the decoder's cost.

Measured 2026-10-01, M4 Pro 24 GB, mlx 0.32.2, before and after the aligned word loads:

| format | share | GB/s before | GB/s after |
|---|---|---|---|
| IQ4_XS | 37.8% | 152–156 | 167–181 (pure chain 167 → 199) |
| Q4_K | 4.9% | 138–143 | 160–166 (153 → 179) |
| Q5_K | 8.9% | 110–114 | 129–139 (119 → 143) |
| IQ3_S | 26.9% | 134–146 | unchanged (2-byte-aligned rows; an unaligned vector load is slower than the bytes) |
| Q8_0 | reference | 208–213 | |
| MLX affine 4-bit | reference | 213–221 | |

In the real step (`decode_ab.py`, one load, greedy text identical):

| | old | new | |
|---|---|---|---|
| serial decode, 512-token prompt | 11.8 tok/s | 12.6 tok/s | +6.7% |
| drafted decode (DFlash2 on) | 48.1 tok/s | 49.6 tok/s | +3.0% |

`load_width_race.py` keeps the byte-load decoders inline and must print `bit-identical
True` for every row: the word loads read the same bytes in the same order.
