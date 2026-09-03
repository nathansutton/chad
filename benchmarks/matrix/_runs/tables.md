
### Harness x engine — same tasks, same weights, same laptop

| Arm                 | Passed | Tests   | Median wall (passed) | Total prefill | Total generated | Timeouts |
|---------------------|--------|---------|----------------------|---------------|-----------------|----------|
| pi+llama            | 7/8    | 128/148 | 388s                 | 41,631        | 42,322          | 3        |
| opencode+llama      | 3/8    | 82/148  | 446s                 | 176,499       | 42,857          | 5        |
| chad+llama          | 7/8    | 141/148 | 375s                 | 44,226        | 28,682          | 1        |
| chad+mlx *          | 8/8    | 148/148 | 589s                 | 35,998        | 49,265          | 1        |
| chad+mlx-nodflash * | 6/8    | 97/148  | 310s                 | 18,142        | 20,132          | 2        |
| dsh+llama           | 6/8    | 105/148 | 528s                 | 93,566        | 40,526          | 0        |
| goose+llama         | 6/8    | 92/148  | 1200s                | 184,138?      | 39,114?         | 4        |
| mini+llama          | 4/8    | 87/148  | 745s                 | 66,235?       | 42,368?         | 4        |
| crush+llama         | 5/8    | 88/148  | 580s                 | 178,677?      | 63,901?         | 3        |
| cline+llama         | 6/8    | 116/148 | 1073s                | 95,514        | 51,789          | 0        |
| codex+llama         | 6/8    | 97/148  | 379s                 | 88,613        | 36,522          | 2        |

`*` token counts are the harness's own, not llama-server's: the MLX arm is in-process
and no server sees it. Cached tokens are subtracted so both columns mean the same thing.
`?` the server was still busy when this arm's counters were read — treat the count as a floor.

Sampler, forced identically on every arm (proxy for the llama arms, `CHAD_*` for the
MLX arms, cross-checked): min_p 0.05, presence_penalty 0.0, repeat_penalty 1.0, temperature 1.0, top_k 20, top_p 0.95

### Per task (wall seconds if passed; otherwise tests passed, `T` = timed out)

| Task          | pi+llama | opencode+llama | chad+llama | chad+mlx | chad+mlx-nodflash | dsh+llama | goose+llama | mini+llama | crush+llama | cline+llama | codex+llama |
|---------------|----------|----------------|------------|----------|-------------------|-----------|-------------|------------|-------------|-------------|-------------|
| bowling       | 1200     | T 9/31         | 825        | 955      | T 0/31            | x 0/31    | x 0/31      | T 0/31     | T 0/31      | 1171        | T 0/31      |
| grade-school  | 141      | 408            | 111        | 70       | 97                | 289       | 395         | 398        | 349         | 339         | 247         |
| affine-cipher | 148      | 446            | 110        | 90       | 256               | 463       | 587         | 818        | 453         | 462         | 274         |
| transpose     | 1200     | T 0/12         | 736        | 589      | 401               | x 0/12    | 1200        | T 9/12     | T 8/12      | x 0/12      | 1021        |
| wordy         | 655      | T 24/25        | 375        | 124      | 316               | 1086      | x 0/25      | 725        | T 0/25      | 1073        | 645         |
| book-store    | T 0/20   | T 0/20         | T 13/20    | 852      | T 0/20            | 955       | 1200        | T 0/20     | 1124        | x 0/20      | T 0/20      |
| dominoes      | 326      | 596            | 284        | 297      | 300               | 390       | 1200        | T 6/13     | 580         | 1171        | 341         |
| go-counting   | 388      | T 0/11         | 419        | 1200     | 310               | 528       | 1200        | 745        | 591         | 549         | 379         |
