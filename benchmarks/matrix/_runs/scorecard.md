### Local-fitness scorecard — same weights, same laptop, same tasks

#### What it feels like

| Arm                 | tax: turn-1 prompt (tok) | wait before 1st token, turn 1 | uncached tok / later turn (med) | wait / later turn (med · p90) | cache reuse | prefill s / task (med) | exp. tok/s | pass (gate) |
|---------------------|--------------------------|-------------------------------|---------------------------------|-------------------------------|-------------|------------------------|------------|-------------|
| pi+llama            | 2,008                    | 23.3 s                        | 77                              | 1.7 s · 23 s                  | 99%         | 47                     | 8.0        | 7/8 T3      |
| opencode+llama      | 18,057                   | 237.6 s                       | 113                             | 2.5 s · 44 s                  | 99%         | 400                    | 5.9        | 3/8 T5      |
| chad+llama          | 2,562                    | 28.2 s                        | 46                              | 1.0 s · 14 s                  | 99%         | 54                     | 6.9        | 7/8 T1      |
| dsh+llama           | 8,052                    | 96.6 s                        | 100                             | 2.0 s · 34 s                  | 99%         | 141                    | 6.6        | 6/8         |
| goose+llama         | 9,576                    | 119.1 s                       | 3,009                           | 39.4 s · 88 s                 | 76%         | 347                    | 5.0        | 6/8 T4      |
| mini+llama          | 1,171                    | 12.8 s                        | 204                             | 3.0 s · 19 s                  | 96%         | 56                     | 7.9        | 4/8 T4      |
| crush+llama         | 16,263                   | 207.7 s                       | 96                              | 1.9 s · 41 s                  | 100%        | 304                    | 5.5        | 5/8 T3      |
| cline+llama         | 5,876                    | 66.9 s                        | 400                             | 5.6 s · 56 s                  | 96%         | 144                    | 7.1        | 6/8         |
| codex+llama         | 7,804                    | 90.2 s                        | 852                             | 11.6 s · 28 s                 | 91%         | 121                    | 6.3        | 6/8 T2      |
| chad+mlx *          | 2,562                    | 1.2 s †                       | 49                              | 1.1 s · 20 s                  | 99%         | 25                     | 15.6       | 8/8 T1      |
| chad+mlx-nodflash * | 2,566                    | 1.2 s †                       | 35                              | 1.1 s · 14 s                  | 99%         | 25                     | 11.1       | 6/8 T2      |

Every column but the last two is llama-server's own accounting, read through the proxy
(`_runs/turns.jsonl`), never a harness's self-report. **tax** = prompt tokens of the
first agent request (`prompt_n + cache_n`: system prompt + tool schemas + task); **wait,
turn 1** = the server's `prompt_ms` for that request; **uncached / later turn** =
`prompt_n` on agent turns 2+, pooled median; **wait / later turn** = `prompt_ms` on those
turns, median and p90; **cache reuse** = `cache_n / (cache_n + prompt_n)` on those turns;
**prefill s / task** = Σ `prompt_ms` over every request of a task, side requests included;
**exp. tok/s** = generated tokens / wall clock. Side requests (title / summary calls with
no tool schemas) are excluded from the per-turn columns and counted in the next table.
The pass column is a gate, not a ranking.
`*` in-process arm: the same fields from chad's own prefill trace, self-reported — no server saw it. `†` its turn-1 wait is a system-prompt prefix restored from disk, not a cold prefill; the chad+llama row is the cold number for the same prompt.

#### Shape of the harness

| Arm                 | tools | system prompt (chars) | prefix churn | side requests (concurrent · abandoned) | round trips / task | model busy | prefill share | ctx at exit |
|---------------------|-------|-----------------------|--------------|----------------------------------------|--------------------|------------|---------------|-------------|
| pi+llama            | 4     | 4,052                 | 0/29         | 0                                      | 5                  | 98%        | 14%           | 6,606       |
| opencode+llama      | 10    | 49,096                | 0/17         | 10 (8 · 2)                             | 4                  | 118%       | 46%           | 22,543      |
| chad+llama          | –     | –                     | –            | 0                                      | 6                  | 97%        | 15%           | 8,790       |
| dsh+llama           | 25    | 4,188                 | 0/34         | 8 (8 · 8)                              | 6                  | 99%        | 20%           | 16,828      |
| goose+llama         | 18    | 23,829                | 0/29         | 8 (8 · 0)                              | 6                  | 102%       | 50%           | 14,812      |
| mini+llama          | 1     | 62                    | 0/43         | 0                                      | 6                  | 71%        | 10%           | 7,360       |
| crush+llama         | 26    | 37,456                | 0/30         | 16 (16 · 0)                            | 6                  | 112%       | 49%           | 22,055      |
| cline+llama         | 26    | 4,326                 | 0/28         | 0                                      | 6                  | 98%        | 26%           | 13,165      |
| codex+llama         | 10    | 20,751                | 0/19         | 0                                      | 4                  | 98%        | 41%           | 11,760      |
| chad+mlx *          | –     | –                     | –            | 0                                      | 6                  | 82%        | 20%           | 7,870       |
| chad+mlx-nodflash * | –     | –                     | –            | 0                                      | 6                  | 89%        | 17%           | 8,538       |

**tools** / **system prompt** as the harness sent them on its first agent request;
**prefix churn** = agent turns whose system-message or tool-list hash changed since the
previous agent turn, over turns compared (`–`: chad's raw `/completion` path and the MLX
arms carry no messages to hash — their cache-reuse column is the evidence instead);
**side requests** = no-tools calls beside the agent loop (session titles, summaries),
summed over the arm's grid: how many overlapped an agent turn in time, and how many the
harness abandoned before the server answered (no `timings` came back; counted, not
measured); **model busy** = Σ (`prompt_ms` + `predicted_ms`) /
wall over measured requests — above 100% means two were in flight at once; **round
trips** = agent turns per task; **ctx at exit** = tokens in context at the last agent
turn.

Instrument check: the proxy's own first-byte stamp came back earlier than the server's `prompt_ms` on 113 of 379 requests (llama-server streams a first chunk before a long prefill finishes), so no time-to-first-byte column is printed; the server's prefill time is the wait.

#### Per task — cache reuse (median, agent turns 2+) · wait / later turn (median) · experienced tok/s

| Task          | pi+llama         | opencode+llama        | chad+llama           | dsh+llama             | goose+llama          | mini+llama            | crush+llama            | cline+llama       | codex+llama           | chad+mlx           | chad+mlx-nodflash    |
|---------------|------------------|-----------------------|----------------------|-----------------------|----------------------|-----------------------|------------------------|-------------------|-----------------------|--------------------|----------------------|
| bowling       | 99% · 3.6s · 8.5 | 78% · 69.2s · 6.6 (T) | 100% · 0.9s · 7.8    | 99% · 2.5s · 7.6 (x)  | 97% · 4.8s · 6.8 (x) | 60% · 9.1s · 0.2 (T)  | 95% · 12.9s · 19.9 (T) | 95% · 8.2s · 7.8  | 91% · 10.0s · 8.1 (T) | 99% · 1.9s · 16.1  | – · – · 0.1 (T)      |
| grade-school  | 98% · 1.4s · 6.3 | 100% · 1.0s · 2.6     | 99% · 0.7s · 5.8     | 99% · 1.8s · 4.2      | 75% · 39.4s · 3.0    | 96% · 3.5s · 7.7      | 100% · 1.8s · 2.4      | 97% · 3.5s · 5.6  | 99% · 2.7s · 4.2      | 99% · 0.9s · 7.5   | 99% · 1.0s · 7.5     |
| affine-cipher | 99% · 0.8s · 6.2 | 100% · 0.9s · 3.4     | 99% · 0.8s · 4.6     | 99% · 1.7s · 5.5      | 74% · 43.7s · 3.6    | 99% · 2.0s · 8.3      | 100% · 1.8s · 3.2      | 94% · 9.3s · 6.9  | 91% · 12.5s · 4.5     | 90% · 4.9s · 13.7  | 98% · 2.3s · 10.3    |
| transpose     | 94% · 7.5s · 8.0 | – · – · 6.2 (T)       | 100% · 0.8s · 8.0    | 82% · 22.6s · 7.8 (x) | 93% · 10.3s · 7.0    | 72% · 5.7s · 3.4 (T)  | 100% · 1.8s · 14.6 (T) | – · – · 8.1 (x)   | 71% · 53.5s · 7.0     | 80% · 10.7s · 15.1 | 99% · 1.1s · 12.4    |
| wordy         | 97% · 2.3s · 8.3 | 99% · 2.2s · 5.7 (T)  | 91% · 4.9s · 7.4     | 95% · 6.2s · 7.3      | 96% · 5.6s · 6.7 (x) | 94% · 7.6s · 8.1      | 99% · 2.3s · 6.6 (T)   | 97% · 3.7s · 6.7  | 94% · 7.8s · 6.7      | 99% · 0.9s · 16.4  | 100% · 1.1s · 12.6   |
| book-store    | – · – · 8.2 (T)  | 93% · 20.6s · 6.2 (T) | 91% · 7.0s · 6.5 (T) | 99% · 2.2s · 6.7      | 82% · 34.0s · 5.0    | 58% · 13.9s · 5.0 (T) | 100% · 1.8s · 6.0      | – · – · 7.5 (x)   | 90% · 11.6s · 7.5 (T) | 100% · 0.7s · 16.9 | 79% · 7.4s · 0.2 (T) |
| dominoes      | 98% · 2.2s · 7.1 | 100% · 1.7s · 4.6     | 98% · 2.5s · 5.0     | 99% · 2.2s · 5.2      | 62% · 73.2s · 4.4    | 84% · 3.6s · 8.6 (T)  | 100% · 1.8s · 4.6      | 98% · 4.9s · 7.4  | 91% · 14.1s · 5.3     | 99% · 1.1s · 16.0  | 100% · 1.0s · 12.7   |
| go-counting   | 99% · 0.8s · 8.1 | 90% · 27.8s · 6.6 (T) | 98% · 1.9s · 7.7     | 99% · 1.6s · 6.6      | 67% · 57.3s · 4.9    | 98% · 3.0s · 8.0      | 99% · 2.2s · 5.0       | 84% · 17.2s · 6.3 | 90% · 12.9s · 6.0     | 61% · 17.3s · 1.8  | 99% · 1.0s · 11.9    |
`(x)` failed tests, `(T)` timed out — the numbers still describe the turns that happened.
