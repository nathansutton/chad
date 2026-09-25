# Design

A laptop running a 27B model is short of two things: the time it takes to read a prompt,
and the room to hold one. chad is built around not wasting either. The agent owns the
inference engine, so nothing it has already read gets read again. The harness stays out of
the model's context: five tools, a shell the model already knows, and help that arrives
inside tool results instead of as more tools.

1.x did the opposite, and we spent a summer finding that out. What each module does is in
[architecture](architecture.md).

## Reading is the expensive part

An agent loop sends the whole conversation back on every step. On an M4 Pro this model
reads about 100 tokens a second, so a 9,000-token transcript is a minute and a half of
silence before the first new token, and a coding task is ten to forty steps.

So chad does not talk to an inference server. The loop owns the MLX process, keeps the KV
cache alive between steps, and diffs each new prompt against the token ids already in the
cache. Only the tail that changed gets computed. This is one real task, a C++ exercise,
read out of its trajectory. `ctx` is the prompt at that step and `new` is what the engine
had to read.

```
step    ctx    new  prefill  action
   3   2953    470    4.9 s  bash: ls -la && cat run-tests.sh
   4   3515    391    4.3 s  bash: cat all_your_base.h all_your_base.cpp CMakeLists.txt
   5   5551   1946   19.4 s  bash: ls -la test/ && head -30 test/tests-main.cpp
   6   6909    166    2.2 s  bash: sh run-tests.sh 2>&1 | tail -20
   7   7571    559    6.0 s  write_todos
   8   7899     75    1.2 s  write: all_your_base.h, all_your_base.cpp
   9   8491     75    1.2 s  bash: g++ -fsyntax-only ...
  10   8809     99    1.6 s  bash: sh run-tests.sh 2>&1 | tail -25
  11   9201    270    3.3 s  write_todos
  12   9421     75    1.2 s  done
```

The context tripled and the cost of a step did not move. The slow step is the one where a
tool printed 1,900 tokens the model had never seen, which is the only reading that should
cost anything. Over the task the model was shown 70,320 prompt tokens and the engine read
4,126 of them, in 45 seconds. Through a stateless server the same ten steps are about
twelve minutes of prefill.

The system prompt and tool schemas are also checkpointed to disk, so a new session in any
directory starts warm.

## The cache only appends

Qwen3.8-27B is a hybrid. 48 of its 64 layers are recurrent, and a recurrent layer's state
is a fixed-size summary of everything so far. That is why 128k tokens of cache cost 4.6 GB
rather than ~17. It is also why the cache cannot be rewound: an attention
layer can drop rows, and a recurrent state has no rows to drop.

Everything in the loop is arranged so the transcript only grows at the end. Nothing above
the conversation may change between steps, which rules out timestamps, fresh directory
listings and per-session ids in the prompt. Compaction rewrites the middle and costs a
full re-read, so it trims oldest first and takes enough in one pass that it does not come
back the next step. Speculative decoding has to undo rejected drafts, and gets away with
it because the recurrent layers reassign their state rather than mutate it, so a snapshot
is a pointer and a rollback is free.

## One model

The engine is fitted to one checkpoint. The drafter reads this model's hidden states at
five specific layers. The attention kernel is written for this model's head shape over an
8-bit cache. The context budget is computed from this model's bytes per token. Aim the
same engine at other weights and it still runs, slower, with each of those missing.

A model menu is the first thing most local harnesses show you. We would rather the first
thing be the task. `--model` will load other weights and `--backend llama` will talk to a
llama.cpp server. They are there so the fit can be measured, not as a menu.

## The weights

We shipped the wrong weights first. 2.x defaulted to a ternary build of this model: 7.2 GB,
as fast as a 3-bit quant, and room for ~150k of context on 24 GB. The tasks we tested on
could not tell the two apart, so we took the window. Harder tasks could. On 36 paired
polyglot tasks the ternary passed 24 and Unsloth's 3-bit GGUF of the same model passed 34,
ten tasks up and none down. All twelve ternary failures ran out the 20-minute wall clock,
and it generated 3.3× the tokens getting there. The window was not what ran out: the median trial peaked under 20k
on either.

The obvious fix failed next. Converting Unsloth's file into MLX's own format, the same
bits per weight in the same places, passed 6 of 9 trials where the file passes 9. The bit map was
never the good part; the quantizer was. Unsloth fits i-quants and K-quants against an
importance matrix, three quarters of the file's bytes are i-quants, and an i-quant is a
codebook that MLX's `scale * q + bias` format cannot express. Re-quantizing throws away
exactly what made the file worth having.

Serving the GGUF from llama.cpp keeps the quant and loses the engine-owned cache this page
opens with. So chad reads the GGUF itself. The llama.cpp blocks load byte for byte, and a
port of llama.cpp's dequantizers to Metal decodes them inside chad's own kernels, bit-exact
for all 15 block formats; the drafter, the attention kernel and the append-only cache run
unchanged on top. It is an odd thing for an MLX project to do, and the only way we found
to keep both the quant and the loop.

The cost is 6 GB of weights: ~74k of context on 24 GB instead of the ternary's ~238k.
`UD-IQ3_XXS` is one `--model` away at 10.9 GB and ~138k, passing 8 of the same 9. Every
run cited here is published; see [`RUNS.md`](../benchmarks/polyglot/RUNS.md).

## Five tools

1.13.0 had 21 tools: `read`, `grep`, `glob`, line-addressed editors, a tree-sitter repo
map, and an LSP layer for symbols, references and renames. Each one was added after
watching the agent fail a benchmark task in a way the tool would have prevented. So were
the 56 switches around them. A score would tick up, the change would stay, and the prompt
grew to 5,406 tokens before the user had typed anything.

That was benchmaxxing, and it did not survive a control. Two runs of the same
configuration differed from each other by about as much as our changes differed from the
baseline. We had been fitting noise, one plausible tool at a time.

The trajectories said the same thing more plainly. These are the tool calls from 355
tasks on 1.13.0 with all 21 tools offered (per-trial counts:
[`history/toolmix-1.13.0.jsonl`](https://huggingface.co/datasets/nathansutton/chad-polyglot-runs/blob/main/history/toolmix-1.13.0.jsonl)):

```
18,431 tool calls
  bash                                13,230   71.8%
    of which cat / head / sed -n       1,474
    of which grep / rg / find          1,237
  read                                 1,453    7.9%
  write                                1,103    6.0%
  edit                                 1,097    6.0%
  grep + glob                            414    2.2%
  line-addressed editors                 179    1.0%
  ten symbol and repo-map tools          104    0.6%
```

`read` earned its place and took half the file reads. `grep` and `glob` took a quarter of
the searches. The line editors lost to plain `edit` six to one. The LSP layer and the repo
map, the most engineering in the project, were touched in 42 of 355 tasks. The model
reaches for the shell it saw a billion times in pretraining. A dialect it has to be taught
in the prompt mostly stays untaught, and the teaching is paid for in context on every
task whether it is used or not.

2.x has `bash`, `edit`, `write`, `write_todos` and `done`. The schemas are 829 tokens
instead of 3,499. `read` went too, because `cat` and `sed -n` were already doing half its
job, and the model did not miss it.

We tested the idea once more from the other side. Ranked search is the one thing a shell
cannot do, so we built a BM25 index and offered it as a sixth tool. The index was good:
the right file was in the top 20 for 26 of 26 questions. With and without it the agent
solved the same 6 of 6 tasks and found the answer just as fast, and the tool output in
context grew 28%, because the model ran the search and then ran its usual greps anyway. A
tool the model half-adopts is only a cost. It was deleted.

## Help goes in the results

If the model is going to live in `bash`, the place to help it is in what `bash` returns.
chad does ten things there. None of them tells the model what to do. Each one states
something true that the raw output left out.

Three keep anything from being thrown away. A single enormous line, like a minified
bundle, is clipped so it cannot eat the whole output budget. A result over the size cap is
clipped. Old results are trimmed when the context window fills. In all three the full
text is written to a file first and the notice names the path, so the model can `grep`
for what was dropped instead of running the command again.

Four make failures explain themselves. A command that prints nothing says why: the `sed`
range was past the end of the file, or this stage of the pipeline matched nothing. A
failed `edit` says whether the change is already applied, or shows the first line and
column where the text it sent differs from the file. A test run that had to be trimmed
keeps the rows naming the tests that failed. A failing test run is shown beside how the
same command went before the first edit, so "was this already broken?" does not need
another run.

Two answer questions the model would otherwise spend a step asking. The first time a
source file comes back through `cat` or `sed`, a one-line map of its symbols comes with
it. And the prompt ends with the compilers, interpreters and package managers on this
machine, with versions, and the common ones that are missing.

The last is a trap. `rg -rn` is not recursive grep. It replaces every match with the
letter `n`, and the output looks plausible enough to act on. chad says so.

Each of the ten can be switched off by name with `CHAD_DISABLE`, which is how they get
tested: one at a time, against the loop without it.

## What is next

Prefill used to be the wait. Now it is thinking. Across twelve tasks in six languages the
model was shown a million prompt tokens and the engine read 67,000 of them: 13 minutes of
reading against 84 minutes of generating, and 82% of what it generated was inside
`<think>`. Ten of the twelve passed. The two that failed were the same puzzle, alphametics,
in Go and in JavaScript, and neither was a wrong answer. The model stated the right
algorithm in a few hundred tokens and then worked test cases by hand inside one `<think>`
block, 21,000 tokens of it, until the clock ran out. The drafter helps least exactly there,
because reasoning is the text it predicts worst. The run, trajectories included, is
`design-sample` in [`RUNS.md`](../benchmarks/polyglot/RUNS.md).

The 1.x reflex would be a nudge, a budget and a switch. The 2.x rule is that a change
ships when a paired run on the laptop says it helped, and the eval for that is
`benchmarks/polyglot`.

## Borrowed

Rescue parsing for malformed tool calls, argument repair and the loop guard come from
[forge](https://github.com/antoinezambelli/forge). The keep-going and verify-by-running
prompt lines come from [opencode](https://github.com/anomalyco/opencode). The prompt
skeleton is [OpenHarness](https://github.com/HKUDS/OpenHarness), minus its advice to
prefer dedicated tools over `bash`. `write_todos`, and refusing `done` until something has
been run, come from [deepagents](https://github.com/langchain-ai/deepagents). The drafter
is z-lab's DFlash2.
