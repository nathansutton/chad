# The eight tasks

Eight Python exercises from [Exercism](https://exercism.org)'s
[python track](https://github.com/exercism/python), as packaged by the
[aider polyglot benchmark](https://github.com/Aider-AI/polyglot-benchmark): the pristine
stub, the test file, and the exercise's `instructions.md`. Nothing else — no `.meta/`
example solutions, no hints — so an arm sees exactly the three files a person opening the
exercise would.

| task | tests | why it is in the set |
|---|---|---|
| grade-school | 20 | the quickest; the smoke task |
| affine-cipher | 16 | string handling with an error path |
| transpose | 12 | the one with the fussy edge cases |
| wordy | 25 | a small parser |
| book-store | 20 | a search / optimisation problem |
| dominoes | 13 | a chaining problem |
| go-counting | 11 | tests that will not even collect against the stub |
| bowling | 31 | long test file, many error cases — the one most arms time out on |

The spread is deliberate: a set where every arm scores 0 measures nothing, and so does a
set where every arm scores 1.

`run.py` reads these out of git (`git show HEAD:…`), never from the working tree, so a
solution left behind in a stub can never leak into a later arm. Editing a task therefore
means committing it before it takes effect.

Exercise content is © Exercism, MIT licensed ([LICENSE](LICENSE)).
