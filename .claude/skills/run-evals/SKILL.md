---
name: run-evals
description: Run the evaluation suite and compare it with the last saved result. Use after changing chunking, retrieval, prompts, or the model.
---

1. Run the offline retrieval evaluation first; it costs nothing.
2. Run the answer evaluation only if the user approved paid calls and `OPENAI_API_KEY` is set.
3. Save results to `evals/results/` with the date, git commit, and model.
4. Compare with the previous result and report which metrics moved and which questions changed outcome.
5. Never edit the question set to improve a score. Propose changes separately.
