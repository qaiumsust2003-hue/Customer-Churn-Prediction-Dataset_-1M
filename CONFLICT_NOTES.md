# Merge Conflict Notes

## What happened
The same line of `README.md` (the **Research question** line) was edited differently on two branches:

- **`main`** shortened it to a short-form question (commit `1003c27`).
- **`proposal`** reworded it to match the proposal and point to `proposal.md` (commit `73a10ba`).

When Pull Request #3 (`proposal` into `main`) was opened, GitHub reported it as not mergeable (`mergeable_state: dirty`). Merging `main` into `proposal` locally reproduced the conflict:

```
<<<<<<< HEAD
**Research question:** on 1,000,000 customers, does gradient boosting beat logistic regression on PR-AUC, and which signals drive churn risk? (details in `proposal.md`)
=======
**Research question (short form):** can gradient boosting identify likely churners better than logistic regression?
>>>>>>> main
```

## How I resolved it
1. Ran `git merge main` on the `proposal` branch, which stopped with `CONFLICT (content): Merge conflict in README.md`.
2. Opened `README.md` and compared the two versions. The `main` version was a short question with no metric, while the `proposal` version named the dataset size and the metric but was less explicit about the evaluation setting.
3. Kept the intent of both: the dataset size and the PR-AUC metric on a held-out test set from the proposal side, and the plain "does gradient boosting rank churners better than logistic regression" phrasing from the main side. I added pointers to `proposal.md` and `methodology.md` and deleted all conflict markers.
4. Checked that no `<<<<<<<`, `=======` or `>>>>>>>` markers remained, staged the file with `git add README.md`, and committed the merge.
5. Pushed `proposal`, after which GitHub showed the PR as mergeable, and merged PR #3 into `main`.

## Final resolved line
```
**Research question:** on 1,000,000 customers, does gradient boosting rank churners better than logistic regression (PR-AUC on a held-out test set), and which signals drive churn risk? See `proposal.md` for the proposal and `methodology.md` for the full plan.
```

## Lesson
Pulling or merging `main` into the feature branch before opening the PR would have surfaced the conflict earlier. Keeping one agreed wording per README line, or editing different lines, avoids this kind of conflict.
