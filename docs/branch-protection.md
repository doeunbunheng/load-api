# Branch protection for `main`

The capstone requires a green `test` check before a pull request can merge.

1. Open the repository on GitHub.
2. Go to **Settings → Branches → Add branch protection rule** (or **Rulesets**).
3. Branch name pattern: `main`.
4. Enable **Require a pull request before merging**.
5. Enable **Require status checks to pass before merging**.
6. Search for and select the check named **`test`** (workflow `CI`, job `test`).
   It appears in the list only after the workflow has run at least once.
7. Enable **Require branches to be up to date before merging**.
8. Enable **Do not allow bypassing the above settings**.
9. Save the rule and capture a screenshot for the evidence report.

To prove the rule works, open a pull request that changes `MAX_LOAN_AMOUNT_USD` in
`main.py` from `5000` to `500`. The `test` job fails on
`tests/test_contract.py::test_loan_amount_limit_is_5000` and the **Merge** button
stays disabled. Close the pull request without merging.
