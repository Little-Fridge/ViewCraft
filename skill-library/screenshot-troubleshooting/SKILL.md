---
name: screenshot-troubleshooting
description: Diagnose visible application failures from screenshots by reading exact errors, separating observed state from hypotheses and choosing a discriminating next check.
---

# Screenshot Diagnosis

1. **Read the blocking state.** Identify the application/page, intended action and foreground dialog. Crop the entire error with its title, exact code and nearby input labels. Also inspect the control the user was trying to use. An obscured background control may not describe the active state.
2. **Extract facts, then hypotheses.** Save `observed: exact message, selected field/state; unknown: ...` in the action's `note`. Classify the visible failure: rejected input, missing access, unavailable resource, connection failure or operation still running. A disabled button alone does not identify the cause.
3. **Prefer a visible repair when one is explicit.** If the message names a missing field or invalid value, point to that field and the required constraint. If a path or resource is missing, identify the exact resource to check. If access is denied, verify the selected account and that resource's access. Do not turn every network-looking error into advice to change DNS or reinstall.
4. **Choose one check that separates the remaining causes.** Write `check -> outcome A supports X; outcome B supports Y`. For example, if upload rejection could depend on file size, checking the displayed file size against the displayed limit can distinguish that cause from a format restriction. Use only constraints and controls visible in the screenshot or explicitly supplied by the user.
5. **Define observable success.** State what should appear after the next step, such as the dialog closing and the requested item appearing. A button click is not proof of success. With several screenshots, match the same account, object and workflow stage before interpreting the transition.

Return the visible problem, likely cause with its supporting clue, and one precise next action or request for missing evidence. If the screenshot cannot distinguish causes, say what exact error detail is needed. Do not invent menu paths, claim to have operated the app, or repeat visible credentials.

In ViewCraft, save the working record in the next action’s `note` and finish with `{"action":"answer","answer":"..."}`. Follow the runtime’s image-number and coordinate manifest. In another visual agent, use its inspection tools and return the same result directly.
