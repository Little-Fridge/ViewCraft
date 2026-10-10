---
name: screenshot-troubleshooting
description: Diagnose visible application failures from screenshots by reading exact errors, separating observed state from hypotheses and choosing a discriminating next check.
---

# Screenshot Diagnosis

## Use
Accept a screenshot and the intended action. If the goal is absent, infer it only when the visible error makes the workflow clear. Screenshots provide state at one moment, not a history of actions or permission to operate the application.

## Establish what is visible
Read the application/page, active tab, selected object, dialog title, error code and complete message. Crop small text with its associated label and control. Distinguish an error, an empty result, a disabled action and a pending operation. Inspect the relevant foreground dialog before interpreting obscured background controls.

## Build a causal shortlist
Separate observations from hypotheses. For each plausible cause, identify what visible evidence supports it and what is missing. A permission message may support an access issue; it does not prove that the server is down. A network-looking failure does not establish a specific proxy, DNS or certificate problem without further evidence.

## Choose the next check
Prefer the smallest check that separates the leading explanations. Name the exact visible control or requested log detail and describe what each outcome would imply. If the screenshot already exposes an invalid field, point to that field instead of giving a generic reinstall/reset sequence. Do not invent menu names, settings, versions or controls not shown.

For a sequence of screenshots, align the same workflow state and compare transitions. Treat a button's presence as availability, not proof that it was clicked or that the action succeeded. After any user-authorized follow-up action, use the new state to revise the diagnosis.

## Deliver
Give a concise diagnosis with three parts: what the screenshot establishes, the most likely explanation with uncertainty, and one actionable next check. Add alternatives only if they change the next step. Avoid repeating visible secrets or unnecessary personal information. If evidence is insufficient, ask for the specific missing detail rather than requesting an indiscriminate log dump.

## Execution
In ViewCraft, use the observation actions and source IDs supplied by the runtime. Finish with its `action=answer` object, with the complete requested output in the string `answer` field. In another visual agent, use its available inspection tools and return the same user-facing result directly.
