---
name: document-comparison
description: Compare document versions by matching sections and records, then report substantive text, numeric and table changes with paired source references.
---

# Document Revision Review

1. **Establish OLD and NEW.** Preserve the supplied file order and comparison scope. Make an index of relevant clause IDs, section headings or product codes in both versions. Pair by these keys; page numbers and line breaks often change during reflow.
2. **Open matched blocks together.** Use `view` for the corresponding pages, then `zoom` each clause or table row with its heading. If one old section was split, read all its new parts before calling text deleted. Save `key | old file:page | new file:page | checked/pending` in the action's `note`.
3. **Compare one field at a time.** Check names, obligations, negation, quantities, dates and exceptions. Include qualifiers such as “at least”, “unless”, “per item” and “excluding”. For numbers, read the unit, denominator, currency and period in both versions. Unchanged digits under a changed header can still represent a substantive change.
4. **Match table rows by identity.** Use the item code plus section when codes repeat. Check headers independently from values. Record `key | old value + qualifier | new value + qualifier | paired sources`. Treat moved content as a relocation, and line-wrap/font changes as formatting unless those changes are requested.
5. **Check every absence claim.** Search the destination section, adjacent continuation pages and index before declaring deletion. Do the reverse for additions. If only excerpts are supplied, label the item “not found in supplied pages”. Reconcile changed summaries against detailed tables; report inconsistency instead of selecting the convenient version.

Return a change table with field/clause, old content, new content, factual difference and both source locations. Keep uncertain matches separate. Explain the observed wording or value change; do not manufacture legal, financial or operational consequences.

In ViewCraft, save the working record in the next action’s `note` and finish with `{"action":"answer","answer":"..."}`. Follow the runtime’s image-number and coordinate manifest. In another visual agent, use its inspection tools and return the same result directly.
