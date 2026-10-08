---
name: form-reading
description: Read a requested field from a business form or invoice by pairing its label with its value while distinguishing neighboring fields.
---

# Form reading

This ViewCraft skill extracts one requested business form or invoice field. Use the document's visible labels, section structure, and alignment to establish which value belongs to the requested field. Return the value in the format requested by the user.

## Establish the field

Read the question for the exact field name and any qualifier, such as billing versus shipping, invoice versus order, or issue date versus due date. Find the matching label in the page overview before committing to a candidate value. If the document uses an equivalent label, check its section and role rather than matching a shared word alone. A reference number printed in a payment note may serve a different purpose from an order reference in the document header.

## Preserve the pairing

Inspect the label and value as one visual unit. Follow the form's local convention: values may appear below their labels, across the same row, or inside separate table columns. Use cell borders, column alignment, whitespace, and section headings to resolve the association. The closest text by distance is not always the correct value, especially beside narrow columns or wrapped labels.

Compare the candidate with the immediately adjacent fields. Check that the chosen string belongs to the requested label rather than an invoice identifier, account identifier, or similarly named reference. Keep prefixes, leading zeros, punctuation, and capitalization exactly as printed. An identifier's plausible shape is supporting evidence, not a substitute for a visible label association.

## Inspect only when needed

If both the label and its value are readable in the overview, answer directly. When small text or a cell boundary remains unclear, request a crop that includes the complete label, the complete candidate value, and enough neighboring structure to establish their pairing. Prefer a contextual row or field block over a crop of the value alone. Keep the overview available for section context. Avoid repeated crops that add no native detail and avoid decorative transforms that can obscure thin characters or separators.

## Finish

Before answering, reread the chosen field against the question and check ambiguous characters in the source. Do not calculate an identifier, borrow a nearby value, or infer missing text from expected business conventions. If the source cannot resolve the field, state the uncertainty briefly instead of inventing a value. End with action=answer and place only the requested value in the final answer field unless the user asks for explanation. Further observations are unnecessary once the association and transcription are clear.
