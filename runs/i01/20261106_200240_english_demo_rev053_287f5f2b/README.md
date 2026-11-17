> Submission packaging: use the repository root README for the current launch and verification commands. The following description records the original historical revision.

# English Demonstration and Saved Replay

This is the English presentation revision of the 2026-11-04 classroom demo. The original launcher now forwards here. Start `Start Demo.command` in a terminal and provide the API key only at its hidden prompt, or use a terminal where DEEPSEEK_API_KEY is already configured. The application remains local at http://127.0.0.1:7862/.

The page labels, two case names, stage names, progress/status/error messages, teaching scenarios, evidence conditions/limitations and displayed answers are English. The installed Gradio frontend locale is set to English, including its built-in Loading message, even when the browser language is Chinese.

Live Demo keeps the existing two-call workflow: fresh frozen-classifier inference, task-identification API, unchanged graph/TF-IDF retrieval, answer API, then evidence-ID checks. The answer prompt explicitly requires English. Chinese output is retained in the response/generated-answer files and blocked from publication; it does not trigger a translation request or an automatic retry. Request caps, model, retry policy and canonical evidence cards are unchanged. Only manually initiated live runs call the provider.

Saved replay is read-only and makes zero new model calls. It prefers records in this revision, then existing 2026-11-04 records, then the original November 1 fallback. The four Chinese historical answers have separately stored English presentation translations bound to the exact original answer hashes. Translated replay is visibly labelled; original answers, evidence IDs, source records and evaluation scores are not rewritten. Conditions and limitations are translated for display only. The canonical evidence, query construction, training and E02 packages remain unchanged.

## Validation

- 17 local tests passed: original workflow/HTTP control cases, all eight saved records, two November 1 fallback records, four scenarios, 16 evidence cards, English-only failures and Chinese live-output blocking with raw retention and no extra calls.
- Actual browser checks with a Chinese browser locale: both cases, saved replay, expanded scenario/evidence, English Loading label, document language en, zero visible Chinese characters. See validation/browser_acceptance.json and validation/english_interface.png.
- Both launchers passed zsh -n. No real model requests, fresh classifier inference, retrieval, training, D04 test or scoring were performed during validation. Only synthetic providers and historical records were used.
- The staging server at port 7864 was stopped after validation. The normal port 7862 was not started. New live provider responses were not collected for this language revision.

Initial fallback citation-footer and frontend-locale URL issues were fixed; their records remain under validation/. Original known failures and all old raw records remain intact.

The original launcher is the only modified file inside the historical demo directory. Its original bytes and exact diff are in before/ and diffs/. Historical binding/manifest files were not rewritten; the old launcher pin remains a historical pin. The new binding identifies this presentation revision, and validation/compatibility.json accounts for the exact authorized launcher and main-plan changes.

See validation/final_verification.json for static package/protection checks. R01 remains in progress; this interface revision does not deliver the formal report, PPT or complete classroom presentation.
