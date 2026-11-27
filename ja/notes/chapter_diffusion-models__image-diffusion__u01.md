# Notes on chapter_diffusion-models/image-diffusion.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27 15:57 UTC)

- 4b6d1696: "1.2M-parameter" is rendered as 「120 万パラメータ」 per `ja/style.md` §4; the number check's regex splits "1.2M" at `.2M` and matches "1" alone.
- 7d6a41f2: "1.2 million" and "35.7 million" are converted to 「120 万」 and 「3570 万」 per `ja/style.md` §4 (万/億 for English number words without commas), so the literal substrings `1.2` and `35.7` do not appear in Japanese.
