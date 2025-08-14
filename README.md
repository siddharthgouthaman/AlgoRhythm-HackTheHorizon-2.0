AlgoRhythm – Backend (Modal + ACE‑Step)

AI music generation service running on Modal with ACE‑Step (music), Qwen2‑7B (LLM prompts/lyrics), and SDXL‑Turbo (cover art). Outputs are stored in AWS S3.

✨ What this service does
	•	Generates music from:
	•	a plain description (LLM builds tags & lyrics),
	•	a prompt + described lyrics (LLM writes lyrics),
	•	a prompt + your own lyrics.
	•	Generates a cover image (album art) with SDXL‑Turbo.
	•	Uploads .wav and .png files to S3 and returns their keys (and simple categories).
