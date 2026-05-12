from __future__ import annotations
 
import os
from dataclasses import dataclass
from dotenv import load_dotenv
 
from vault.cli.logger import done, fail, log, step, warn

load_dotenv()

@dataclass
class LLMResponse:
    answer: str
    model: str
    input_tokens:  int = 0
    output_tokens: int = 0


# Gemini Backend
class GeminiBackend:
    MODEL = "gemini-2.5-flash-lite"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            fail(
                "LLM",
                "GEMINI_API_KEY not set. "
                "Get a free key at https://aistudio.google.com/app/apikey\n"
                "  Then run: set GEMINI_API_KEY=your_key_here",
            )
            raise SystemExit(1)

    def ask(self, prompt: str) -> LLMResponse:
        try:
            from google import genai
        except ImportError:
            fail("LLM", "google-generativeai not installed. Run: pip install -U google-genai")
            raise SystemExit(1)
        
        step("LLM", f"Calling [bold]{self.MODEL}[/bold] (Gemini free tier)")
        log("Only the retrieved note chunks are sent — not your full vault")
        log(f"Approximate prompt size: {len(prompt.split())} words")
 
        client = genai.Client(api_key=self.api_key)

        try:
            response = client.models.generate_content(
                model=self.MODEL,
                contents=prompt,
            )

            answer = response.text.strip()
            done("LLM", "Response received")

            return LLMResponse(answer=answer, model=self.MODEL)
        except Exception as exc:
            fail("LLM", f"Gemini API error: {exc}")
            raise


# Prompt builder
def build_rag_prompt(query: str, chunks) -> str:
    step("PROMPT", "Building grounded RAG prompt")
    log("Injecting retrieved chunks as [context] — model must cite sources")
    log("Instruction: answer ONLY from provided notes, never from general knowledge")

    context_parts = []
    seen_titles = set()
    for chunk in chunks:
        if chunk.note_title not in seen_titles:
            seen_titles.add(chunk.note_title)
        context_parts.append(
            f"--- [{chunk.note_title}] (relevance: {chunk.score:.2f}) ---\n{chunk.text}"
        )

    context = "\n\n".join(context_parts)
    source_list = ", ".join(f"[{t}]" for t in seen_titles)

    log(f"Context sources: {source_list}")
    done("PROMPT", f"Prompt built from {len(chunks)} chunk(s) across {len(seen_titles)} note(s)")

    return f"""You are a personal knowledge assistant. Your job is to answer questions \
based ONLY on the notes provided below.

Rules:
1. Answer using ONLY information from the provided notes.
2. For every claim you make, cite the source note in [Note Title] format.
3. If the notes do not contain enough information to answer, say:
   "I couldn't find a clear answer in your notes. The closest relevant notes are: {source_list}"
4. Do not use any outside knowledge. Do not hallucinate.
5. Be concise but complete. Use bullet points where appropriate.

--- YOUR NOTES (retrieved context) ---

{context}

--- END OF NOTES ---

Question: {query}

Answer (cite [Note Title] for every claim):"""