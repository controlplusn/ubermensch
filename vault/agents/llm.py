from __future__ import annotations
 
import os
import requests
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


# Ollama backend
class OllamaBackend:
    """
    Prerequisites:
        1. Install: https://ollama.ai
        2. Pull: ollama pull llama3
        3. Serve: ollama serve (auto-starts on most systems)
    """

    DEFAULT_MODEL = "llama3"
    DEFAULT_HOST  = "http://localhost:11434"

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", self.DEFAULT_MODEL)
        self.host  = host  or os.environ.get("OLLAMA_HOST",  self.DEFAULT_HOST)

    def ask(self, prompt: str) -> LLMResponse:
        try:
            import requests
        except ImportError:
            fail("LLM", "requests not installed. Run: pip install requests")
            raise SystemExit(1)
 
        step("LLM", f"Calling [bold]{self.model}[/bold] (Ollama local)")
        log(f"Host: {self.host}  — fully local, no data leaves your machine")
        log(f"Prompt size: ~{len(prompt.split())} words")

        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model":  self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data   = resp.json()
            answer = data.get("response", "").strip()
 
            in_tok  = data.get("prompt_eval_count", 0)
            out_tok = data.get("eval_count", 0)
 
            done("LLM", f"Response received  ({in_tok} in / {out_tok} out tokens)")
            return LLMResponse(
                answer=answer,
                model=self.model,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
 
        except requests.exceptions.ConnectionError:
            fail(
                "LLM",
                f"Cannot connect to Ollama at {self.host}\n"
                "  Make sure Ollama is running: ollama serve\n"
                "  Or install it from: https://ollama.ai",
            )
            raise
        except requests.exceptions.Timeout:
            fail("LLM", f"Ollama timed out — model '{self.model}' may be too large for your machine")
            raise
        except Exception as exc:
            fail("LLM", f"Ollama error: {exc}")
            raise
    
    def raw(self, prompt: str) -> str:
        return self.ask(prompt).answer
    
    def list_models(self) -> list[str]:
        try:
            
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []
        
    def is_available(self) -> bool:
        try:
            requests.get(f"{self.host}/api/tags", timeout=3)
            return True
        except Exception:
            return False


# Factory
def get_backend(
    backend_name: str = "gemini",
    api_key: str | None = None,
    model: str | None = None,
) -> GeminiBackend | OllamaBackend:
    name = backend_name.lower().strip()

    if name == "ollama":
        step("LLM", "Initializing Ollama local backend")

        backend = OllamaBackend(model=model)

        if not backend.is_available():
            warn("LLM", "Ollama server not detected at localhost:11434")
            log("Start it with: ollama serve")
            log("Install from:  https://ollama.ai")
        else:
            models = backend.list_models()
            log(f"Ollama running  |  Available models: {', '.join(models) or 'none pulled yet'}")

            if model and model not in models:
                warn("LLM", f"Model '{model}' not pulled yet")
                log(f"Pull it with: ollama pull {model}")
            done("LLM", f"Ollama ready  |  Model: {backend.model}")

        return backend
    
    elif name == "gemini":
        return GeminiBackend(api_key=api_key, model=model)
    
    else:
        warn("LLM", f"Unknown backend '{backend_name}' — defaulting to Gemini")
        return GeminiBackend(api_key=api_key)
    


# Prompt builder
def build_rag_prompt(query: str, chunks) -> str:
    step("PROMPT", "Building grounded RAG prompt")
    log("Injecting retrieved chunks as [context] — model must cite sources")
    log("Instruction: answer ONLY from provided notes, never from general knowledge")

    context_parts = []
    seen_titles = []
    
    for chunk in chunks:
        if chunk.note_title not in seen_titles:
            seen_titles.append(chunk.note_title)
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