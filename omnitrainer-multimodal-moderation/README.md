# OmniTrainer: Multimodal Customer Service Trainer

AI-powered moderation and training app for ACME customer service interactions.

The project includes:

- Pydantic structured moderation outputs for text, image, audio, and video.
- Pydantic AI moderation agents and a simulated LLM customer.
- FastAPI moderation endpoints.
- Gradio multimodal chat interface.
- Phoenix/OpenTelemetry tracing setup.
- Pydantic Evals suites for moderation behavior.

## Setup

```bash
uv sync --dev
uv pip install -e .
cp env.example .env
```

Fill `.env` with local credentials:

```text
USER_API_KEY=my-api-key
GEMINI_API_KEY=your-vocareum-or-google-gemini-key
GOOGLE_GEMINI_BASE_URL=https://gemini.vocareum.com
DEFAULT_GOOGLE_MODEL=gemini-2.5-flash-lite
```

For local OpenAI-compatible providers, set `MODEL_PROVIDER` and the matching API key as shown in `env.example`.

## Test

```bash
uv run pytest tests/ -vv
```

## Run

```bash
uv run multimodal-moderation
```

Then open:

- Gradio app: http://localhost:7860/
- FastAPI docs: http://localhost:8000/docs
- Phoenix traces: http://localhost:6006/projects

## Evals

```bash
uv run evals/text/test_cases.py
uv run evals/image/test_cases.py
uv run evals/audio/test_cases.py
uv run evals/video/test_cases.py
```
