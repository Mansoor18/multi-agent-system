# Multi-Agent Customer Support System

A multi-agent AI system built with CrewAI that autonomously handles 
customer support queries through a pipeline of specialised agents.

## Architecture
Customer Query
      ↓
Classifier Agent → Categorises query (BILLING/TECHNICAL/SHIPPING/RETURNS/GENERAL)
      ↓
Research Agent → Identifies issues, policies, and solutions
      ↓
Response Agent → Crafts professional empathetic customer response

## Tech Stack
- CrewAI for multi-agent orchestration
- Groq API (LLaMA 3.1) as LLM backend
- LiteLLM for model provider abstraction
- Python-dotenv for environment management

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your Groq API key to `.env`:
```
GROQ_API_KEY=your_key_here
```

```bash
cd src
python main.py
```

## Sample Output
- Billing queries → Refund process and timeline
- Technical queries → Troubleshooting steps and escalation path
- All responses → Empathetic, actionable, professional

## Agents
- **Classifier Agent** — Triages and prioritises incoming queries
- **Research Agent** — Finds relevant policies and solutions
- **Response Agent** — Crafts the final customer communication