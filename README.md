# Medical Clinic — Appointment Assistance System

A full-stack AI-powered appointment chatbot built with **LangGraph**, **GPT-4o-mini**, and **Flask**. Patients can book, reschedule, or cancel appointments through a web interface. The system collects patient details, drafts a professional email using GPT-4o-mini, and presents it for Human-in-the-Loop (HITL) review before submission.

---

# Project Structure


PROJECT_4/
├── .env              # contains required keys
├── .gitignore
├── frontend.html     # User Facing Web UI
├── server.py         
├── main.py           # Main Code
└── README.md


---

## Setup

### 1. Create project folder

set up folders in environment


### 2. Install dependencies

pip install python-dotenv flask flask-cors langgraph langchain langchain-openai openai


### 3. Add your OpenAI API key
Create a `.env` file in the project root:

OPENAI_API_KEY=sk-proj-your-key-here


### 4. Start the server

python server.py


### 5. Open in browser

http://localhost:8080


---

## Purpose

### Web App

The patient interacts with a chat widget on the web page. The frontend collects information step by step and makes two API calls to the Flask backend.


### Conversation Flow


Patient opens page
    → Greeted with 4 options: Book / Reschedule / Cancel / Emergency
    → Selects intent
    → Bot collects: Full Name → Phone Number → Preferred Day → Preferred Time
    → GPT-4o-mini drafts a professional email
    → Patient reviews the draft  ← Human-in-the-Loop step
    → Patient confirms YES → request submitted
    → Patient confirms NO  → starts over


### Emergency Detection

At any point in the conversation, if the patient types words like *chest pain*, *stroke*, *can't breathe*, *seizure*, etc., the bot immediately displays a **🚨 Call 911** message and halts the normal flow.

---

## LangGraph Pipeline

The backend processes every message through a LangGraph workflow:


## Middleware Components

| Middleware | What it does |
|---|---|
| `OpenAIModerationMiddleware` | Scans messages for harmful content, escalates if flagged |
| `PIIMiddleware` | Detects SSN, DOB, phone, email, MRN — logs field types only, never values |
| `ContextEditingMiddleware` | Trims messages over 2000 chars, detects urgency signals |
| `ToolCallLimitMiddleware` | Hard cap of 15 LLM calls per run |
| `HumanInTheLoopMiddleware` | Flags cancel/reschedule intents for staff review |
| `ModelFallbackMiddleware` | Catches GPT API failures and serves a safe static response |

---

## (`main.py`)

For testing without the browser:


# Interactive conversation loop
python main.py


## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install <module>` inside activated `.venv` |
| `localhost:5000` access denied | Mac AirPlay blocks port 5000 — use port 8080 |
| `AuthenticationError` | Check `.env` has the correct full API key |
| `No module named dotenv` | Run `pip install python-dotenv` |
| Frontend not connecting to backend | Confirm `server.py` is running on port 8080 |

---
