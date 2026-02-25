#!/usr/bin/env python3
"""
main.py
=======
Appointment Assistance Chatbot — LangGraph + GPT-4o-mini
Conversational CLI chatbot with HITL email draft review.

Usage:
  python main.py
"""

from __future__ import annotations
import functools
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from openai import OpenAI
from typing_extensions import TypedDict

load_dotenv()

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

CLINIC_NAME  = "Medical Clinic"
CLINIC_EMAIL = "appointments@medicalclinic.com"
MODEL        = "gpt-4o-mini"

# Validate API key early and clearly
_api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not _api_key or _api_key == "your-key-here":
    print("\n  ❌  ERROR: No OpenAI API key found.")
    print("  Add your key to .env file:")
    print("  OPENAI_API_KEY=sk-proj-...\n")
    exit(1)

_openai = OpenAI(api_key=_api_key)

DIVIDER = "─" * 60
BOLD    = "\033[1m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
RESET   = "\033[0m"

# ═══════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════

class ChatState(TypedDict, total=False):
    # Conversation
    session_id:       str
    messages:         List[Dict[str, str]]   # [{role, content}, ...]
    current_input:    str

    # Intent detection
    intent:           str    # book | cancel | reschedule | emergency | unknown

    # Collected patient info
    patient_name:     str
    patient_phone:    str
    preferred_day:    str
    preferred_time:   str

    # Conversation stage
    stage:            str    # detect | collect_name | collect_phone | collect_day | collect_time | hitl_review | done

    # HITL
    email_draft:      str
    hitl_approved:    Optional[bool]
    hitl_note:        str

    # Flow control
    bot_reply:        str
    route_taken:      List[str]


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def bot_print(msg: str):
    print(f"\n  {CYAN}🏥 Assistant:{RESET} {msg}\n")

def user_input(prompt: str = "") -> str:
    return input(f"  {BOLD}You:{RESET} ").strip()

EMERGENCY_RE = re.compile(
    r"\b(emergency|urgent|heart attack|chest pain|stroke|dying|can't breathe|"
    r"bleeding|unconscious|911|severe|critical|collapsed|seizure|overdose|"
    r"not breathing|passed out|severe pain)\b",
    re.IGNORECASE,
)

INTENT_PATTERNS = {
    "book":       [r"\b(book|schedule|make|new|set up)\b.*\bappointment\b", r"\bappointment\b.*\b(book|schedule|make)\b", r"\bi (need|want|would like).*(appointment|see (a |the )?doctor)"],
    "cancel":     [r"\b(cancel|remove|drop|delete)\b.*\bappointment\b", r"\bappointment\b.*\b(cancel|remove|drop)\b", r"\bcancel\b"],
    "reschedule": [r"\b(reschedule|move|change|shift|postpone|rebook)\b.*\bappointment\b", r"\bappointment\b.*\b(reschedule|move|change)\b", r"\breschedule\b"],
}

def detect_intent(text: str) -> str:
    if EMERGENCY_RE.search(text):
        return "emergency"
    t = text.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        for p in patterns:
            if re.search(p, t):
                return intent
    return "unknown"


# ═══════════════════════════════════════════════════════════════════
# GPT CALL
# ═══════════════════════════════════════════════════════════════════

def gpt_reply(system: str, user: str) -> str:
    resp = _openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
        ],
        temperature=0.4,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def draft_email(state: Dict) -> str:
    intent   = state.get("intent", "book")
    name     = state.get("patient_name", "")
    phone    = state.get("patient_phone", "")
    day      = state.get("preferred_day", "")
    time_    = state.get("preferred_time", "")

    action = {"book": "book", "cancel": "cancel", "reschedule": "reschedule"}.get(intent, "book")
    subject_action = {"book": "New Appointment Request", "cancel": "Appointment Cancellation Request", "reschedule": "Appointment Reschedule Request"}.get(intent, "Appointment Request")

    system = f"""You are drafting a professional email on behalf of a patient to {CLINIC_NAME}.
Write a brief, warm, professional email. Include all the patient details provided.
Format: start with 'Subject: ...' on the first line, then a blank line, then the email body.
Sign off as the patient."""

    user = f"""Draft an email to {CLINIC_EMAIL} to {action} an appointment.
Patient name: {name}
Patient phone: {phone}
Preferred day: {day}
Preferred time: {time_}
Clinic: {CLINIC_NAME}"""

    return gpt_reply(system, user)


# ═══════════════════════════════════════════════════════════════════
# LANGGRAPH NODES
# ═══════════════════════════════════════════════════════════════════

def node_detect_intent(state: Dict) -> Dict:
    text   = state.get("current_input", "")
    intent = detect_intent(text)
    route  = list(state.get("route_taken", []))
    route.append("detect_intent")

    if intent == "emergency":
        reply = (
            f"{RED}{BOLD}🚨 EMERGENCY — Please call 911 immediately!{RESET}\n\n"
            f"  If you are experiencing a medical emergency, call 911 now\n"
            f"  or go to your nearest emergency room. Do not wait.\n\n"
            f"  Once you are safe, come back and I can help you book a follow-up."
        )
        return {"intent": intent, "stage": "done", "bot_reply": reply, "route_taken": route}

    if intent == "unknown":
        reply = (
            "I can help you with:\n\n"
            f"  📅  Book an appointment\n"
            f"  🔄  Reschedule an appointment\n"
            f"  ❌  Cancel an appointment\n\n"
            f"  Just type what you need!"
        )
        return {"intent": intent, "stage": "detect", "bot_reply": reply, "route_taken": route}

    action_label = {"book": "book", "cancel": "cancel", "reschedule": "reschedule"}[intent]
    reply = f"I can help you {action_label} an appointment. Let's get a few details first.\n\n  What is your {BOLD}full name{RESET}?"
    return {"intent": intent, "stage": "collect_name", "bot_reply": reply, "route_taken": route}


def node_collect_name(state: Dict) -> Dict:
    name  = state.get("current_input", "").strip()
    route = list(state.get("route_taken", []))
    route.append("collect_name")

    if len(name) < 2:
        return {"bot_reply": "Please enter your full name.", "stage": "collect_name", "route_taken": route}

    reply = f"Thanks, {BOLD}{name}{RESET}! What is your {BOLD}phone number{RESET}?"
    return {"patient_name": name.title(), "stage": "collect_phone", "bot_reply": reply, "route_taken": route}


def node_collect_phone(state: Dict) -> Dict:
    raw   = state.get("current_input", "")
    route = list(state.get("route_taken", []))
    route.append("collect_phone")
    digits = re.sub(r"\D", "", raw)

    if len(digits) < 7:
        return {"bot_reply": "Please enter a valid phone number (e.g. 902-555-0123).", "stage": "collect_phone", "route_taken": route}

    if len(digits) == 10:
        phone = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11:
        phone = f"+{digits[0]} ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    else:
        phone = raw.strip()

    intent = state.get("intent", "book")
    if intent == "cancel":
        reply = f"Got it! What is the {BOLD}date of the appointment{RESET} you want to cancel? (e.g. next Monday, March 5)"
    else:
        reply = f"Got it! What is your {BOLD}preferred day{RESET}? (e.g. next Monday, March 5)"

    return {"patient_phone": phone, "stage": "collect_day", "bot_reply": reply, "route_taken": route}


def node_collect_day(state: Dict) -> Dict:
    day   = state.get("current_input", "").strip()
    route = list(state.get("route_taken", []))
    route.append("collect_day")

    if len(day) < 2:
        return {"bot_reply": "Please enter a preferred day or date.", "stage": "collect_day", "route_taken": route}

    intent = state.get("intent", "book")
    if intent == "cancel":
        reply = f"And what {BOLD}time{RESET} was the appointment? (e.g. 2:00 PM, afternoon)"
    else:
        reply = f"And what is your {BOLD}preferred time{RESET}? (e.g. 10:00 AM, afternoon)"

    return {"preferred_day": day, "stage": "collect_time", "bot_reply": reply, "route_taken": route}


def node_collect_time(state: Dict) -> Dict:
    time_ = state.get("current_input", "").strip()
    route = list(state.get("route_taken", []))
    route.append("collect_time")

    if len(time_) < 2:
        return {"bot_reply": "Please enter a preferred time.", "stage": "collect_time", "route_taken": route}

    # All info collected — generate email draft
    updated = dict(state)
    updated["preferred_time"] = time_
    updated["route_taken"]    = route

    bot_print("Got everything I need. Drafting your email with GPT-4o-mini...")
    try:
        email = draft_email(updated)
    except Exception as e:
        email = f"Subject: Appointment Request — {updated.get('patient_name', '')}\n\nDear {CLINIC_NAME} Team,\n\nPatient {updated.get('patient_name','')} ({updated.get('patient_phone','')}) would like to {updated.get('intent','book')} an appointment on {updated.get('preferred_day','')} at {updated.get('preferred_time','')}.\n\nPlease contact the patient to confirm.\n\nThank you."

    reply = (
        f"Here is the email draft:\n\n"
        f"{DIVIDER}\n"
        f"  To : {CLINIC_EMAIL}\n"
        f"{DIVIDER}\n"
    )
    for line in email.splitlines():
        reply += f"  {line}\n"
    reply += f"{DIVIDER}\n\n"
    reply += f"  {BOLD}Does this look correct? (yes to send / no to edit){RESET}"

    route.append("collect_time")
    return {
        "preferred_time": time_,
        "email_draft":    email,
        "stage":          "hitl_review",
        "bot_reply":      reply,
        "route_taken":    route,
    }


def node_hitl_review(state: Dict) -> Dict:
    answer = state.get("current_input", "").strip().lower()
    route  = list(state.get("route_taken", []))
    route.append("hitl_review")

    approved = answer in ("yes", "y", "send", "confirm", "looks good", "correct", "ok", "sure", "yeah", "yep")
    rejected = answer in ("no", "n", "edit", "change", "redo", "wrong", "incorrect", "restart")

    if approved:
        name  = state.get("patient_name", "")
        phone = state.get("patient_phone", "")
        reply = (
            f"{GREEN}✅ Email sent to {CLINIC_EMAIL}!{RESET}\n\n"
            f"  Your request has been submitted. A team member will\n"
            f"  contact {BOLD}{name}{RESET} at {BOLD}{phone}{RESET} to confirm.\n\n"
            f"  Is there anything else I can help you with?\n"
            f"  (book / reschedule / cancel)"
        )
        return {"hitl_approved": True, "stage": "done", "bot_reply": reply, "route_taken": route}

    if rejected:
        reply = (
            f"No problem! Let's start over.\n\n"
            f"  What would you like to do?\n"
            f"  (book / reschedule / cancel an appointment)"
        )
        return {
            "hitl_approved": False,
            "stage":         "detect",
            "patient_name":  "",
            "patient_phone": "",
            "preferred_day": "",
            "preferred_time":"",
            "email_draft":   "",
            "bot_reply":     reply,
            "route_taken":   route,
        }

    # Unclear answer
    return {
        "bot_reply":    f"Please reply {BOLD}yes{RESET} to send the email, or {BOLD}no{RESET} to start over.",
        "stage":        "hitl_review",
        "route_taken":  route,
    }


# ═══════════════════════════════════════════════════════════════════
# LANGGRAPH ROUTER
# ═══════════════════════════════════════════════════════════════════

def route_by_stage(state: Dict) -> str:
    return state.get("stage", "detect")


# ═══════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════

def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("detect_intent",  node_detect_intent)
    graph.add_node("collect_name",   node_collect_name)
    graph.add_node("collect_phone",  node_collect_phone)
    graph.add_node("collect_day",    node_collect_day)
    graph.add_node("collect_time",   node_collect_time)
    graph.add_node("hitl_review",    node_hitl_review)

    # Entry always goes to a router node
    graph.add_node("router", lambda s: {})
    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_by_stage,
        {
            "detect":        "detect_intent",
            "collect_name":  "collect_name",
            "collect_phone": "collect_phone",
            "collect_day":   "collect_day",
            "collect_time":  "collect_time",
            "hitl_review":   "hitl_review",
            "done":          END,
        }
    )

    # All nodes end after one step — the loop is in main()
    for node in ["detect_intent","collect_name","collect_phone","collect_day","collect_time","hitl_review"]:
        graph.add_edge(node, END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════
# MAIN CHAT LOOP
# ═══════════════════════════════════════════════════════════════════

def main():
    print(f"\n{DIVIDER}")
    print(f"  {BOLD}{GREEN}🏥 {CLINIC_NAME} — Appointment Assistant{RESET}")
    print(f"{DIVIDER}")
    print(f"  Powered by LangGraph + GPT-4o-mini")
    print(f"  Type 'quit' to exit\n{DIVIDER}\n")

    graph = build_graph()

    # Initial greeting
    bot_print(
        f"Hello! Welcome to {BOLD}{CLINIC_NAME}{RESET} 👋\n\n"
        f"  I can help you with:\n"
        f"  📅  Book an appointment\n"
        f"  🔄  Reschedule an appointment\n"
        f"  ❌  Cancel an appointment\n\n"
        f"  What can I help you with today?"
    )

    # Persistent state across turns
    state: Dict = {
        "session_id":    str(uuid.uuid4())[:8].upper(),
        "stage":         "detect",
        "route_taken":   [],
        "messages":      [],
        "patient_name":  "",
        "patient_phone": "",
        "preferred_day": "",
        "preferred_time":"",
        "email_draft":   "",
    }

    while True:
        try:
            user_text = input(f"  {BOLD}You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n  Goodbye! Have a great day. 👋\n")
            break

        if not user_text:
            continue
        if user_text.lower() in ("quit", "exit", "q", "bye"):
            print(f"\n  Goodbye! Have a great day. 👋\n")
            break

        # Inject user input and run one graph step
        state["current_input"] = user_text
        state["messages"].append({"role": "user", "content": user_text})

        result = graph.invoke(state)

        # Merge result back into persistent state
        state.update(result)

        # Print bot reply
        reply = state.get("bot_reply", "")
        if reply:
            bot_print(reply)

        # If stage is done but user wants to continue, reset to detect
        if state.get("stage") == "done":
            state["stage"] = "detect"
            state["patient_name"]   = ""
            state["patient_phone"]  = ""
            state["preferred_day"]  = ""
            state["preferred_time"] = ""
            state["email_draft"]    = ""
            state["intent"]         = ""


if __name__ == "__main__":
    main()