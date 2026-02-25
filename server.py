#!/usr/bin/env python3
"""
server.py
=========
Flask server that:
  - Serves frontend.html at /
  - Exposes POST /api/chat  → runs LangGraph pipeline → returns bot reply
  - Exposes POST /api/email → generates GPT-4o-mini email draft

Usage:
  pip install flask flask-cors
  python server.py
  Then open http://localhost:8080
"""

from __future__ import annotations
import functools
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from langgraph.graph import StateGraph, END
from openai import OpenAI
from typing_extensions import TypedDict

load_dotenv()

# ── Validate API key ──────────────────────────────────────────────
_api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not _api_key or _api_key == "your-key-here":
    print("\n  ❌  ERROR: No OpenAI API key found in .env\n")
    exit(1)

_openai  = OpenAI(api_key=_api_key)
app      = Flask(__name__, static_folder=".")
CORS(app)

CLINIC_NAME  = "Medical Clinic"
CLINIC_EMAIL = "appointments@medicalclinic.com"
MODEL        = "gpt-4o-mini"
MAX_CALLS    = 15
MAX_CHARS    = 2000


# ═══════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════

class AppointmentState(TypedDict, total=False):
    run_id:             str
    raw_message:        str
    intent:             str
    extracted_entities: Dict[str, Any]
    missing_fields:     List[str]
    moderation_flag:    bool
    pii_detected:       bool
    pii_fields:         List[str]
    call_count:         int
    hitl_required:      bool
    hitl_reason:        str
    hitl_approved:      Optional[bool]
    draft_response:     str
    final_response:     str
    status:             str
    route_taken:        List[str]
    error:              str


# ═══════════════════════════════════════════════════════════════════
# MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════

def _log(label: str, msg: str):
    print(f"  [MIDDLEWARE:{label}] {msg}")

def OpenAIModerationMiddleware(state):
    flagged = [r"\b(bomb|weapon|kill|suicide|abuse)\b", r"\b(hack|exploit|injection)\b"]
    text = state.get("raw_message", "")
    for p in flagged:
        if re.search(p, text, re.IGNORECASE):
            _log("Moderation", "Flagged")
            return {"moderation_flag": True, "status": "ESCALATE"}
    _log("Moderation", "Cleared")
    return {"moderation_flag": False}

def PIIMiddleware(state):
    rules = {
        "ssn":   re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "dob":   re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"),
        "phone": re.compile(r"\b\(?\d{3}\)?[\s\-]\d{3}[\s\-]\d{4}\b"),
        "email": re.compile(r"\b[\w.+-]+@[\w-]+\.\w+\b"),
        "mrn":   re.compile(r"\bMRN[:\s]?\d{6,}\b", re.IGNORECASE),
    }
    found = [n for n, p in rules.items() if p.search(state.get("raw_message", ""))]
    if found:
        _log("PII", f"Detected field types: {found}")
        return {"pii_detected": True, "pii_fields": found}
    return {"pii_detected": False, "pii_fields": []}

def ContextEditingMiddleware(state):
    text = state.get("raw_message", "")
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + " [TRIMMED]"
    entities = dict(state.get("extracted_entities") or {})
    if re.search(r"\b(urgent|emergency|asap)\b", text, re.IGNORECASE):
        entities["urgency"] = "high"
    return {"raw_message": text, "extracted_entities": entities}

def ToolCallLimitMiddleware(state):
    count = state.get("call_count", 0)
    if count >= MAX_CALLS:
        return {"status": "ESCALATE", "error": "Max call limit exceeded"}
    return {}

def HumanInTheLoopMiddleware(state):
    if state.get("hitl_required") is not None:
        return {}
    triggers = {
        "cancel":     "Cancellation requires staff confirmation",
        "reschedule": "Reschedule requires supervisor approval",
    }
    reason = triggers.get(state.get("intent", ""))
    if reason:
        _log("HITL", f"Required — {reason}")
        return {"hitl_required": True, "hitl_reason": reason}
    return {"hitl_required": False}

def ModelFallbackMiddleware(primary_fn, fallback_fn):
    def _wrapped(state):
        try:
            r = primary_fn(state)
            _log("ModelFallback", "Primary succeeded")
            return r
        except Exception as exc:
            _log("ModelFallback", f"Primary failed ({exc.__class__.__name__}), using fallback")
            return fallback_fn(state)
    return _wrapped


# ═══════════════════════════════════════════════════════════════════
# NLU
# ═══════════════════════════════════════════════════════════════════

_INTENT_PATTERNS = {
    "reschedule": [r"\b(reschedule|move|change|shift|postpone|rebook)\b.*\bappointment\b", r"\bappointment\b.*\b(reschedule|move|change)\b"],
    "cancel":     [r"\b(cancel|drop|remove|delete)\b.*\bappointment\b", r"\bappointment\b.*\b(cancel|drop)\b", r"\bcancel\b"],
    "prep_instructions": [r"\b(prep|preparation|prepare|instructions?)\b.*\b(mri|ct|scan|colonoscopy|endoscopy|blood test|lab)\b", r"\b(fasting|fast)\b"],
}

def classify_intent(text):
    t = text.lower()
    for intent, patterns in _INTENT_PATTERNS.items():
        for p in patterns:
            if re.search(p, t):
                return intent
    return "unknown"

def extract_entities(text):
    entities = {}
    date_re = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|next\s+\w+|tomorrow|today|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b", re.IGNORECASE)
    time_re = re.compile(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm)|morning|afternoon|evening|noon)\b", re.IGNORECASE)
    proc_re = re.compile(r"\b(mri|ct scan|x-?ray|ultrasound|colonoscopy|endoscopy|blood test|lab work|imaging|surgery)\b", re.IGNORECASE)
    if d  := date_re.findall(text): entities["date_mentions"] = d
    if t  := time_re.findall(text): entities["time_mentions"] = t
    if pr := proc_re.findall(text): entities["procedures"]    = pr
    return entities

def find_missing_fields(intent, entities):
    required = {"reschedule": ["date_mentions"], "cancel": [], "prep_instructions": ["procedures"], "unknown": []}
    return [f for f in required.get(intent, []) if not entities.get(f)]


# ═══════════════════════════════════════════════════════════════════
# GPT RESPONSE GENERATION
# ═══════════════════════════════════════════════════════════════════

_SYSTEM = f"""You are a warm, professional appointment assistant for {CLINIC_NAME}.
Help patients with bookings, rescheduling, cancellations, and procedure prep instructions.
- Be concise, friendly, reassuring. Under 120 words unless giving prep instructions.
- Never confirm bookings yourself — say staff will follow up.
- No greetings or sign-offs."""

def generate_response(state):
    intent   = state.get("intent", "unknown")
    entities = state.get("extracted_entities", {})
    missing  = state.get("missing_fields", [])
    raw_msg  = state.get("raw_message", "")

    parts = [f"Patient request: {raw_msg}", f"Intent: {intent}"]
    if entities.get("date_mentions"): parts.append(f"Date: {', '.join(entities['date_mentions'])}")
    if entities.get("procedures"):    parts.append(f"Procedure: {', '.join(entities['procedures'])}")
    if missing:                       parts.append(f"Missing info needed: {', '.join(missing)}")
    if intent == "prep_instructions": parts.append("Give detailed accurate prep instructions.")
    elif missing:                     parts.append("Politely ask for missing info.")
    elif intent in ("reschedule","cancel"): parts.append("Acknowledge and say staff will confirm.")
    else:                             parts.append("Clarify what they need.")

    resp = _openai.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":_SYSTEM},{"role":"user","content":"\n".join(parts)}],
        temperature=0.4, max_tokens=300,
    )
    return {"draft_response": resp.choices[0].message.content.strip(), "call_count": state.get("call_count",0)+1}

def fallback_response(state):
    return {"draft_response": f"Thank you. A team member will follow up shortly. You can also reach us at {CLINIC_EMAIL}.", "call_count": state.get("call_count",0)+1}


# ═══════════════════════════════════════════════════════════════════
# LANGGRAPH NODES
# ═══════════════════════════════════════════════════════════════════

def node_input_validation(state):
    return {"status":"NEED_INFO","error":"Empty message"} if not state.get("raw_message","").strip() else {}

def node_moderation_check(state):   return OpenAIModerationMiddleware(state)
def node_pii_check(state):          return PIIMiddleware(state)
def node_context_edit(state):       return ContextEditingMiddleware(state)
def node_call_limit_check(state):   return ToolCallLimitMiddleware(state)

def node_intent_classify(state):
    intent = classify_intent(state.get("raw_message",""))
    print(f"  [NLU] Intent: {intent}")
    return {"intent": intent, "call_count": state.get("call_count",0)+1}

def node_entity_extract(state):
    existing = dict(state.get("extracted_entities") or {})
    existing.update(extract_entities(state.get("raw_message","")))
    return {"extracted_entities": existing, "call_count": state.get("call_count",0)+1}

def node_missing_field_check(state):
    missing = find_missing_fields(state.get("intent","unknown"), state.get("extracted_entities",{}))
    return {"missing_fields": missing}

def node_hitl_gate(state):          return HumanInTheLoopMiddleware(state)

def node_hitl_review(state):
    # In web mode HITL is handled by the frontend confirm step — auto-approve here
    _log("HITL", "Web mode — flagging for frontend review")
    return {"hitl_approved": None}   # frontend will confirm

_response_node = ModelFallbackMiddleware(generate_response, fallback_response)
def node_response_generate(state):  return _response_node(state)

def node_finalize(state):
    moderated  = state.get("moderation_flag", False)
    missing    = state.get("missing_fields", [])
    intent     = state.get("intent","unknown")
    status     = state.get("status","")
    draft      = state.get("draft_response","")
    error      = state.get("error","")

    if status == "ESCALATE" or moderated:
        final_status = "ESCALATE"
        final_resp   = "We were unable to process your message. Please contact our office directly." if moderated else f"Your request requires staff attention. {error}"
    elif status == "NEED_INFO" or missing or intent == "unknown":
        final_status = "NEED_INFO"
        final_resp   = draft
    else:
        final_status = "READY"
        final_resp   = draft

    print(f"  [Finalize] Status: {final_status}")
    return {"status": final_status, "final_response": final_resp}


# ═══════════════════════════════════════════════════════════════════
# LANGGRAPH GRAPH
# ═══════════════════════════════════════════════════════════════════

def route_after_moderation(state):     return "escalate" if state.get("moderation_flag") else "continue"
def route_after_call_limit(state):     return "escalate" if state.get("status")=="ESCALATE" else "continue"
def route_after_missing(state):        return "need_info" if (state.get("missing_fields") or state.get("intent")=="unknown") else "continue"
def route_after_hitl_gate(state):      return "hitl" if state.get("hitl_required") else "generate"

def build_graph():
    g = StateGraph(AppointmentState)
    g.add_node("input_validation",    node_input_validation)
    g.add_node("moderation_check",    node_moderation_check)
    g.add_node("pii_check",           node_pii_check)
    g.add_node("context_edit",        node_context_edit)
    g.add_node("call_limit_check",    node_call_limit_check)
    g.add_node("intent_classify",     node_intent_classify)
    g.add_node("entity_extract",      node_entity_extract)
    g.add_node("missing_field_check", node_missing_field_check)
    g.add_node("hitl_gate",           node_hitl_gate)
    g.add_node("hitl_review",         node_hitl_review)
    g.add_node("response_generate",   node_response_generate)
    g.add_node("finalize",            node_finalize)

    g.set_entry_point("input_validation")
    g.add_edge("input_validation",    "moderation_check")
    g.add_edge("pii_check",           "context_edit")
    g.add_edge("context_edit",        "call_limit_check")
    g.add_edge("intent_classify",     "entity_extract")
    g.add_edge("entity_extract",      "missing_field_check")
    g.add_edge("hitl_review",         "response_generate")
    g.add_edge("response_generate",   "finalize")
    g.add_edge("finalize",            END)

    g.add_conditional_edges("moderation_check",    route_after_moderation, {"continue":"pii_check",          "escalate":"finalize"})
    g.add_conditional_edges("call_limit_check",    route_after_call_limit, {"continue":"intent_classify",    "escalate":"finalize"})
    g.add_conditional_edges("missing_field_check", route_after_missing,    {"continue":"hitl_gate",          "need_info":"response_generate"})
    g.add_conditional_edges("hitl_gate",           route_after_hitl_gate,  {"generate":"response_generate",  "hitl":"hitl_review"})

    return g.compile()

_graph = build_graph()


# ═══════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(".", "frontend.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Receives: { message: str }
    Returns:  { reply: str, status: str, hitl_required: bool, route: [...] }
    """
    data    = request.get_json() or {}
    message = data.get("message", "").strip()
    run_id  = f"APPT-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8].upper()}"

    print(f"\n  [API] /chat  run={run_id}  len={len(message)}")

    result = _graph.invoke({
        "run_id":             run_id,
        "raw_message":        message,
        "route_taken":        [],
        "call_count":         0,
        "extracted_entities": {},
    })

    return jsonify({
        "reply":        result.get("final_response", ""),
        "status":       result.get("status", "READY"),
        "hitl_required":result.get("hitl_required", False),
        "hitl_reason":  result.get("hitl_reason", ""),
        "intent":       result.get("intent", "unknown"),
        "route":        result.get("route_taken", []),
        "run_id":       run_id,
    })


@app.route("/api/email", methods=["POST"])
def api_email():
    """
    Receives: { intent, name, phone, day, time }
    Returns:  { subject: str, body: str }
    Calls GPT-4o-mini to draft the email.
    """
    data   = request.get_json() or {}
    intent = data.get("intent", "book")
    name   = data.get("name", "")
    phone  = data.get("phone", "")
    day    = data.get("day", "")
    time_  = data.get("time", "")

    print(f"\n  [API] /email  intent={intent}  name=[redacted]")

    action = {"book":"book a new appointment","cancel":"cancel my appointment","reschedule":"reschedule my appointment"}.get(intent,"book a new appointment")
    subj_label = {"book":"New Appointment Request","cancel":"Appointment Cancellation Request","reschedule":"Appointment Reschedule Request"}.get(intent,"Appointment Request")

    system = f"""You are drafting a short, professional email on behalf of a patient to {CLINIC_NAME}.
Return ONLY the email body — no subject line, no extra commentary.
Sign off with the patient's name."""

    user = f"""Draft an email to {CLINIC_EMAIL} to {action}.
Patient name: {name}
Patient phone: {phone}
{'Appointment to ' + ('cancel' if intent=='cancel' else 'reschedule') + ':' if intent != 'book' else 'Preferred appointment:'}
  Day:  {day}
  Time: {time_}
Clinic: {CLINIC_NAME}"""

    try:
        resp = _openai.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.3, max_tokens=300,
        )
        body = resp.choices[0].message.content.strip()
    except Exception as e:
        body = f"Dear {CLINIC_NAME} Team,\n\nI would like to {action}.\n\nPatient: {name}\nPhone: {phone}\nDay: {day}\nTime: {time_}\n\nPlease contact me to confirm.\n\nThank you,\n{name}"

    return jsonify({
        "subject": f"{subj_label} — {name}",
        "body":    body,
    })


@app.route("/api/appointments", methods=["POST"])
def api_appointments():
    """
    Receives: { name: str }
    Returns:  { found: bool, appointments: [...] }
    """
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    print(f"\n  [API] /appointments  name=[redacted]")

    try:
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "appointments.json")
        with open(json_path) as f:
            all_appointments = json.load(f)
    except Exception as e:
        return jsonify({"found": False, "appointments": [], "error": str(e)})

    matched_key = next((k for k in all_appointments if k.lower() == name.lower()), None)

    if not matched_key:
        return jsonify({"found": False, "appointments": []})

    return jsonify({
        "found":        True,
        "name":         matched_key,
        "appointments": all_appointments[matched_key]
    })


@app.route("/api/appointments/update", methods=["POST"])
def api_appointments_update():
    """
    Receives: { name, appointment_id, new_date, new_time }
    Updates appointment in appointments.json.
    """
    data     = request.get_json() or {}
    name     = data.get("name", "").strip()
    apt_id   = data.get("appointment_id", "").strip()
    new_date = data.get("new_date", "").strip()
    new_time = data.get("new_time", "").strip()
    print(f"\n  [API] /appointments/update  id={apt_id}")

    try:
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "appointments.json")
        with open(json_path) as f:
            all_appointments = json.load(f)

        matched_key = next((k for k in all_appointments if k.lower() == name.lower()), None)
        if not matched_key:
            return jsonify({"success": False, "error": "Patient not found"})

        updated = None
        for apt in all_appointments[matched_key]:
            if apt["id"] == apt_id:
                apt["date"] = new_date
                apt["time"] = new_time
                updated = apt
                break

        if not updated:
            return jsonify({"success": False, "error": "Appointment not found"})

        with open(json_path, "w") as f:
            json.dump(all_appointments, f, indent=2)

        return jsonify({"success": True, "appointment": updated})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ═══════════════════════════════════════════════════════════════════
# ENTRY
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  {CLINIC_NAME} — Appointment Server")
    print(f"  http://localhost:8080")
    print(f"  Model : {MODEL}")
    print(f"{'='*55}\n")
    app.run(debug=True, port=8080)