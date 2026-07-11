"""
SignBridge AI — Knowledge Engine
=================================
Self-contained factual/medical knowledge layer. No API key, no network
call, no rate limit. Runs fully offline. Unchanged from the original —
included here only so this patch folder is a complete drop-in set.
"""

import re

KNOWLEDGE_BASE = [
    {
        "keywords": ["chest", "heart attack"],
        "topic": "Chest Pain",
        "response": "Chest pain can be a sign of a serious heart-related emergency. "
                     "If it is sudden, severe, or comes with shortness of breath or "
                     "sweating, treat it as an emergency and seek immediate medical help.",
        "category": "emergency",
    },
    {
        "keywords": ["help", "emergency", "sos"],
        "topic": "Medical Emergency",
        "response": "A medical emergency is any sudden condition that poses an immediate "
                     "risk to life or health. Stay calm, alert someone nearby, and contact "
                     "emergency services as quickly as possible.",
        "category": "emergency",
    },
    {
        "keywords": ["breathe", "breathing", "dyspnea"],
        "topic": "Breathing Difficulty",
        "response": "Difficulty breathing can result from asthma, anxiety, infection, or a "
                     "more serious cardiac or respiratory issue. Sit upright, stay calm, and "
                     "seek medical attention if it does not ease quickly.",
        "category": "emergency",
    },
    {
        "keywords": ["blood", "bleeding", "hemorrhage"],
        "topic": "Bleeding",
        "response": "For active bleeding, apply firm, direct pressure to the wound with a "
                     "clean cloth and keep the area raised if possible. Seek medical help "
                     "immediately for heavy or uncontrolled bleeding.",
        "category": "emergency",
    },
    {
        "keywords": ["seizure", "epileptic"],
        "topic": "Seizure",
        "response": "During a seizure, clear the area of hard or sharp objects, do not "
                     "restrain the person, and time the episode. Seek emergency help if it "
                     "lasts more than five minutes or repeats.",
        "category": "emergency",
    },
    {
        "keywords": ["stroke"],
        "topic": "Stroke Warning Signs",
        "response": "Stroke symptoms include sudden facial drooping, arm weakness, and "
                     "slurred speech (the F.A.S.T. signs). Every minute matters — call "
                     "emergency services immediately if these appear.",
        "category": "emergency",
    },
    {
        "keywords": ["unconscious", "fainted", "faint", "syncope"],
        "topic": "Loss of Consciousness",
        "response": "If someone is unconscious but breathing, lay them on their side, check "
                     "their airway is clear, and seek medical help. If they are not "
                     "breathing, emergency medical care is needed immediately.",
        "category": "emergency",
    },
    {
        "keywords": ["poison", "poisoning"],
        "topic": "Poisoning",
        "response": "If poisoning is suspected, do not induce vomiting unless instructed by "
                     "a medical professional. Contact emergency services or a poison "
                     "control helpline right away with details of what was consumed.",
        "category": "emergency",
    },
    {
        "keywords": ["fire"],
        "topic": "Fire Safety",
        "response": "In case of fire, leave the area immediately, stay low to avoid smoke, "
                     "and alert others. Do not use elevators. Call emergency services once "
                     "safely outside.",
        "category": "emergency",
    },
    {
        "keywords": ["ambulance", "call 108", "call 112"],
        "topic": "Emergency Contact Numbers (India)",
        "response": "In India, dial 112 for a general emergency, or 108 specifically for "
                     "ambulance and medical emergencies. Both are free and available "
                     "nationwide.",
        "category": "emergency",
    },
    {
        "keywords": ["dizzy", "dizziness", "vertigo"],
        "topic": "Dizziness",
        "response": "Dizziness can be caused by dehydration, low blood sugar, inner-ear "
                     "issues, or low blood pressure. Sit or lie down, hydrate, and seek "
                     "medical advice if it persists or recurs often.",
        "category": "medical",
    },
    {
        "keywords": ["fever"],
        "topic": "Fever",
        "response": "A fever is a body temperature above roughly 38\u00B0C (100.4\u00B0F) and "
                     "often signals infection. Rest, fluids, and monitoring are usually "
                     "advised; seek care if it is high, persistent, or paired with other "
                     "severe symptoms.",
        "category": "medical",
    },
    {
        "keywords": ["fall", "fell", "injury"],
        "topic": "Fall Injury",
        "response": "After a fall, check for pain, swelling, or inability to move the "
                     "affected area before attempting to stand. Seek medical evaluation for "
                     "head injuries or suspected fractures.",
        "category": "medical",
    },
    {
        "keywords": ["hospital"],
        "topic": "When to Go to a Hospital",
        "response": "Go to a hospital or emergency room for symptoms that are sudden, "
                     "severe, or rapidly worsening \u2014 such as chest pain, breathing "
                     "difficulty, heavy bleeding, or loss of consciousness.",
        "category": "medical",
    },
    {
        "keywords": ["doctor"],
        "topic": "Seeing a Doctor",
        "response": "A general physician can assess most non-emergency symptoms and refer "
                     "you to a specialist if needed. For urgent or severe symptoms, go "
                     "directly to an emergency department instead of waiting for an "
                     "appointment.",
        "category": "medical",
    },
    {
        "keywords": ["pain"],
        "topic": "Pain Management",
        "response": "Persistent or severe pain should be evaluated by a medical "
                     "professional rather than self-treated, especially if it is sudden, "
                     "intense, or unexplained.",
        "category": "medical",
    },
    {
        "keywords": ["water", "thirsty", "dehydrat"],
        "topic": "Hydration",
        "response": "Adults generally need about 2\u20133 liters of water per day. Adequate "
                     "hydration supports body temperature regulation, organ function, and "
                     "overall health.",
        "category": "health",
    },
    {
        "keywords": ["food", "hungry", "eat"],
        "topic": "Nutrition",
        "response": "A balanced diet with adequate protein, vegetables, and hydration "
                     "supports energy, immunity, and recovery. Skipping meals for extended "
                     "periods can affect concentration and health.",
        "category": "health",
    },
    {
        "keywords": ["toilet", "washroom", "restroom"],
        "topic": "Basic Needs",
        "response": "Regular access to washroom facilities is a basic need; prolonged "
                     "delay in addressing this can cause discomfort and, rarely, urinary "
                     "tract issues.",
        "category": "health",
    },
    {
        "keywords": ["tired", "sleep", "rest"],
        "topic": "Rest & Sleep",
        "response": "Most adults need 7\u20139 hours of sleep per night. Persistent fatigue "
                     "despite adequate rest can be worth discussing with a doctor.",
        "category": "health",
    },
    {
        "keywords": ["deaf", "hearing"],
        "topic": "Deafness & Hearing Loss",
        "response": "Deafness or hearing loss can range from mild to profound and may be "
                     "present from birth or acquired later in life. Sign language, hearing "
                     "aids, or assistive technology can all support communication.",
        "category": "info",
    },
    {
        "keywords": ["hello", "hi"],
        "topic": "Greeting",
        "response": "A greeting is a simple, universal way to open communication and "
                     "signal friendly intent \u2014 essential in any first interaction.",
        "category": "info",
    },
    {
        "keywords": ["good", "fine", "okay"],
        "topic": "Wellbeing Check-In",
        "response": "Regularly checking in on how someone is feeling supports both "
                     "physical and mental wellbeing, and helps catch problems early.",
        "category": "info",
    },
    {
        "keywords": ["please", "thank"],
        "topic": "Communication Courtesy",
        "response": "Polite phrases like \u2018please\u2019 and \u2018thank you\u2019 help build "
                     "trust and cooperation, especially in cross-ability communication "
                     "where clarity and goodwill matter most.",
        "category": "info",
    },
    {
        "keywords": ["call"],
        "topic": "Making Contact",
        "response": "When direct speech is not possible, having a clear way to request a "
                     "phone call or alert someone nearby is an essential accessibility "
                     "feature.",
        "category": "info",
    },
]


def lookup_knowledge(phrase: str) -> dict:
    if not phrase:
        return {"response": None, "topic": None, "category": None, "success": False}

    normalized = re.sub(r"[^\w\s]", " ", phrase.lower())

    for entry in KNOWLEDGE_BASE:
        for kw in entry["keywords"]:
            if kw in normalized:
                return {
                    "response": entry["response"],
                    "topic":    entry["topic"],
                    "category": entry["category"],
                    "success":  True,
                }

    return {"response": None, "topic": None, "category": None, "success": False}


def knowledge_base_size() -> int:
    return len(KNOWLEDGE_BASE)
