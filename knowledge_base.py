"""
SignBridge AI — Knowledge Engine
=================================
Self-contained factual/medical knowledge layer. No API key, no network
call, no rate limit. Runs fully offline.

IMPORTANT (production note): this content is compiled from general
public-health and first-aid guidance for demonstration purposes. Before
any real deployment (hospitals, schools, public use), have a licensed
medical professional review and sign off on every entry in
KNOWLEDGE_BASE below. Treat this file as a single source of truth a
doctor can audit line-by-line — that's the point of owning it instead
of calling a black-box API.
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

    # ── Added to cover gesture.js's expanded 60-word sign vocabulary
    #    (previously only ~half of it had a matching entry here) ──
    {
        "keywords": ["sick", "illness", "unwell"],
        "topic": "Feeling Sick",
        "response": "Feeling generally sick or unwell can have many causes, from minor "
                     "infections to something more serious. Note any other symptoms "
                     "(fever, pain, breathing difficulty) and seek medical advice if it "
                     "does not improve within a day or two, or worsens quickly.",
        "category": "medical",
    },
    {
        "keywords": ["medicine", "medication", "prescription"],
        "topic": "Medicine",
        "response": "Always take medicine exactly as prescribed or as directed on the "
                     "label. If you are unsure about a dose, an interaction with another "
                     "medicine, or a possible allergic reaction, check with a pharmacist "
                     "or doctor before taking it.",
        "category": "medical",
    },
    {
        "keywords": ["nurse"],
        "topic": "Nurses",
        "response": "Nurses provide direct patient care, monitor symptoms, administer "
                     "medication, and are often the first point of contact in a hospital "
                     "or clinic — you can flag a nurse for most immediate, non-emergency "
                     "needs.",
        "category": "medical",
    },
    {
        "keywords": ["allerg"],
        "topic": "Allergic Reaction",
        "response": "Mild allergic reactions (sneezing, itching, mild rash) can often be "
                     "managed with antihistamines. A severe reaction — swelling of the "
                     "face or throat, difficulty breathing, or dizziness — is a medical "
                     "emergency; seek help immediately.",
        "category": "emergency",
    },
    {
        "keywords": ["temperature"],
        "topic": "Body Temperature",
        "response": "A normal body temperature is roughly 36.5\u201337.5\u00B0C (97.7\u201399.5\u00B0F). "
                     "A reading above that range suggests fever; a reading noticeably "
                     "below it can also signal a problem. Either extreme, especially "
                     "with other symptoms, is worth medical attention.",
        "category": "medical",
    },
    {
        "keywords": ["bandage", "dressing", "wound care"],
        "topic": "Bandaging a Wound",
        "response": "Clean a wound gently before covering it, use a sterile bandage or "
                     "dressing, and change it if it becomes wet or dirty. Seek medical "
                     "care for deep cuts, wounds that won't stop bleeding, or signs of "
                     "infection (increasing redness, warmth, swelling, or pus).",
        "category": "medical",
    },
    {
        "keywords": ["blood pressure", "hypertension", "pressure"],
        "topic": "Blood Pressure",
        "response": "Normal blood pressure is roughly 90/60 to 120/80 mmHg. Consistently "
                     "high readings (hypertension) often have no symptoms but raise the "
                     "risk of heart disease and stroke over time, so regular checks and "
                     "medical follow-up matter even without feeling unwell.",
        "category": "medical",
    },
    {
        "keywords": ["better", "improving", "recover"],
        "topic": "Feeling Better",
        "response": "Improvement is a good sign, but it's still worth finishing any "
                     "prescribed medication course and watching for symptoms returning "
                     "or worsening again before considering the issue fully resolved.",
        "category": "health",
    },
    {
        "keywords": ["worse", "worsening", "deteriorat"],
        "topic": "Symptoms Worsening",
        "response": "Symptoms getting worse, rather than improving with rest and time, "
                     "is a signal to seek medical attention rather than wait — this is "
                     "especially true for pain, breathing difficulty, or fever.",
        "category": "health",
    },
    {
        "keywords": ["home"],
        "topic": "Home",
        "response": "Being able to communicate a need to go home, or that you feel safer "
                     "at home, is an important part of care coordination for someone "
                     "who is deaf or speech-impaired.",
        "category": "info",
    },
    {
        "keywords": ["school"],
        "topic": "School",
        "response": "Inclusive education for deaf and hard-of-hearing students works "
                     "best with sign language support, captioning, or an interpreter "
                     "available for classroom communication.",
        "category": "info",
    },
    {
        "keywords": ["famil"],
        "topic": "Family",
        "response": "Involving family in communication and care decisions, where the "
                     "person wants that, helps ensure continuity of support beyond a "
                     "single hospital visit or appointment.",
        "category": "info",
    },
    {
        "keywords": ["friend"],
        "topic": "Friends",
        "response": "Having a trusted friend present, or reachable, can help someone "
                     "communicate more confidently in a medical or public-service "
                     "setting.",
        "category": "info",
    },
    {
        "keywords": ["mother", "mom"],
        "topic": "Mother / Family Contact",
        "response": "Identifying a parent or guardian as a point of contact is often "
                     "important in medical settings, especially for minors or when "
                     "someone wants a family member involved in decisions.",
        "category": "info",
    },
    {
        "keywords": ["father", "dad"],
        "topic": "Father / Family Contact",
        "response": "Identifying a parent or guardian as a point of contact is often "
                     "important in medical settings, especially for minors or when "
                     "someone wants a family member involved in decisions.",
        "category": "info",
    },
    {
        "keywords": ["child"],
        "topic": "Child",
        "response": "When the person needing help is a child, communication should "
                     "involve a parent or guardian wherever possible, and language/tone "
                     "should be kept age-appropriate.",
        "category": "info",
    },
    {
        "keywords": ["angry", "anger"],
        "topic": "Feeling Angry",
        "response": "Frustration or anger is a valid response to communication "
                     "barriers or pain. Naming the emotion clearly can help the person "
                     "assisting you understand the urgency or nature of the situation.",
        "category": "info",
    },
    {
        "keywords": ["happy"],
        "topic": "Feeling Happy",
        "response": "Being able to express positive emotions, not just needs and "
                     "symptoms, is part of full communication access.",
        "category": "info",
    },
    {
        "keywords": ["sad"],
        "topic": "Feeling Sad",
        "response": "Persistent sadness is worth mentioning to a trusted person or "
                     "professional, particularly if it lasts more than a couple of "
                     "weeks or comes with changes in sleep, appetite, or energy.",
        "category": "info",
    },
    {
        "keywords": ["scared", "afraid", "fear"],
        "topic": "Feeling Scared",
        "response": "Fear is a common, valid response in medical or emergency "
                     "situations. Clearly communicating that you're scared can help "
                     "whoever is assisting you adjust their approach and pace.",
        "category": "info",
    },
    {
        "keywords": ["confused", "confusion"],
        "topic": "Feeling Confused",
        "response": "If you don't understand something being explained to you, saying "
                     "so clearly is important — especially for medical instructions, "
                     "where a misunderstanding can matter. It's always fine to ask for "
                     "something to be repeated or simplified.",
        "category": "info",
    },
    {
        "keywords": ["love"],
        "topic": "Expressing Care",
        "response": "Being able to express affection and care for family or friends is "
                     "part of full communication access, not just conveying needs.",
        "category": "info",
    },
]


def lookup_knowledge(phrase: str) -> dict:
    """
    Search the local knowledge base for a matching topic in the given
    phrase. Matches on the MOST SPECIFIC (longest) matching keyword
    across all entries — not simply the first entry in list order —
    so a specific multi-word keyword (e.g. "blood pressure") always
    wins over a shorter, more general one from a different entry (e.g.
    "blood", from the Bleeding entry) that happens to appear earlier
    in KNOWLEDGE_BASE. Ties (equal-length matches) fall back to list
    order, so entry order still matters as a tiebreaker only.
    """
    if not phrase:
        return {"response": None, "topic": None, "category": None, "success": False}

    normalized = re.sub(r"[^\w\s]", " ", phrase.lower())

    best_entry = None
    best_len = -1
    for entry in KNOWLEDGE_BASE:
        for kw in entry["keywords"]:
            if kw in normalized and len(kw) > best_len:
                best_entry = entry
                best_len = len(kw)

    if best_entry:
        return {
            "response": best_entry["response"],
            "topic":    best_entry["topic"],
            "category": best_entry["category"],
            "success":  True,
        }

    return {"response": None, "topic": None, "category": None, "success": False}


def knowledge_base_size() -> int:
    return len(KNOWLEDGE_BASE)
