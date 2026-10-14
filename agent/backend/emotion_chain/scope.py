"""Keep the companion an emotional-support companion.

Qwen is a general instruction model, so asked for a Python script it will
write one. Two deterministic checks sit around it:

* ``detect_out_of_scope`` reads the user's message. A match does not skip the
  pipeline -- the emotion models still run and the reply still responds to how
  the user feels -- it adds a directive telling Qwen not to do the task.
  The patterns are deliberately narrow (a request verb next to a task noun),
  so "I'm stressed about my Python exam" stays in scope. Anything they miss
  is left to the scope rules in the system prompts.
* ``reply_contains_code`` reads Qwen's answer. Code is never a valid reply
  from this companion, so a reply that contains some is replaced by
  ``out_of_scope_reply`` whatever the input check decided.

Like the crisis screen, this is a routing aid, not a classifier.
"""

from __future__ import annotations

import re

_FLAGS = re.IGNORECASE

# Verbs that make a message a request rather than a mention. Words such as
# "finish", "complete" or "explain" are left out: "I can't finish my program"
# is about a degree as often as about code.
_REQUEST = (
    r"(?:write|give|generate|create|make|build|fix|debug|implement|code|"
    r"refactor|optimi[sz]e|provide|send)"
)
_DETERMINER = r"(?:(?:me|us)\s+)?(?:(?:a|an|the|some|this|that|my)\s+)?"
_LANGUAGES = (
    r"(?:python|java(?:script)?|typescript|c\+\+|c#|golang|rust|php|ruby|kotlin|"
    r"swift|sql|html|css|bash|powershell|react|node(?:\.?js)?)"
)
_HUMAN_LANGUAGES = (
    r"(?:english|sinhala|tamil|french|spanish|german|italian|portuguese|hindi|"
    r"chinese|mandarin|japanese|korean|arabic|russian|dutch)"
)

# (category, patterns). The first category that matches is reported. Nouns
# with an everyday meaning (class, method, program, error, problem) are not
# used on their own -- "a method to calm down" is in scope.
_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    (
        "coding",
        tuple(
            re.compile(pattern, _FLAGS | re.MULTILINE)
            for pattern in (
                # "write me a python script", "fix this function", "give code"
                rf"\b{_REQUEST}\s+{_DETERMINER}(?:\w+\s+){{0,2}}?"
                r"(?:code|script|function|snippet|algorithm|regex)s?\b",
                # "make a calculator in python", "build it using react"
                rf"\b{_REQUEST}\b[^.?!\n]*\b(?:in|using)\s+{_LANGUAGES}(?![\w+#])",
                # "python code", "sql query"
                rf"\b{_LANGUAGES}\s+(?:code|script|program|function|query|snippet)s?\b",
                r"\bhow\s+(?:do|can|would)\s+i\s+(?:code|program|implement|compile)\b",
                # pasted code or a stack trace
                r"```",
                r"^\s*(?:def|class)\s+\w+\s*[(:]",
                r"^\s*(?:import\s+[\w.]+\s*$|from\s+[\w.]+\s+import\b|#include\s*<)",
                r"\bTraceback \(most recent call last\)",
            )
        ),
    ),
    (
        "maths or homework",
        tuple(
            re.compile(pattern, _FLAGS)
            for pattern in (
                r"\b(?:solve|calculate|compute|differentiate|integrate|simplify|"
                r"factori[sz]e|evaluate)\s+(?:(?:this|the|for|me)\s+)*"
                r"(?:equations?|integrals?|derivatives?|expressions?|[\dx(])",
                r"\bwhat\s+is\s+[\d.]+\s*(?:[-+*/x×÷^]|times|plus|minus|divided by)\s*[\d.]+",
                r"\b(?:do|solve|answer|write)\s+(?:my|this|these|the)\s+"
                r"(?:homework|assignment|worksheet|quiz|exam\s+questions)\b",
            )
        ),
    ),
    (
        "translation",
        tuple(
            re.compile(pattern, _FLAGS)
            for pattern in (
                # Not bare "translate": "I can't translate my feelings into
                # words" is exactly what this companion is for.
                r"\btranslate\s+(?:this|that|it|these|the\s+following)\b",
                rf"\btranslate\b.{{0,80}}?\b(?:in)?to\s+{_HUMAN_LANGUAGES}\b",
                rf"\bhow\s+do\s+(?:you|i)\s+say\b.{{0,60}}\bin\s+{_HUMAN_LANGUAGES}\b",
            )
        ),
    ),
    (
        "writing",
        (
            # Journals and letters are left out on purpose: writing one can be
            # part of coping, so Qwen may help with those.
            re.compile(
                r"\b(?:write|draft|compose|generate|create)\s+"
                + _DETERMINER
                + r"(?:\w+\s+){0,2}?(?:essay|article|report|assignment|"
                r"cover\s+letter|resume|cv|email|e-mail|blog(?:\s+post)?|"
                r"speech|thesis|proposal|product\s+description)s?\b",
                _FLAGS,
            ),
        ),
    ),
)

_CODE_IN_REPLY = tuple(
    re.compile(pattern, _FLAGS | re.MULTILINE)
    for pattern in (
        r"```",
        r"^\s*(?:def|class)\s+\w+\s*[(:]",
        r"^\s*(?:import\s+[\w.]+\s*$|from\s+[\w.]+\s+import\b|#include\s*<)",
        r"^\s*(?:public|private|static)\s+[\w<>\[\]]+\s+\w+\s*\(",
        r"^\s*(?:function\s+\w*\s*\(|const\s+\w+\s*=|let\s+\w+\s*=)",
        r"\bconsole\.log\(|\bSystem\.out\.print|\bprint\(\s*['\"f]",
        r"^\s*SELECT\s.+\sFROM\s",
        r"^\s*<(?:html|div|script|body)\b",
    )
)

_TASK_PHRASES = {
    "coding": "writing or fixing code",
    "maths or homework": "solving maths or homework problems",
    "translation": "translating text",
    "writing": "writing essays, emails or other documents",
}


def detect_out_of_scope(text: str) -> str | None:
    """The task category the message asks for, or None when it is in scope."""

    for category, patterns in _PATTERNS:
        if any(pattern.search(text) for pattern in patterns):
            return category
    return None


def reply_contains_code(text: str) -> bool:
    return any(pattern.search(text) for pattern in _CODE_IN_REPLY)


def scope_directive(category: str | None) -> str:
    """The system-prompt addition for a turn that asks for an off-topic task."""

    task = _TASK_PHRASES.get(category or "", "general tasks like that")
    return (
        f"The user's latest message seems to involve {task}. That is outside "
        "what you do: never do it, not even partly, and include no code, "
        "solutions, translations or drafts. If they are asking you to do it, "
        "kindly say in one sentence that it is not something you can help with "
        "here. If they only mention it while talking about how they feel, do "
        "not bring up what you cannot do. Either way, respond to any feeling "
        "they expressed (for example stress about a deadline) and invite them "
        "to share how they are doing, in 2-3 sentences in total."
    )


def out_of_scope_reply(category: str | None = None) -> str:
    """Fixed reply used when Qwen does the task anyway or cannot be reached."""

    task = _TASK_PHRASES.get(category or "", "code, homework or other tasks like that")
    return (
        f"I'm here for emotional support, so I can't help with {task}. "
        "If something around it is weighing on you, like a deadline, feeling "
        "stuck, or the pressure of it all, I'd really like to hear about that. "
        "How are you feeling right now?"
    )
