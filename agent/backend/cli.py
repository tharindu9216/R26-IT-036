"""Command-line runner for the emotional support inference pipeline."""

from __future__ import annotations

import argparse
import json

import services
import settings
from emotion_chain.core import CURRENT_EMOTIONS
from emotion_chain.deviation_tracker import compute_deviation
from emotion_chain.reply_generator import ReplySignals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Current user message")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--previous-emotion",
        choices=CURRENT_EMOTIONS,
        default=None,
        help=(
            "Previous turn's emotion. One invocation is one turn, so deviation "
            "tracking needs it passed in; omit it for the first turn."
        ),
    )
    parser.add_argument(
        "--reply",
        action="store_true",
        help=(
            "Also generate the routed reply, using Qwen wherever the active "
            "config places it (this process, or a support node)"
        ),
    )
    args = parser.parse_args()

    chain = services.build_emotion_chain(device=args.device)
    result = chain.predict(args.text)
    deviation = compute_deviation(args.previous_emotion, result.current.label)
    output = {
        "current": {
            "label": result.current.label,
            "confidence": result.current.confidence,
            "probabilities": result.current.probabilities,
        },
        "next": {
            "label": result.next.label,
            "confidence": result.next.confidence,
            "probabilities": result.next.probabilities,
        },
        "forecast": result.forecast.as_dict(),
        "deviation": deviation.as_dict(),
        "reply_route": result.reply_route,
        "intensifies": result.intensifies,
    }
    if args.reply:
        generator = services.build_reply_generator(device=args.device)
        reply = generator.generate(
            args.text,
            ReplySignals(
                result.current.label,
                result.next.label,
                next_emotion_confidence=result.next.confidence,
                next_negative_min_confidence=settings.NEXT_NEGATIVE_MIN_CONFIDENCE,
                previous_emotion=deviation.previous_emotion,
                deviation_level=deviation.level,
                deviation_score=deviation.score,
            ),
        )
        output["reply"] = {
            "text": reply.text,
            "route": reply.route,
            "source": reply.source,
            "support_contacts": [
                contact.as_dict() for contact in reply.support_contacts
            ],
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
