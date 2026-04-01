"""
Interactive multi-turn assistant demo.

Run with:
    python -m src.assistant.demo

This demo shows how ConversationBufferMemory preserves constraints across
turns. Try a sequence like:
    Turn 1: "Find me headphones under 3000"
    Turn 2: "make it cheaper"
    Turn 3: "show me Sony only"
"""
from __future__ import annotations

from src.assistant.chain import run_assistant
from src.assistant.schemas import ConversationSession


def main() -> None:
    user_id = "U_001"

    # Session starts as None — chain.py creates it on the first turn
    session: ConversationSession | None = None

    print("Multi-turn Recommendation Assistant")
    print("Type 'quit' to exit.\n")

    while True:
        user_message = input("You: ").strip()
        if not user_message or user_message.lower() == "quit":
            print("Ending session.")
            break

        # Pass session in — get updated session back
        # This is what makes multi-turn memory work:
        # the session carries ConversationBufferMemory forward each turn.
        response, session = run_assistant(
            user_id=user_id,
            user_message=user_message,
            session=session,
        )

        print(f"\nAssistant: {response.assistant_message}\n")

        if response.recommendations:
            print("Recommendations:")
            for rec in response.recommendations:
                print(f"  {rec.rank}. [{rec.product_id}] {rec.title} "
                      f"— Rs. {rec.price:.0f}  (score: {rec.score:.3f})")
                for reason in rec.reasons:
                    print(f"     • {reason}")
            print()


if __name__ == "__main__":
    main()