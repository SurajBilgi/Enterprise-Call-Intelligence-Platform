"""
Dummy Data Generator for Call Transcripts
==========================================

Generates realistic call transcripts with varying importance levels.

KEY FEATURE: Importance Distribution
- 15% High-priority calls (complaints, critical issues)
- 30% Medium-priority calls (inquiries, questions)
- 55% Low-priority calls (general conversation)

This simulates real-world distribution where most calls are routine,
but important calls need deep analysis.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
from faker import Faker

from models.schemas import (
    SpeakerTurn,
    RawTranscript,
    CallMetadata,
    ClientSegment,
    ProductCategory,
)
from utils.logger import get_logger

logger = get_logger(__name__)
fake = Faker()


class CallTranscriptGenerator:
    """
    Generate realistic call transcripts with controlled importance distribution.

    System Design Note:
    -------------------
    This generator creates data that mimics real-world scenarios:
    - Most calls are routine (low importance)
    - Some calls have questions/issues (medium importance)
    - Few calls are critical complaints (high importance)

    This distribution is CRITICAL for testing:
    - Cost optimization (selective enrichment)
    - Importance scoring accuracy
    - Query routing effectiveness
    """

    # High-priority complaint templates
    COMPLAINT_TEMPLATES = [
        "I'm extremely frustrated with {product}. I've been trying to {action} for {duration} and nothing works!",
        "This is unacceptable! I was charged {amount} incorrectly on my {product}. I demand a refund immediately.",
        "I've called {num_times} times about this issue with {product} and nobody has helped me. I want to speak to a manager!",
        "Your {product} system is completely broken. I can't access my account and I'm losing money because of this.",
        "I'm going to cancel my {product} and switch to a competitor if this isn't resolved today. This is terrible service.",
        "I received a notice about {issue} on my {product} but I never authorized this. This feels like fraud.",
        "The fees on my {product} keep increasing without notice. This is not what I signed up for.",
    ]

    # Medium-priority inquiry templates
    INQUIRY_TEMPLATES = [
        "Hi, I have a question about my {product}. Can you help me understand {question}?",
        "I'm trying to {action} on my {product} but I'm not sure how to do it. Can you guide me?",
        "I noticed {observation} on my {product} statement. Is this normal?",
        "Can you explain what {term} means on my {product}?",
        "I want to make changes to my {product}. What options do I have?",
        "How do I {action} for my {product}? I couldn't find it on the website.",
    ]

    # Low-priority general templates
    GENERAL_TEMPLATES = [
        "Hi, I'm calling to check my {product} balance.",
        "Just wanted to confirm my recent transaction on {product} went through.",
        "I'm interested in learning more about your {product} offerings.",
        "Can you send me a copy of my {product} statement?",
        "I'd like to update my contact information for my {product} account.",
        "Just calling to say thank you for helping me last week with my {product}.",
    ]

    # Agent responses
    AGENT_RESPONSES = [
        "I understand your concern. Let me look into this right away.",
        "I apologize for the inconvenience. I'll help you resolve this.",
        "Let me pull up your account information. One moment please.",
        "I see what you're referring to. Let me explain what happened.",
        "I can definitely help you with that. Here's what we can do.",
        "Thank you for bringing this to our attention. Let's get this sorted out.",
        "I've reviewed your account and I can see the issue.",
    ]

    PRODUCTS = {
        ProductCategory.CREDIT_CARD: {
            "actions": [
                "make a payment",
                "dispute a charge",
                "increase my limit",
                "check my balance",
            ],
            "issues": [
                "unauthorized charges",
                "declined transaction",
                "high interest rates",
                "missing rewards",
            ],
            "terms": ["APR", "credit limit", "minimum payment", "rewards points"],
        },
        ProductCategory.RETIREMENT_FUND: {
            "actions": [
                "make a contribution",
                "check my balance",
                "change my allocation",
                "withdraw funds",
            ],
            "issues": [
                "performance concerns",
                "contribution limits",
                "tax implications",
                "beneficiary changes",
            ],
            "terms": ["vesting schedule", "401(k) match", "RMD", "rollover"],
        },
        ProductCategory.MORTGAGE: {
            "actions": [
                "make a payment",
                "refinance",
                "check my balance",
                "get a payoff quote",
            ],
            "issues": [
                "payment processing delays",
                "escrow shortages",
                "interest rate concerns",
                "late fees",
            ],
            "terms": ["principal", "escrow", "PMI", "amortization"],
        },
        ProductCategory.ETF: {
            "actions": [
                "buy shares",
                "sell shares",
                "check performance",
                "rebalance portfolio",
            ],
            "issues": [
                "trading errors",
                "performance concerns",
                "fee questions",
                "dividend timing",
            ],
            "terms": ["expense ratio", "NAV", "dividend yield", "tracking error"],
        },
        ProductCategory.SAVINGS_ACCOUNT: {
            "actions": [
                "make a deposit",
                "withdraw funds",
                "check my balance",
                "set up auto-transfer",
            ],
            "issues": [
                "interest rate changes",
                "account fees",
                "withdrawal limits",
                "access problems",
            ],
            "terms": [
                "APY",
                "minimum balance",
                "overdraft protection",
                "monthly maintenance fee",
            ],
        },
    }

    REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_dataset(
        self,
        num_calls: int = 1000,
        high_priority_ratio: float = 0.15,
        medium_priority_ratio: float = 0.30,
    ) -> List[Dict[str, Any]]:
        """
        Generate a dataset of call transcripts.

        Args:
            num_calls: Total number of calls to generate
            high_priority_ratio: Ratio of high-priority (complaint) calls
            medium_priority_ratio: Ratio of medium-priority (inquiry) calls

        Returns:
            List of generated call data
        """
        logger.info(
            "generating_dataset",
            num_calls=num_calls,
            high_priority_ratio=high_priority_ratio,
            medium_priority_ratio=medium_priority_ratio,
        )

        calls = []

        # Calculate distribution
        num_high = int(num_calls * high_priority_ratio)
        num_medium = int(num_calls * medium_priority_ratio)
        num_low = num_calls - num_high - num_medium

        # Generate high-priority calls
        for i in range(num_high):
            call = self._generate_call(call_id=f"CALL_{i:06d}", importance="high")
            calls.append(call)

        # Generate medium-priority calls
        for i in range(num_high, num_high + num_medium):
            call = self._generate_call(call_id=f"CALL_{i:06d}", importance="medium")
            calls.append(call)

        # Generate low-priority calls
        for i in range(num_high + num_medium, num_calls):
            call = self._generate_call(call_id=f"CALL_{i:06d}", importance="low")
            calls.append(call)

        # Shuffle to mix importance levels
        random.shuffle(calls)

        # Save to files
        self._save_calls(calls)

        logger.info(
            "dataset_generated",
            total_calls=num_calls,
            high_priority=num_high,
            medium_priority=num_medium,
            low_priority=num_low,
        )

        return calls

    def _generate_call(self, call_id: str, importance: str) -> Dict[str, Any]:
        """Generate a single call transcript"""

        # Select product and metadata
        product = random.choice(list(ProductCategory))
        product_info = self.PRODUCTS.get(
            product, self.PRODUCTS[ProductCategory.CREDIT_CARD]
        )

        # Select template based on importance
        if importance == "high":
            template = random.choice(self.COMPLAINT_TEMPLATES)
            num_turns = random.randint(6, 12)  # Longer calls for complaints
        elif importance == "medium":
            template = random.choice(self.INQUIRY_TEMPLATES)
            num_turns = random.randint(4, 8)
        else:
            template = random.choice(self.GENERAL_TEMPLATES)
            num_turns = random.randint(2, 6)

        # Fill template
        customer_opening = template.format(
            product=product.value.replace("_", " ").title(),
            action=random.choice(product_info["actions"]),
            issue=random.choice(product_info["issues"]),
            question=random.choice(product_info["terms"]),
            observation=random.choice(product_info["issues"]),
            term=random.choice(product_info["terms"]),
            duration="weeks" if importance == "high" else "days",
            amount=f"${random.randint(50, 500)}",
            num_times=random.randint(3, 7),
        )

        # Generate conversation
        transcript = []
        current_time = 0.0

        # Customer opens
        turn_duration = random.uniform(3, 8)
        transcript.append(
            SpeakerTurn(
                speaker="customer",
                text=customer_opening,
                start_time=current_time,
                end_time=current_time + turn_duration,
                duration=turn_duration,
            )
        )
        current_time += turn_duration

        # Agent responds
        turn_duration = random.uniform(2, 5)
        transcript.append(
            SpeakerTurn(
                speaker="agent",
                text=random.choice(self.AGENT_RESPONSES),
                start_time=current_time,
                end_time=current_time + turn_duration,
                duration=turn_duration,
            )
        )
        current_time += turn_duration

        # Continue conversation
        for _ in range(num_turns - 2):
            speaker = random.choice(["customer", "agent"])

            if speaker == "customer":
                if importance == "high":
                    text = random.choice(
                        [
                            "This is not acceptable. I need this fixed now.",
                            "I've been a customer for years and this is how you treat me?",
                            "I'm very disappointed with this service.",
                            f"The issue is with {random.choice(product_info['issues'])}.",
                        ]
                    )
                elif importance == "medium":
                    text = random.choice(
                        [
                            "Okay, thank you for checking.",
                            f"So what about {random.choice(product_info['terms'])}?",
                            "That makes sense. Is there anything else I need to know?",
                            "I appreciate your help with this.",
                        ]
                    )
                else:
                    text = random.choice(
                        [
                            "Great, thank you.",
                            "That's all I needed.",
                            "Perfect, I appreciate it.",
                            "Okay, that helps.",
                        ]
                    )
            else:  # agent
                text = random.choice(
                    [
                        "Let me check on that for you.",
                        "I've made that update to your account.",
                        "Is there anything else I can help you with today?",
                        "I understand. Let me see what I can do.",
                        "I've escalated this to our supervisor.",
                        "You should see this resolved within 24-48 hours.",
                    ]
                )

            turn_duration = random.uniform(2, 6)
            transcript.append(
                SpeakerTurn(
                    speaker=speaker,
                    text=text,
                    start_time=current_time,
                    end_time=current_time + turn_duration,
                    duration=turn_duration,
                )
            )
            current_time += turn_duration

        # Generate metadata
        call_timestamp = datetime.now() - timedelta(days=random.randint(0, 90))

        metadata = CallMetadata(
            call_id=call_id,
            client_id=f"CLIENT_{random.randint(1000, 9999)}",
            client_segment=random.choice(list(ClientSegment)),
            product=product,
            call_duration_seconds=current_time,
            call_timestamp=call_timestamp,
            region=random.choice(self.REGIONS),
            agent_id=f"AGENT_{random.randint(100, 999)}",
            call_outcome=random.choice(["resolved", "escalated", "pending"]),
        )

        # Create raw transcript
        raw_transcript = RawTranscript(
            call_id=call_id,
            transcript=transcript,
            metadata=metadata.model_dump(),
            ingestion_timestamp=datetime.now(),
        )

        return {
            "transcript": raw_transcript,
            "metadata": metadata,
            "importance": importance,  # Ground truth for evaluation
        }

    def _save_calls(self, calls: List[Dict[str, Any]]):
        """Save calls to JSON files"""
        for call_data in calls:
            call_id = call_data["transcript"].call_id
            file_path = self.output_dir / f"{call_id}.json"

            # Convert to dict for JSON serialization
            data = {
                "call_id": call_id,
                "transcript": [
                    {
                        "speaker": turn.speaker,
                        "text": turn.text,
                        "start_time": turn.start_time,
                        "end_time": turn.end_time,
                        "duration": turn.duration,
                    }
                    for turn in call_data["transcript"].transcript
                ],
                "metadata": call_data["metadata"].model_dump(mode="json"),
                "ingestion_timestamp": call_data[
                    "transcript"
                ].ingestion_timestamp.isoformat(),
                "ground_truth_importance": call_data["importance"],  # For evaluation
            }

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)


def main():
    """Generate dummy dataset"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate dummy call transcripts")
    parser.add_argument(
        "--num-calls", type=int, default=1000, help="Number of calls to generate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="storage/raw_transcripts",
        help="Output directory",
    )
    parser.add_argument(
        "--high-priority-ratio",
        type=float,
        default=0.15,
        help="Ratio of high-priority calls",
    )
    parser.add_argument(
        "--medium-priority-ratio",
        type=float,
        default=0.30,
        help="Ratio of medium-priority calls",
    )

    args = parser.parse_args()

    generator = CallTranscriptGenerator(output_dir=args.output_dir)
    generator.generate_dataset(
        num_calls=args.num_calls,
        high_priority_ratio=args.high_priority_ratio,
        medium_priority_ratio=args.medium_priority_ratio,
    )

    print(f"✓ Generated {args.num_calls} call transcripts in {args.output_dir}")


if __name__ == "__main__":
    main()
