"""Run the healthcare multi-agent system end-to-end."""

from __future__ import annotations

from agentic import HealthcareMultiAgentSystem, load_settings


def main() -> None:
    settings = load_settings()
    system = HealthcareMultiAgentSystem(settings=settings)
    report = system.run(
        patient_query=(
            "My child has had a persistent fever for two days and I would like to use "
            "our Sehat Card for care. Which facility should we visit?"
        ),
        citizen_profile={
            "name": "Sara",
            "age": 6,
            "city": "Rawalpindi",
            "area": "Shamsabad",
            "region": "Punjab",
            "nser_score": 27,
            "income_per_month_pkrs": 26000,
            "family_size": 5,
            "preferred_language": "Urdu",
            "conditions": ["No chronic conditions"],
        },
    )
    print(report.to_json())


if __name__ == "__main__":
    main()
