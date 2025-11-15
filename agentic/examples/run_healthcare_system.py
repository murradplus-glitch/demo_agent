"""Run the healthcare multi-agent system end-to-end."""

from __future__ import annotations

from agentic import HealthcareMultiAgentSystem, load_settings


def main() -> None:
    settings = load_settings()
    system = HealthcareMultiAgentSystem(settings=settings)
    report = system.run(
        patient_query=(
            "I'm a 48 year old with type 2 diabetes. I have had a dry cough and "
            "a mild fever for the last three days and I'm unsure if I should go to urgent care."
        ),
        labs="Blood glucose 155 mg/dL, BP 128/84",
    )
    print(report.to_json())


if __name__ == "__main__":
    main()
