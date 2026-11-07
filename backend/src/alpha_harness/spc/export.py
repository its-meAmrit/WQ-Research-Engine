"""Getting a tested prompt out of Alpha Harness and into WorldQuant's submission form.

There is no documented public SPC submission interface, so export is manual and says so: it
produces the prompt, its metadata and the exact JSON a real run produced, and checks them against
the documented limits first. Nothing here submits anything, and the package says it is not an
official submission.
"""

from typing import TYPE_CHECKING, Any

from ..db.models import SpcPrompt, SpcPromptVersion, SpcRun, SpcRunStatus
from .schema import (
    MAX_ACTIVE_PROMPTS,
    MODEL_VERSION_MAX_CHARS,
    PROFILE,
    PROMPT_MAX_CHARS,
)
from .views import Check, ExportOut

if TYPE_CHECKING:
    from collections.abc import Callable

PACKAGE_FORMAT = "alpha-harness-spc-package/1"
NOT_SUBMITTED = (
    "A local export made by Alpha Harness. It has not been submitted anywhere: WorldQuant "
    "documents no public SPC submission interface, so the prompt and sample are entered by "
    "hand in the platform."
)


def _check(check_id: str, label: str, ok: bool, detail: str, *, warn: bool = False) -> Check:
    status = "pass" if ok else ("warn" if warn else "fail")
    return Check(id=check_id, label=label, status=status, detail=detail)


def readiness(
    prompt: SpcPrompt,
    version: SpcPromptVersion | None,
    run: SpcRun | None,
    *,
    latest_number: int,
    active_prompts: int,
) -> list[Check]:
    """Every submission-readiness check, blocking ones as ``fail`` and the rest as ``warn``."""
    text = version.prompt_text if version else ""
    checks = [
        _check("named", "Prompt has a name", bool(prompt.name.strip()), prompt.name or "Name it."),
        _check(
            "text",
            "Prompt text within the limit",
            0 < len(text) <= PROMPT_MAX_CHARS,
            f"{len(text):,} of {PROMPT_MAX_CHARS:,} characters.",
        ),
        _check(
            "model_version",
            "Model version within the limit",
            len(prompt.model_version or "") <= MODEL_VERSION_MAX_CHARS,
            f"{len(prompt.model_version or '')} of {MODEL_VERSION_MAX_CHARS} characters"
            + ("; none given, which is allowed." if not prompt.model_version else "."),
        ),
        _check(
            "weight",
            "Weight is 0.00 to 1.00",
            0 <= prompt.weight_hundredths <= 100,
            f"{prompt.weight_hundredths / 100:.2f}",
        ),
        _check(
            "active",
            "Weight is above zero",
            prompt.weight_hundredths > 0,
            "A weight of 0 makes the prompt inactive: it will not run for SPC."
            if prompt.weight_hundredths == 0
            else "The prompt will be active.",
            warn=True,
        ),
        _check(
            "model",
            "A model is chosen",
            bool(run and run.model_ref),
            run.model_ref if run else "Test the prompt with a model first.",
        ),
    ]
    if run is None or SpcRunStatus(run.status) is not SpcRunStatus.COMPLETED or not run.valid:
        checks.append(
            _check(
                "tested",
                "Tested, with output that passes the schema",
                False,
                "No completed run with valid output exists for this version. Run it from "
                "Prompt Studio.",
            )
        )
        return checks
    checks.append(
        _check(
            "tested",
            "Tested, with output that passes the schema",
            True,
            f"Run {run.id}, {run.finished_at:%Y-%m-%d %H:%M} UTC, {run.model_ref}.",
        )
    )
    # The sample and prompt are tied to each other by the run's own snapshot, not by
    # comparing against whatever the prompt says now.
    checks.append(
        _check(
            "same_text",
            "Sample came from this exact prompt text",
            bool(version and run.prompt_snapshot == version.prompt_text),
            "The run's saved prompt matches version "
            f"{version.number if version else '?'} word for word."
            if version and run.prompt_snapshot == version.prompt_text
            else "The saved prompt differs from this version.",
        )
    )
    chosen = prompt.preferred_model
    checks.append(
        _check(
            "same_model",
            "Sample came from the selected model",
            chosen is None or chosen == run.model_ref,
            f"Tested on {run.model_ref}."
            if chosen is None or chosen == run.model_ref
            else f"Tested on {run.model_ref}, but {chosen} is selected. Test it again.",
        )
    )
    checks.append(
        _check(
            "latest",
            "This is the newest version",
            version is not None and version.number == latest_number,
            "Up to date."
            if version and version.number == latest_number
            else f"Version {latest_number} exists. Export the tested one, or test the newer.",
            warn=True,
        )
    )
    meta = run.meta_snapshot
    drift = [
        label
        for label, now, then in (
            ("name", prompt.name, meta.get("name")),
            ("update frequency", prompt.update_frequency, meta.get("update_frequency")),
            ("weight", prompt.weight_hundredths, meta.get("weight_hundredths")),
            ("model version", prompt.model_version or None, meta.get("model_version") or None),
        )
        if then is not None and now != then
    ]
    checks.append(
        _check(
            "metadata",
            "Submission details unchanged since the test",
            not drift,
            "Unchanged." if not drift else f"Changed since the test: {', '.join(drift)}.",
            warn=True,
        )
    )
    validation = run.validation or {}
    checks.append(
        _check(
            "warnings",
            "No validation warnings",
            not validation.get("warningCount"),
            f"{validation.get('warningCount', 0)} warning(s); see the run.",
            warn=True,
        )
    )
    checks.append(
        _check(
            "schema",
            "Field names confirmed against WorldQuant's example",
            PROFILE.verified,
            "Confirmed." if PROFILE.verified else PROFILE.note,
            warn=True,
        )
    )
    research = run.research or {}
    got_evidence = research.get("status") == "done" and research.get("sources")
    checks.append(
        _check(
            "research",
            "Tested with current evidence",
            bool(got_evidence),
            f"{research.get('sources')} sources from {research.get('provider')}."
            if got_evidence
            else "No web evidence was used, so the sample reflects the model's own knowledge.",
            warn=True,
        )
    )
    if run.corrects_run_id is not None:
        checks.append(
            _check(
                "corrected",
                "Sample came from the prompt alone",
                False,
                f"This answer came from a corrective pass on run {run.corrects_run_id}, which "
                "showed the model its earlier errors. The prompt on its own may not reproduce "
                "it. Improve the prompt and test again.",
                warn=True,
            )
        )
    checks.append(
        _check(
            "limit",
            f"Within {MAX_ACTIVE_PROMPTS} active prompts",
            active_prompts <= MAX_ACTIVE_PROMPTS,
            f"{active_prompts} local prompts have a weight above zero. WorldQuant allows "
            f"{MAX_ACTIVE_PROMPTS} active prompts per consultant.",
            warn=True,
        )
    )
    return checks


def build(
    prompt: SpcPrompt,
    version: SpcPromptVersion | None,
    run: SpcRun | None,
    checks: list[Check],
    *,
    exported_at: str,
    models: Callable[[str], str] = lambda ref: ref,
) -> ExportOut:
    """The export view. ``package`` is present only when nothing blocks."""
    ready = not any(c.status == "fail" for c in checks)
    package: dict[str, Any] | None = None
    if ready and version is not None and run is not None:
        package = {
            "format": PACKAGE_FORMAT,
            "notice": NOT_SUBMITTED,
            "exportedAt": exported_at,
            "submission": {
                "name": prompt.name,
                "model": models(run.model_ref),
                "modelVersion": prompt.model_version,
                "updateFrequency": prompt.update_frequency,
                "weight": f"{prompt.weight_hundredths / 100:.2f}",
                "promptCharacters": len(version.prompt_text),
            },
            "prompt": version.prompt_text,
            # The model's own answer, untouched: what the platform's schema was checked against.
            "sampleOutput": run.sample_json,
            "test": {
                "runId": run.id,
                "promptVersion": version.number,
                "finishedAt": run.finished_at.isoformat() if run.finished_at else None,
                "provider": run.provider,
                "model": run.model_id,
                "positions": (run.validation or {}).get("itemCount"),
                "validation": {
                    "valid": run.valid,
                    "warnings": (run.validation or {}).get("warningCount", 0),
                    "schemaVerified": PROFILE.verified,
                },
                "research": (run.research or {}).get("status"),
                "correctedFrom": run.corrects_run_id,
            },
            "schema": {
                "keys": list(PROFILE.keys),
                "verified": PROFILE.verified,
                "note": PROFILE.note,
            },
        }
    return ExportOut(
        prompt_id=prompt.id,
        prompt_name=prompt.name,
        version_number=version.number if version else None,
        run_id=run.id if run else None,
        ready=ready,
        checks=checks,
        schema_verified=PROFILE.verified,
        schema_note=PROFILE.note,
        prompt_text=version.prompt_text if version and ready else None,
        sample_json=run.sample_json if run and ready else None,
        package=package,
    )
