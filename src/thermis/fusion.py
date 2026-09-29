from dataclasses import dataclass, field


@dataclass(frozen=True)
class FusionInput:
    actionable_probability: float
    tabular_stage_2: dict[str, float]
    image_stage_2: dict[str, float] | None = None
    thermal_severity: float = 0.0
    infrastructure_exposure: float = 0.0
    context_completeness: float = 1.0

    @classmethod
    def example(cls, context_completeness: float = 1.0) -> "FusionInput":
        return cls(
            actionable_probability=0.8,
            tabular_stage_2={"industrial_fire": 0.8, "persistent_industrial_heat_or_flare": 0.2},
            thermal_severity=0.7,
            infrastructure_exposure=0.6,
            context_completeness=context_completeness,
        )


@dataclass(frozen=True)
class RiskPolicy:
    disagreement_threshold: float = 0.70
    critical_threshold: float = 75.0
    high_threshold: float = 50.0
    moderate_threshold: float = 25.0


@dataclass(frozen=True)
class PredictionResult:
    final_class: str
    confidence: float
    risk_score: float
    risk_level: str
    review_required: bool
    probabilities: dict[str, float] = field(default_factory=dict)


def fuse_prediction(request: FusionInput, policy: RiskPolicy) -> PredictionResult:
    tabular_class = max(request.tabular_stage_2, key=request.tabular_stage_2.get)
    tabular_probability = request.tabular_stage_2[tabular_class]
    review_required = False
    probabilities = dict(request.tabular_stage_2)
    if request.image_stage_2:
        image_class = max(request.image_stage_2, key=request.image_stage_2.get)
        image_probability = request.image_stage_2[image_class]
        if (
            image_class != tabular_class
            and tabular_probability >= policy.disagreement_threshold
            and image_probability >= policy.disagreement_threshold
        ):
            review_required = True
        else:
            probabilities = {
                key: 0.8 * request.tabular_stage_2.get(key, 0.0)
                + 0.2 * request.image_stage_2.get(key, 0.0)
                for key in set(request.tabular_stage_2) | set(request.image_stage_2)
            }
    final_class = (
        "other_or_uncertain" if review_required else max(probabilities, key=probabilities.get)
    )
    confidence = max(probabilities.values(), default=0.0) * max(
        0.0, min(1.0, request.context_completeness)
    )
    industrial_probability = probabilities.get("industrial_fire", 0.0)
    risk_score = 100 * (
        0.40 * request.actionable_probability
        + 0.25 * industrial_probability
        + 0.15 * request.thermal_severity
        + 0.15 * request.infrastructure_exposure
        + 0.05 * request.context_completeness
    )
    if review_required:
        risk_score *= 0.85
    risk_score = max(0.0, min(100.0, risk_score))
    risk_level = (
        "critical" if risk_score >= policy.critical_threshold else
        "high" if risk_score >= policy.high_threshold else
        "moderate" if risk_score >= policy.moderate_threshold else "low"
    )
    return PredictionResult(
        final_class=final_class,
        confidence=confidence,
        risk_score=risk_score,
        risk_level=risk_level,
        review_required=review_required,
        probabilities=probabilities,
    )
