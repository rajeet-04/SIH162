from thermis.fusion import FusionInput, RiskPolicy, fuse_prediction


def test_material_model_disagreement_requires_review() -> None:
    result = fuse_prediction(
        FusionInput(
            actionable_probability=0.93,
            tabular_stage_2={"industrial_fire": 0.88, "persistent_industrial_heat_or_flare": 0.04},
            image_stage_2={"industrial_fire": 0.08, "persistent_industrial_heat_or_flare": 0.84},
            thermal_severity=0.8,
            infrastructure_exposure=0.7,
            context_completeness=1.0,
        ),
        RiskPolicy(),
    )
    assert result.final_class == "other_or_uncertain"
    assert result.review_required is True


def test_missing_context_reduces_confidence() -> None:
    complete = FusionInput.example(context_completeness=1.0)
    incomplete = FusionInput.example(context_completeness=0.5)
    assert fuse_prediction(incomplete, RiskPolicy()).confidence < fuse_prediction(
        complete, RiskPolicy()
    ).confidence
