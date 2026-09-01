from thermis.evaluation import PromotionMetrics, promotion_decision


def test_model_passes_only_when_every_hard_gate_passes() -> None:
    metrics = PromotionMetrics(
        industrial_fire_recall=0.86,
        macro_f1=0.78,
        persistent_false_alert_rate=0.14,
        cpu_latency_ms_p95=420.0,
        severe_segment_collapse=False,
    )
    assert promotion_decision(metrics).promoted is True


def test_segment_collapse_blocks_promotion() -> None:
    metrics = PromotionMetrics(
        industrial_fire_recall=0.90,
        macro_f1=0.82,
        persistent_false_alert_rate=0.10,
        cpu_latency_ms_p95=300.0,
        severe_segment_collapse=True,
    )
    assert promotion_decision(metrics).promoted is False
