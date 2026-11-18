from src.thai_mod_api.binary_metrics import calculate_binary_metrics


def test_binary_metrics_report_support_and_defamation_recall():
    metrics = calculate_binary_metrics(
        [0, 0, 1, 1], [0, 1, 1, 0], sources=["real", "real", "real", "real"]
    )
    assert metrics["per_class"]["0"]["support"] == 2
    assert metrics["per_class"]["1"]["support"] == 2
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["real_only"]["defamation_recall"] == 0.5
