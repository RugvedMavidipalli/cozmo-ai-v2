from cozmo_ai_v2.pipeline.poses import evaluate_refinement_acceptance


def test_refinement_acceptance_requires_objective_and_loop_residual_improvement():
    accepted, reasons = evaluate_refinement_acceptance(
        max_correction=0.10,
        max_total_correction=0.75,
        loop_edges=2,
        objective_before=0.12,
        objective_after=0.08,
        loop_residual_before=0.30,
        loop_residual_after=0.20,
        loop_gap_before=7.33,
        loop_gap_after=7.31,
    )

    assert accepted is True
    assert reasons == ()


def test_refinement_rejects_worse_objective_loop_residual_and_gap():
    accepted, reasons = evaluate_refinement_acceptance(
        max_correction=0.2334,
        max_total_correction=0.75,
        loop_edges=2,
        objective_before=0.12,
        objective_after=0.13,
        loop_residual_before=0.30,
        loop_residual_after=0.32,
        loop_gap_before=7.33,
        loop_gap_after=7.44,
    )

    assert accepted is False
    assert any("objective" in reason for reason in reasons)
    assert any("loop residual" in reason for reason in reasons)
    assert any("gap worsened" in reason for reason in reasons)


def test_refinement_without_a_loop_edge_keeps_raw_arkit_trajectory():
    accepted, reasons = evaluate_refinement_acceptance(
        max_correction=0.0,
        max_total_correction=0.75,
        loop_edges=0,
        objective_before=0.0,
        objective_after=0.0,
        loop_residual_before=None,
        loop_residual_after=None,
        loop_gap_before=0.0,
        loop_gap_after=0.0,
    )

    assert accepted is False
    assert any("no loop-closure" in reason for reason in reasons)
