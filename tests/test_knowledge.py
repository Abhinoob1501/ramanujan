from ramanujan.memory.knowledge import HashingEmbedder, KnowledgeBase, format_for_prompt


def make_kb(tmp_path):
    return KnowledgeBase(tmp_path / "knowledge.db", embedder=HashingEmbedder())


def seed(kb):
    kb.add_insight(
        run_id="run1", task_name="churn-tabular",
        hypothesis="Gradient boosting beats linear models on tabular churn data",
        approach="HistGradientBoostingClassifier with 5-fold CV",
        insight="Boosted trees dominated on this small tabular dataset; scaling irrelevant for trees.",
        metric_name="roc_auc", metric_value=0.87,
    )
    kb.add_insight(
        run_id="run2", task_name="cifar-images",
        hypothesis="Data augmentation improves CNN generalization on small images",
        approach="ResNet with random crops and flips",
        insight="Augmentation added 2 points of accuracy on image classification with CNNs.",
        metric_name="accuracy", metric_value=0.91,
    )


def test_retrieval_ranks_by_relevance(tmp_path):
    kb = make_kb(tmp_path)
    seed(kb)
    items = kb.retrieve("tabular dataset with gradient boosting trees churn", top_k=2)
    assert len(items) == 2
    assert items[0].task_name == "churn-tabular"
    assert items[0].similarity > items[1].similarity


def test_exclude_run_filters_own_insights(tmp_path):
    kb = make_kb(tmp_path)
    seed(kb)
    items = kb.retrieve("gradient boosting tabular churn", top_k=5, exclude_run="run1")
    assert all(item.run_id != "run1" for item in items)


def test_empty_kb_and_prompt_formatting(tmp_path):
    kb = make_kb(tmp_path)
    assert kb.retrieve("anything at all") == []
    assert format_for_prompt([]) == ""

    seed(kb)
    block = format_for_prompt(kb.retrieve("tabular boosting churn", top_k=1))
    assert "PRIOR KNOWLEDGE" in block
    assert "Boosted trees dominated" in block
    assert "0.8700" in block


def test_count(tmp_path):
    kb = make_kb(tmp_path)
    assert kb.count() == 0
    seed(kb)
    assert kb.count() == 2
