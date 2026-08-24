from src.data.loader import load_multi_domain


def test_multi_domain_loader():
    dataset = load_multi_domain(math_ratio=0.7, n_total=20)
    assert len(dataset) == 20

    # Hubi in columns-ku midaysan yihiin
    assert set(dataset.column_names) == {"prompt", "answer", "test_list", "domain"}

    domains = [item["domain"] for item in dataset]
    assert "math" in domains
    assert "code" in domains