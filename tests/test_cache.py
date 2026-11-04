from pulse_qc.cache import Cache, code_hash, config_hash, make_key


def test_hit_and_miss(tmp_path):
    c = Cache(tmp_path)
    calls = []

    def compute():
        calls.append(1)
        return {"x": 1}

    k = make_key("s", input="a", config="b", code="c")
    assert c.get_or_compute("s", k, compute) == {"x": 1}
    assert c.get_or_compute("s", k, compute) == {"x": 1}
    assert len(calls) == 1
    assert c.stage_summary() == {"s": {"hits": 1, "misses": 1}}


def test_key_changes_with_any_part():
    base = dict(input="a", config="b", code="c")
    k = make_key("s", **base)
    assert make_key("s", **{**base, "input": "z"}) != k
    assert make_key("s", **{**base, "config": "z"}) != k
    assert make_key("s", **{**base, "code": "z"}) != k
    assert make_key("t", **base) != k
    assert make_key("s", **base) == k


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_code_hash_tracks_source():
    def f():
        return 1

    def g():
        return 2

    assert code_hash(f) == code_hash(f)
    assert code_hash(f) != code_hash(g)
