import pytest
from src.diversity import DiversityTracker, QUERY_TYPES, CONVERGENCE_TARGET


def _make_example(db_id, sql, difficulty="easy"):
    return {
        "db_id": db_id,
        "source": "spider",
        "difficulty": difficulty,
        "type": "clean",
        "query_types": [],
        "messages": [
            {"role": "system", "content": "{}"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": sql},
        ],
    }


def test_tracker_starts_empty():
    t = DiversityTracker()
    assert t.total == 0
    assert t.per_db == {}
    assert t.query_type_counts == {qt: 0 for qt in QUERY_TYPES}


def test_add_example_increments_total():
    t = DiversityTracker()
    t.add(_make_example("concert_singer", "SELECT name FROM singer"))
    assert t.total == 1


def test_per_db_count():
    t = DiversityTracker()
    t.add(_make_example("concert_singer", "SELECT name FROM singer"))
    t.add(_make_example("concert_singer", "SELECT count(*) FROM singer"))
    t.add(_make_example("debit_card_specializing", "SELECT id FROM t"))
    assert t.per_db["concert_singer"] == 2
    assert t.per_db["debit_card_specializing"] == 1


def test_query_type_join_detected():
    t = DiversityTracker()
    t.add(_make_example("db1", "SELECT a.name FROM singer a JOIN concert c ON a.id = c.singer_id"))
    assert t.query_type_counts["join"] >= 1


def test_query_type_aggregate_detected():
    t = DiversityTracker()
    t.add(_make_example("db1", "SELECT count(*) FROM singer"))
    assert t.query_type_counts["aggregate"] >= 1


def test_query_type_group_by_detected():
    t = DiversityTracker()
    t.add(_make_example("db1", "SELECT name, count(*) FROM singer GROUP BY name"))
    assert t.query_type_counts["group_by"] >= 1


def test_query_type_order_by_detected():
    t = DiversityTracker()
    t.add(_make_example("db1", "SELECT name FROM singer ORDER BY age DESC"))
    assert t.query_type_counts["order_by"] >= 1


def test_query_type_subquery_detected():
    t = DiversityTracker()
    t.add(_make_example("db1", "SELECT name FROM singer WHERE id IN (SELECT singer_id FROM concert)"))
    assert t.query_type_counts["subquery"] >= 1


def test_query_type_limit_detected():
    t = DiversityTracker()
    t.add(_make_example("db1", "SELECT name FROM singer LIMIT 5"))
    assert t.query_type_counts["limit"] >= 1


def test_difficulty_counts():
    t = DiversityTracker()
    t.add(_make_example("db1", "SELECT 1", difficulty="easy"))
    t.add(_make_example("db1", "SELECT 1", difficulty="medium"))
    t.add(_make_example("db1", "SELECT 1", difficulty="hard"))
    assert t.difficulty_counts["easy"] == 1
    assert t.difficulty_counts["medium"] == 1
    assert t.difficulty_counts["hard"] == 1


def test_convergence_not_met_when_total_below_target():
    t = DiversityTracker()
    for _ in range(100):
        t.add(_make_example("db1", "SELECT name FROM singer JOIN concert ON singer.singer_id = concert.singer_id GROUP BY name ORDER BY count(*) DESC LIMIT 5 WHERE name IN (SELECT name FROM singer)"))
    met, report = t.convergence_check()
    assert met is False
    assert "total" in report


def test_convergence_report_lists_short_buckets():
    t = DiversityTracker()
    # No window functions added — bucket should be reported short
    t.add(_make_example("db1", "SELECT name FROM singer", difficulty="easy"))
    met, report = t.convergence_check()
    assert met is False
    assert "window" in report.lower()


def test_add_updates_example_query_types_inplace():
    t = DiversityTracker()
    ex = _make_example("db1", "SELECT count(*) FROM singer GROUP BY name")
    t.add(ex)
    assert "aggregate" in ex["query_types"]
    assert "group_by" in ex["query_types"]
