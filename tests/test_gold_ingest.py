import json
import sqlite3
import pytest
from src.gold_ingest import load_spider_gold, load_bird_gold, GoldRecord, build_message_list
from src.schema_cache import compile_schema_cache


@pytest.fixture
def tmp_db_with_song(tmp_path):
    db_path = tmp_path / "concert_singer.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
        INSERT INTO singer VALUES (1, 'Alice', 30);
    """)
    conn.commit()
    conn.close()
    return tmp_path, str(db_path)


@pytest.fixture
def spider_json_file(tmp_path, tmp_db_with_song):
    db_dir, db_path = tmp_db_with_song
    data = [
        {
            "db_id": "concert_singer",
            "query": "SELECT name FROM singer WHERE age > 25",
            "query_toks": ["SELECT", "name", "FROM", "singer", "WHERE", "age", ">", "25"],
            "query_toks_no_value": ["select", "name", "from", "singer", "where", "age", ">", "value"],
            "question": "What are the names of all singers older than 25?",
            "question_toks": ["What", "are", "the", "names", "..."],
            "sql": {"select": [], "from": {}, "where": []},
        },
        {
            "db_id": "concert_singer",
            "query": "SELECT count(*) FROM singer",
            "query_toks": ["SELECT", "count", "(", "*", ")", "FROM", "singer"],
            "query_toks_no_value": ["select", "count", "(", "*", ")", "from", "singer"],
            "question": "How many singers are there?",
            "question_toks": ["How", "many", "..."],
            "sql": {"select": [], "from": {}, "where": []},
        },
    ]
    jpath = tmp_path / "train_spider.json"
    jpath.write_text(json.dumps(data))
    return str(jpath)


@pytest.fixture
def bird_json_file(tmp_path, tmp_db_with_song):
    db_dir, db_path = tmp_db_with_song
    data = [
        {
            "question_id": 0,
            "db_id": "concert_singer",
            "question": "What is the age of singer Alice?",
            "evidence": "",
            "SQL": "SELECT age FROM singer WHERE name = 'Alice'",
            "difficulty": "simple",
        }
    ]
    jpath = tmp_path / "dev.json"
    jpath.write_text(json.dumps(data))
    return str(jpath)


# --- load_spider_gold ---

def test_load_spider_gold_returns_gold_records(spider_json_file):
    records = load_spider_gold(spider_json_file)
    assert len(records) == 2
    assert all(isinstance(r, GoldRecord) for r in records)


def test_spider_field_names_mapped_correctly(spider_json_file):
    records = load_spider_gold(spider_json_file)
    r = records[0]
    assert r.db_id == "concert_singer"
    assert r.question == "What are the names of all singers older than 25?"
    assert r.gold_sql == "SELECT name FROM singer WHERE age > 25"
    assert r.source == "spider"
    assert r.difficulty is None  # Spider train_spider has no difficulty field


# --- load_bird_gold ---

def test_load_bird_gold_returns_gold_records(bird_json_file):
    records = load_bird_gold(bird_json_file)
    assert len(records) == 1
    assert isinstance(records[0], GoldRecord)


def test_bird_field_names_mapped_correctly(bird_json_file):
    records = load_bird_gold(bird_json_file)
    r = records[0]
    assert r.db_id == "concert_singer"
    assert r.question == "What is the age of singer Alice?"
    assert r.gold_sql == "SELECT age FROM singer WHERE name = 'Alice'"
    assert r.source == "bird"
    assert r.difficulty == "simple"


# --- build_message_list ---

def test_build_message_list_structure(spider_json_file, tmp_db_with_song):
    db_dir, db_path = tmp_db_with_song
    records = load_spider_gold(spider_json_file)
    cache = compile_schema_cache(db_path)
    messages = build_message_list(records[0], cache)
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_message_list_content(spider_json_file, tmp_db_with_song):
    db_dir, db_path = tmp_db_with_song
    records = load_spider_gold(spider_json_file)
    cache = compile_schema_cache(db_path)
    messages = build_message_list(records[0], cache)
    # System = JSON-serialized schema cache
    system_obj = json.loads(messages[0]["content"])
    assert system_obj["db_id"] == "concert_singer"
    # User = the question
    assert messages[1]["content"] == "What are the names of all singers older than 25?"
    # Assistant = gold SQL
    assert messages[2]["content"] == "SELECT name FROM singer WHERE age > 25"


def test_gold_record_to_example_dict(spider_json_file, tmp_db_with_song):
    db_dir, db_path = tmp_db_with_song
    records = load_spider_gold(spider_json_file)
    cache = compile_schema_cache(db_path)
    messages = build_message_list(records[0], cache)
    example = {
        "db_id": records[0].db_id,
        "source": records[0].source,
        "difficulty": records[0].difficulty,
        "type": "clean",
        "query_types": [],
        "messages": messages,
    }
    assert example["type"] == "clean"
    assert example["source"] == "spider"
    serialized = json.dumps(example)
    assert "concert_singer" in serialized
