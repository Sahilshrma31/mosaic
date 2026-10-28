"""Tests for memory.py. Each test gets its own empty SQLite file (temp_memory fixture)."""


def test_history_round_trip(temp_memory):
    temp_memory.add_message("chat-1", "user", "When is my Razorpay interview?")
    temp_memory.add_message("chat-1", "assistant", "28 Oct, 3 PM.")
    assert temp_memory.recent_history("chat-1") == [
        ("user", "When is my Razorpay interview?"),
        ("assistant", "28 Oct, 3 PM."),
    ]


def test_conversations_are_kept_separate(temp_memory):
    temp_memory.add_message("chat-1", "user", "When is my Razorpay interview?")
    assert temp_memory.recent_history("chat-2") == []


def test_only_the_last_turns_are_replayed(temp_memory):
    turns = temp_memory.HISTORY_TURNS + 2
    for i in range(turns):
        temp_memory.add_message("chat-1", "user", f"q{i}")
        temp_memory.add_message("chat-1", "assistant", f"a{i}")

    history = temp_memory.recent_history("chat-1")
    assert len(history) == temp_memory.HISTORY_TURNS * 2
    assert history[0] == ("user", "q2")  # the two oldest turns were left out
    assert history[-1] == ("assistant", f"a{turns - 1}")


def test_duplicate_fact_is_saved_once(temp_memory):
    assert temp_memory.remember("Prefers backend roles") is True
    assert temp_memory.remember("Prefers backend roles") is False
    assert temp_memory.all_facts() == ["Prefers backend roles"]


def test_fact_whitespace_is_trimmed(temp_memory):
    temp_memory.remember("  Google is my dream company  ")
    assert temp_memory.all_facts() == ["Google is my dream company"]
