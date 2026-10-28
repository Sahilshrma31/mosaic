"""Tests for EvidenceRegistry: numbering tool results and validating the answer's [n] citations."""

from agent import EvidenceRegistry


def mail(mail_id: str, title: str) -> dict:
    return {"source": "gmail", "id": mail_id, "title": title, "url": f"https://mail/{mail_id}", "date": "2026-10-20"}


def cites(tagged: dict) -> list[int]:
    return [item["cite"] for item in tagged["result"]]


def test_items_are_numbered_in_order():
    registry = EvidenceRegistry()
    tagged = registry.tag({"result": [mail("msg_003", "Razorpay Round 1"), mail("msg_004", "Re: Razorpay")]})
    assert cites(tagged) == [1, 2]


def test_same_mail_keeps_its_number_across_searches():
    registry = EvidenceRegistry()
    registry.tag({"result": [mail("msg_003", "Razorpay Round 1"), mail("msg_004", "Re: Razorpay")]})
    tagged = registry.tag({"result": [mail("msg_004", "Re: Razorpay"), mail("msg_005", "Swiggy rejection")]})
    assert cites(tagged) == [2, 3]


def test_error_result_passes_through_untouched():
    assert EvidenceRegistry().tag({"error": "Notion is down"}) == {"error": "Notion is down"}


def test_invented_citation_is_dropped():
    registry = EvidenceRegistry()
    registry.tag({"result": [mail("msg_005", "Swiggy rejection")]})
    answer, sources = registry.finalize("Swiggy rejected you [1]. Interview on 2 Nov [7].")
    assert answer == "Swiggy rejected you [1]. Interview on 2 Nov."
    assert [source["title"] for source in sources] == ["Swiggy rejection"]


def test_citations_are_renumbered_in_reading_order():
    registry = EvidenceRegistry()
    registry.tag({"result": [mail("msg_001", "A"), mail("msg_002", "B"), mail("msg_003", "C")]})
    answer, sources = registry.finalize("First [3]. Second [1].")
    assert answer == "First [1]. Second [2]."
    assert [(source["n"], source["title"]) for source in sources] == [(1, "C"), (2, "A")]


def test_comma_list_is_split_and_deduplicated():
    registry = EvidenceRegistry()
    registry.tag({"result": [mail("msg_001", "A"), mail("msg_002", "B")]})
    answer, _ = registry.finalize("Both say so [1, 2, 2].")
    assert answer == "Both say so [1][2]."


def test_sources_come_from_the_registry_not_the_model():
    registry = EvidenceRegistry()
    registry.tag({"result": [mail("msg_005", "Swiggy rejection")]})
    _, sources = registry.finalize("Rejected [1].")
    assert sources == [
        {"n": 1, "source": "gmail", "title": "Swiggy rejection", "url": "https://mail/msg_005", "date": "2026-10-20"}
    ]
