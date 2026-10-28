"""Tests for email ranking (tools.rank) through the mock search over mock_data/emails.json."""

import pytest

import tools


def top_ids(query: str, max_results: int = 5) -> list[str]:
    return [mail["id"] for mail in tools.search_emails(query, max_results)]


def test_swiggy_rejection_survives_multi_company_search():
    """Regression: common words ("intern", "interview") used to fill the top 5 and push the
    Swiggy rejection (msg_005) out, so the agent reported Swiggy as still "Applied"."""
    assert "msg_005" in top_ids("swiggy microsoft atlassian google intern interview applying")


@pytest.mark.parametrize("mail_id", ["msg_005", "msg_013", "msg_017"])  # Swiggy, CRED, Groww
def test_every_company_in_a_combined_search_gets_a_result(mail_id):
    assert mail_id in top_ids("swiggy cred groww backend intern application")


def test_body_mention_does_not_take_a_companys_slot():
    """The Razorpay invite says "Google Meet" in its body; Google's slot must still go to a
    mail from Google itself (msg_011, Google Careers)."""
    assert "msg_011" in top_ids("swiggy microsoft atlassian google intern interview applying")


def test_extended_deadline_ranks_first():
    """25 Oct (msg_001) was extended to 30 Oct (msg_002): the newer mail must come first."""
    assert top_ids("dbms assignment deadline")[:2] == ["msg_002", "msg_001"]


def test_rejection_ranks_above_same_company_promo():
    results = top_ids("swiggy application")
    assert results.index("msg_005") < results.index("msg_006")  # rejection above the food promo


def test_buried_promotions_mail_is_found():
    assert top_ids("google test register")[0] == "msg_030"


def test_no_match_returns_empty_list():
    assert top_ids("xyzzy") == []


def test_max_results_is_respected():
    assert len(top_ids("intern", max_results=3)) == 3
