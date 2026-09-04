"""
test_condenser.py
==================

WHY THIS FILE EXISTS: condense_question() is the fix for the "retrieval
doesn't understand conversation history" gap (see condenser.py's module
docstring) — and it has a real behavioral branch worth locking down with a
test: it must NOT call Ollama at all when there's no history, since that's
what keeps the very first message of every conversation just as fast as it
was before conversational memory existed. Mocking requests.post here lets
us assert that non-call happened, which a real network call never could
(you can't prove an absence by making the call anyway).
"""

from app.rag.condenser import condense_question


def test_condense_question_with_no_history_returns_the_question_unchanged_and_skips_ollama(mocker):
    mock_post = mocker.patch("app.rag.condenser.requests.post")

    result = condense_question(history=[], question="What are the deadlines?")

    assert result == "What are the deadlines?"
    mock_post.assert_not_called()


def test_condense_question_with_history_calls_ollama_and_returns_the_rewrite(mocker):
    mock_response = mocker.Mock()
    mock_response.json.return_value = {
        "message": {"content": "What are the deadlines in the termination clauses?"}
    }
    mock_post = mocker.patch("app.rag.condenser.requests.post", return_value=mock_response)

    history = [
        {"role": "user", "content": "What are the termination clauses?"},
        {"role": "assistant", "content": "30 days written notice is required."},
    ]
    result = condense_question(history=history, question="What about the deadlines mentioned there?")

    assert result == "What are the deadlines in the termination clauses?"
    mock_post.assert_called_once()

    # The history turns should actually be in the request sent to Ollama —
    # otherwise there'd be nothing for the model to condense AGAINST.
    _, kwargs = mock_post.call_args
    sent_messages = kwargs["json"]["messages"]
    assert {"role": "user", "content": "What are the termination clauses?"} in sent_messages
