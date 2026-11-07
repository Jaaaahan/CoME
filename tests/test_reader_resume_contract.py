from run_memix_answers_from_retrieval import resume_contract_matches


def test_legacy_failure_can_be_retried_without_mutating_contract():
    stored={'model':'qwen','checkpoint_state_sha256':'state'}
    expected={**stored,'retrieval_row_sha256':'retrieval'}
    assert resume_contract_matches(stored,expected,retry_error=True)
    assert 'retrieval_row_sha256' not in stored
    assert not resume_contract_matches(stored,expected,retry_error=False)


def test_retry_never_accepts_other_contract_changes():
    expected={'model':'qwen','checkpoint_state_sha256':'state','retrieval_row_sha256':'retrieval'}
    assert not resume_contract_matches({**expected,'retrieval_row_sha256':'old'},expected,retry_error=True)
    assert not resume_contract_matches({'model':'other','checkpoint_state_sha256':'state'},expected,retry_error=True)
    assert not resume_contract_matches({'model':'qwen'},expected,retry_error=True)
    assert resume_contract_matches(expected,expected,retry_error=False)
