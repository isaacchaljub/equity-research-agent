import pytest

from pipeline.engine import ResilientChat


class _Fail:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        raise RuntimeError("boom")


class _Ok:
    def __init__(self, label):
        self.label = label
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return self.label


def test_primary_success_skips_backup():
    primary, backup = _Ok("primary"), _Ok("backup")
    chat = ResilientChat(primary, backup, retry_waits=(1, 2))
    assert chat.invoke([]) == "primary"
    assert primary.calls == 1
    assert backup.calls == 0


def test_retries_primary_then_falls_back():
    primary, backup = _Fail(), _Ok("backup")
    chat = ResilientChat(primary, backup, retry_waits=(1, 1))
    assert chat.invoke([]) == "backup"
    assert primary.calls == 3


def test_no_backup_raises_after_exhaustion():
    primary = _Fail()
    chat = ResilientChat(primary, None, retry_waits=())
    with pytest.raises(RuntimeError):
        chat.invoke([])
    assert primary.calls == 1


def test_fallback_is_sticky_across_calls():
    primary, backup = _Fail(), _Ok("backup")
    chat = ResilientChat(primary, backup, retry_waits=())
    assert chat.invoke([]) == "backup"
    assert chat.invoke([]) == "backup"
    assert primary.calls == 1


def test_backup_gets_its_own_retries():
    class _FailTwiceThenOk:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("stochastic")
            return "backup-ok"

    backup = _FailTwiceThenOk()
    chat = ResilientChat(_Fail(), backup, retry_waits=(), backup_retries=2)
    assert chat.invoke([]) == "backup-ok"
    assert backup.calls == 3
