import importlib.util
from pathlib import Path


def test_public_tree_blocks_journal_sidecars_and_learned_models(monkeypatch):
    path = Path(__file__).parents[1] / 'scripts/check_public_tree.py'
    spec = importlib.util.spec_from_file_location('public_tree', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ['bot.sqlite', 'bot.sqlite3-wal', 'bot.sqlite-shm', 'bot.db-journal',
                 'entry_model.json', 'frozen_selection.json']:
        monkeypatch.setattr(module.subprocess, 'check_output', lambda *_, file=name, **__: (file + '\0').encode())
        assert module.inspect_index() == 1
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *_, **__: b'src/wonyotti_fr/engine.py\0tests/fixtures/sample.csv\0')
    assert module.inspect_index() == 0
