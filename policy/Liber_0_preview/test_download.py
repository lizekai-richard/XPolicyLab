import hashlib
import json
import sys
from pathlib import Path

import huggingface_hub
import pytest

from . import download_checkpoint


@pytest.mark.parametrize('broken', [None, 'model.pt', 'context/model.safetensors', 'context/tokenizer.json'])
def test_complete_release(monkeypatch, tmp_path, broken):
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        if len(calls) > 1:
            return
        root = Path(kwargs['local_dir'])
        hashes = {}
        for name in kwargs['allow_patterns']:
            if name in ('manifest.json', 'SHA256SUMS'):
                continue
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = name.encode()
            path.write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        (root / 'manifest.json').write_text(json.dumps({'sha256': hashes}))
        if broken:
            (root / broken).write_bytes(b'corrupt')

    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot)
    monkeypatch.setattr(sys, 'argv', ['download_checkpoint.py', '--destination', str(tmp_path / 'checkpoint'),
                                     '--assets-dir', str(tmp_path / 'assets')])
    if broken:
        with pytest.raises(ValueError, match=broken):
            download_checkpoint.main()
        assert len(calls) == 1
    else:
        download_checkpoint.main()
        assert len(calls) == 2
        assert calls[1]['repo_id'] == 'HiDream-ai/HiDream-O1-Image'
    assert calls[0]['repo_id'] == 'LiberAI/Liber0-Preview-Robodojo'
    assert calls[0]['revision'] == '2773a04e3b8f09096ec7beb6105bd6ed80c91da8'
    assert sum(name.startswith('context/') for name in calls[0]['allow_patterns']) == 12
    assert 'context/model.safetensors' in calls[0]['allow_patterns']
    assert 'context/export-provenance.json' not in calls[0]['allow_patterns']
