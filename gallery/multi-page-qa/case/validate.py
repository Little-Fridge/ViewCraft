"""Offline package checks; never instantiate a model client or call an API."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import runpy
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SKILL = HERE.parent
ROOT = SKILL.parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from PIL import Image
from viewcraft.views.config import Coordinates
from viewcraft.views.selection import eval_selection_expr
from viewcraft.evolution.chain import Chain


def main():
    coords = Coordinates.read_snapshot(SKILL)
    Chain._assert_expressions_parse(None, SimpleNamespace(coords=coords))
    metadata = yaml.safe_load((SKILL / 'skill.yaml').read_text())
    assert metadata['id'] == 'multi-page-qa'
    assert metadata['input_type'] == 'images'
    assert metadata['budget'] == {'observations': 6, 'max_images_per_request': 6}
    assert coords.psi.protocol_contract == ''
    assert coords.psi.max_images_per_request == 6
    body = (SKILL / 'SKILL.md').read_text().split('---', 2)[2]
    word_count = len(body.split())
    assert 300 <= word_count <= 600
    request = json.loads((HERE / 'request.json').read_text())
    expected = json.loads((HERE / 'expected.json').read_text())
    assert set(request) == {'question', 'images', 'options'}
    assert len(request['images']) == 4
    assert len(set(request['images'])) == 4
    selected = eval_selection_expr(coords.phi.select_expr, 4, {}, 6)
    assert selected == [0, 1, 2, 3]
    images = {}
    for name in request['images']:
        path = HERE / name
        assert path.resolve().parent == HERE
        with Image.open(path) as im:
            assert im.size == (1200, 1600) and im.mode == 'RGB'
            images[name] = list(im.size)
            im.verify()
    assert expected == {'answer': str(1500 - 1240), 'match': 'exact'}
    artifacts = request['images'] + ['preview.png', 'request.json', 'expected.json']
    def hashes():
        return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in artifacts}
    before = hashes()
    runpy.run_path(str(HERE / 'generate.py'), run_name='__main__')
    assert hashes() == before, 'Generation is not reproducible'
    (SKILL / 'CONFIG_HASH').write_text(coords.config_hash + '\n')
    result = {
        'status': 'passed', 'mode': 'offline_only', 'model_called': False,
        'checks': ['Coordinates.read_snapshot', 'Chain._assert_expressions_parse',
                   'metadata_and_budget', 'skill_word_count', 'public_request_schema',
                   'full_four_page_selection', 'PNG_integrity_and_dimensions',
                   'expected_arithmetic', 'repeat_generation_sha256'],
        'skill_body_words': word_count, 'selected_indices': selected,
        'images': images, 'sha256': before, 'config_hash': coords.config_hash,
        'expected_answer': expected['answer'],
        'model_validation': 'Pending main-agent genuine run; no trajectory fabricated.'
    }
    (HERE / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
