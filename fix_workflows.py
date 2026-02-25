"""Rewrites the Validate models steps in both workflow files to call validate_models.py."""
import re
import pathlib


REPLACEMENT = (
    '      - name: Validate models\n'
    '        run: |\n'
    '          echo "Validating trained models..."\n'
    '          python validate_models.py\n'
)

PATTERN = re.compile(
    r'      - name: Validate models\n'
    r'        run: \|\n'
    r'          echo "Validating trained models\.\.\."\n'
    r'          python -c ".*?          "\n',
    re.DOTALL,
)

for filename in [
    '.github/workflows/data-pipeline.yml',
    '.github/workflows/nightly-pipeline.yml',
]:
    path = pathlib.Path(filename)
    text = path.read_text(encoding='utf-8')
    new_text, count = PATTERN.subn(REPLACEMENT, text)
    if count == 0:
        print(f'WARNING: no matches found in {filename}')
    else:
        path.write_text(new_text, encoding='utf-8')
        print(f'Fixed {count} block(s) in {filename}')
