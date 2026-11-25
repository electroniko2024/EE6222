import json
from pathlib import Path
from pypdf import PdfReader

root = Path(__file__).resolve().parents[2]
md = (root / 'report_zh.md').read_text(encoding='utf-8')
results = json.loads((root / 'results/results.json').read_text())
rows = [line.split('|')[1:-1] for line in md.splitlines()
        if line.startswith(('| 原始特征', '| PCA', '| LDA'))]
assert len(rows) == 6
for row, expected in zip(rows, results['summary']):
    row = [x.strip() for x in row]
    assert row[1] == expected['classifier']
    assert int(row[2]) == expected['dimension']
    assert float(row[3]) == expected['parameter']
    assert row[4] == '{:.2f}%'.format(100*expected['validation_accuracy'])
    assert row[5] == '{:.2f}%'.format(100*expected['test_accuracy'])
    assert int(row[6]) == expected['errors']
pdf = PdfReader(root / 'output/pdf/MENGXIANGYU-G2607496E-中文说明报告.pdf')
assert len(pdf.pages) == 5
text = ''.join(page.extract_text() for page in pdf.pages)
assert all(value in text for value in ['MENG XIANGYU', 'G2607496E',
    '62.5%', '99.17%', '98.61%', '95.56%', '220', '书写者', 'Wilson'])
for page in pdf.pages:
    descriptors = [font.get_object().get('/FontDescriptor')
                   for font in page['/Resources']['/Font'].get_object().values()]
    assert any(value.get_object().get('/FontFile2') for value in descriptors if value)
print('PASS: six result rows, five pages, signature, key claims and embedded Chinese fonts.')
