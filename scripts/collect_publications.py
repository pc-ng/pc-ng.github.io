"""Collect verified public metadata and PDFs; preserve CV publication status.

Run explicitly when updating the bibliography. Network access is not needed
to build or serve the website. Failed downloads are recorded for the owner.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from urllib.parse import quote, urljoin
import hashlib
import json
import re
import shutil
import subprocess
import time
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path('/tmp/pc-ng-research-cache')
CACHE.mkdir(exist_ok=True)
HEADERS = {'User-Agent': 'PaiChetNg-AcademicPortfolio/1.0 (public bibliography review)'}


def fetch(url, accept=None):
    key = hashlib.sha256((url + str(accept)).encode()).hexdigest()
    cache = CACHE / key
    if cache.exists():
        return cache.read_bytes()
    headers = dict(HEADERS)
    if accept:
        headers['Accept'] = accept
    response = requests.get(url, timeout=(10, 35), headers=headers)
    response.raise_for_status()
    cache.write_bytes(response.content)
    return response.content


def norm(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


def metadata(p):
    # Keep bibliography requests paced even when invoked outside the main script.
    time.sleep(1.3)
    p.setdefault('resources', [])
    p.setdefault('attempts', [])
    source = p['source']
    direct_doi = source.split('doi.org/', 1)[1] if 'doi.org/' in source else ''
    if direct_doi.startswith('10.25447/'):
        p['repository_doi'] = direct_doi
        direct_doi = ''
    work = None
    try:
        if direct_doi:
            work = json.loads(fetch('https://api.crossref.org/works/' + quote(direct_doi, safe='')))['message']
        elif p['kind'] != 'dataset':
            url = 'https://api.crossref.org/works?query.title=' + quote(p['title']) + '&rows=4'
            candidates = json.loads(fetch(url))['message']['items']
            for candidate in candidates:
                title = candidate.get('title', [''])[0]
                score = SequenceMatcher(None, norm(p['title']), norm(title)).ratio()
                names = ' '.join(a.get('family', '') for a in candidate.get('author', []))
                if score >= .93 and 'ng' in names.lower().split():
                    work = candidate
                    break
        if work:
            p['doi'] = work['DOI']
            p['metadata_source'] = 'https://api.crossref.org/works/' + quote(work['DOI'], safe='')
            p['publisher_source'] = 'https://doi.org/' + work['DOI']
            p['crossref'] = {k: work[k] for k in ['author', 'container-title', 'volume', 'issue', 'page', 'article-number', 'publisher', 'type', 'link'] if k in work}
    except Exception as exc:
        p['attempts'].append({'url': 'Crossref title/DOI lookup', 'error': str(exc)[:180]})
    arxiv = re.search(r'arxiv.org/(?:abs|pdf)/(\d{4}\.\d{4,5})', source)
    if arxiv:
        p['arxiv'] = arxiv.group(1)
        if not p['doi']:
            p['doi'] = '10.48550/arXiv.' + p['arxiv']
    return p


def download(url, dest, p):
    try:
        raw = fetch(url)
        if not raw.lstrip().startswith(b'%PDF-'):
            raise ValueError('Response is not a PDF')
        dest.write_bytes(raw)
        check = subprocess.run(['pdfinfo', str(dest)], capture_output=True, text=True)
        if check.returncode:
            dest.unlink()
            raise ValueError('Invalid PDF')
        p['pdf_download_source'] = url
        p['pdf'] = '/' + str(dest.relative_to(ROOT))
        return True
    except Exception as exc:
        p['attempts'].append({'url': url, 'error': str(exc)[:200]})
        return False


def obtain_pdf(p):
    dest = ROOT / 'asset/paper' / (p['id'] + '-' + re.sub(r'[^a-z0-9]+', '-', p['title'].lower()).strip('-')[:100] + '.pdf')
    if dest.exists():
        p['pdf'] = '/' + str(dest.relative_to(ROOT))
        return p
    # Original author-hosted papers are preserved, copied under the new asset path.
    for f in (ROOT / 'zpublications').glob('*.pdf'):
        if p.get('legacy_pdf') == f.name:
            shutil.copyfile(f, dest)
            p['pdf'] = '/' + str(dest.relative_to(ROOT))
            p['pdf_download_source'] = 'Original website: /zpublications/' + f.name
            return p
    urls = list(p.get('pdf_candidates', []))
    if p.get('arxiv'):
        urls.append('https://arxiv.org/pdf/' + p['arxiv'])
    if p['source'].endswith('.pdf'):
        urls.append(p['source'])
    if p.get('repository_doi'):
        try:
            article_id = p['repository_doi'].rsplit('.', 1)[1]
            data = json.loads(fetch('https://api.figshare.com/v2/articles/' + article_id))
            for f in data.get('files', []):
                if f['name'].lower().endswith('.pdf'):
                    urls.append(f['download_url'])
            p['resources'].append({'label': 'Repository', 'url': data.get('url_public_html', p['source'])})
        except Exception as exc:
            p['attempts'].append({'url': 'Figshare repository', 'error': str(exc)[:150]})
    # Collect direct public PDF links from the provided conference/author page.
    if p['source'] and not any(h in p['source'] for h in ['dblp.org', 'doi.org', 'arxiv.org']) and not p['source'].endswith('.pdf'):
        try:
            soup = BeautifulSoup(fetch(p['source']), 'html.parser')
            for a in soup.find_all('a', href=True):
                href = urljoin(p['source'], a['href'])
                if '.pdf' in href.lower() and ('paper' in a.get_text().lower() or 'isca-archive' in href):
                    urls.append(href)
        except Exception as exc:
            p['attempts'].append({'url': p['source'], 'error': str(exc)[:150]})
    for link in p.get('crossref', {}).get('link', []):
        if link.get('content-type') == 'application/pdf':
            urls.append(link['URL'])
    for url in dict.fromkeys(urls):
        if download(url, dest, p):
            break
    return p


def bibtex(p):
    meta = p.get('crossref', {})
    fields = {'author': p['authors'].replace(', ', ' and '), 'title': p['title'], 'year': str(p['year'])}
    # Preserve the author's complete CV author list, including original ordering.
    kind = {'journal': 'article', 'conference': 'inproceedings', 'dataset': 'misc'}[p['kind']]
    if p['status'] == 'Preprint':
        kind = 'misc'
    container = meta.get('container-title', [])
    if container:
        fields['journal' if kind == 'article' else 'booktitle'] = container[0]
    elif kind in ['article', 'inproceedings']:
        fields['journal' if kind == 'article' else 'booktitle'] = p.get('bib_venue', p['venue_note'].split(';')[0])
    for key, target in [('volume', 'volume'), ('issue', 'number'), ('page', 'pages'), ('publisher', 'publisher')]:
        if meta.get(key):
            value = str(meta[key])
            fields[target] = re.sub(r'(?<=\d)[–-](?=\d)', '--', value) if target == 'pages' else value
    if 'pages' not in fields:
        page_range = re.search(r'(?:pp\.|:)\s*(\d+)[–-](\d+)', p['venue_note'])
        if page_range:
            fields['pages'] = page_range.group(1) + '--' + page_range.group(2)
    if p['doi']:
        fields['doi'] = p['doi']
    fields['url'] = p.get('publisher_source') or p['source']
    if p.get('arxiv') and (kind == 'misc' or p['doi'].startswith('10.48550')):
        fields.update(eprint=p['arxiv'], archivePrefix='arXiv')
    if p['status'] != 'Published':
        fields['note'] = p['status'] + ('. ' + p.get('bib_note', '') if p.get('bib_note') else '')
    key = 'Ng' + str(p['year']) + p['id'].upper()
    def escape(s):
        return s.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')
    return '@' + kind + '{' + key + ',\n' + ',\n'.join('  ' + k + ' = {' + escape(v) + '}' for k, v in fields.items() if v) + '\n}'


def main():
    path = ROOT / 'data/publications.json'
    pubs = json.loads(path.read_text())
    updates = {
        'c2': {'arxiv': '2601.18386'},
        'c27': {'title': 'Can Pretrained Face Verification Models Distinguish True Identity from Deepfakes?', 'bib_venue': 'BMVC Workshops 2024'},
        'c29': {'title': 'Hyperspectral Skin Vision Challenge: Can Your Camera See Beyond Your Skin?', 'bib_venue': 'ICASSP Workshops 2024'},
        'c40': {'arxiv': '2310.17911', 'source': 'https://proceedings.neurips.cc/paper_files/paper/2023/hash/4c0986bd04d747745beba3752bdf4d9d-Abstract-Datasets_and_Benchmarks.html', 'pdf_candidates': ['https://proceedings.neurips.cc/paper_files/paper/2023/file/4c0986bd04d747745beba3752bdf4d9d-Paper-Datasets_and_Benchmarks.pdf'], 'resources': [{'label': 'Project', 'url': 'https://hyper-skin-2023.github.io/'}, {'label': 'Code', 'url': 'https://github.com/hyperspectral-skin/Hyper-Skin-2023'}, {'label': 'Data access', 'url': 'https://hyperskinsiteapp--hyperskinwebapp.asia-east1.hosted.app/dataAccess'}, {'label': 'Slides', 'url': 'https://neurips.cc/media/neurips-2023/Slides/73523.pdf'}, {'label': 'Challenge', 'url': 'https://uoft-hyperskin.github.io/'}]},
        'c15': {'resources': [{'label': 'Code & data', 'url': 'https://github.com/X-Palm/X-Palm-2026'}]},
        'c14': {'resources': [{'label': 'Project', 'url': 'https://hyper-object.github.io/'}, {'label': 'Code', 'url': 'https://github.com/hyper-object/2026-ICASSP-SPGC'}, {'label': 'Data · Track 1', 'url': 'https://www.kaggle.com/competitions/2026-icassp-hyper-object-challenge-track-1'}, {'label': 'Data · Track 2', 'url': 'https://www.kaggle.com/competitions/2026-icassp-hyper-object-challenge-track-2'}]},
    }
    # File-to-paper mapping is verified using PDF title text, below.
    pdf_text = {}
    for f in (ROOT / 'zpublications').glob('*.pdf'):
        out = subprocess.run(['pdftotext', '-f', '1', '-l', '1', str(f), '-'], capture_output=True, text=True)
        pdf_text[f.name] = norm(out.stdout[:4000])
    for p in pubs:
        p.update(updates.get(p['id'], {}))
        for filename, text in pdf_text.items():
            if norm(p['title']) in text:
                p['legacy_pdf'] = filename
                break
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = {pool.submit(metadata, p): p['id'] for p in pubs}
        complete = {}
        for f in as_completed(futures):
            p = f.result(); complete[p['id']] = p
            print('metadata', p['id'], p['doi'] or 'no verified DOI', flush=True)
    pubs = [complete[p['id']] for p in pubs]
    path.write_text(json.dumps(pubs, indent=2, ensure_ascii=False) + '\n')
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(obtain_pdf, p): p['id'] for p in pubs if p['kind'] != 'dataset'}
        complete = {p['id']: p for p in pubs if p['kind'] == 'dataset'}
        for f in as_completed(futures):
            p = f.result(); complete[p['id']] = p
            print('PDF', p['id'], 'saved' if p['pdf'] else 'unavailable', flush=True)
    pubs = [complete[p['id']] for p in pubs]
    for p in pubs:
        p['bibtex'] = bibtex(p)
    path.write_text(json.dumps(pubs, indent=2, ensure_ascii=False) + '\n')
    print('Total:', len(pubs), 'PDFs:', sum(bool(p['pdf']) for p in pubs), 'DOIs:', sum(bool(p['doi']) for p in pubs), flush=True)


if __name__ == '__main__':
    main()
