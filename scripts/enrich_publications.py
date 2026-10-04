"""Enrich records from SIT's public repository and verify remaining DOIs."""
from collect_publications import ROOT, fetch, download, norm, bibtex
from pathlib import Path
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import subprocess
import time
from urllib.parse import quote


def enrich():
    path = ROOT / 'data/publications.json'
    pubs = json.loads(path.read_text())
    repository = json.loads(Path('/tmp/pc-ng-research-cache/figshare-full.json').read_text())
    older = next(a for a in repository if a['id'] == 24209607)
    if not any(p['id'] == 'c54' for p in pubs):
        pubs.append({'id': 'c54', 'authors': 'Pai Chet Ng', 'title': older['title'], 'year': 2015, 'reference': 'Pai Chet Ng (2015). ' + older['title'] + '. 2015 IEEE 12th International Conference on Networking, Sensing and Control, pp. 592–596.', 'venue_note': '2015 IEEE 12th International Conference on Networking, Sensing and Control, pp. 592–596.', 'source': older['url_public_html'], 'kind': 'conference', 'status': 'Published', 'doi': older['doi'], 'pdf': '', 'resources': [], 'attempts': []})
    # Verified title-level repository matches, not guesses based on filename.
    for p in pubs:
        matches = [(SequenceMatcher(None, norm(p['title']), norm(a.get('title', ''))).ratio(), a) for a in repository]
        score, a = max(matches, key=lambda pair: pair[0])
        if score >= .93:
            p['repository_source'] = a['url_public_html']
            p['repository_article_id'] = a['id']
            doi = a.get('doi', '')
            if doi and not doi.lower().startswith('10.25447'):
                p['doi'] = doi
                p['publisher_source'] = 'https://doi.org/' + doi
                p['metadata_source'] = 'https://api.figshare.com/v2/articles/' + str(a['id'])
            p['pdf_candidates'] = list(dict.fromkeys(p.get('pdf_candidates', []) + [f['download_url'] for f in a.get('files', []) if f['name'].lower().endswith('.pdf')]))
            if not any(r['url'] == a['url_public_html'] for r in p['resources']):
                p['resources'].append({'label': 'Repository', 'url': a['url_public_html']})
    # Existing paper title pages include authoritative DOI strings.
    for p in pubs:
        if not p['doi'] and p['pdf']:
            result = subprocess.run(['pdftotext', '-f', '1', '-l', '2', str(ROOT / p['pdf'].lstrip('/')), '-'], capture_output=True, text=True)
            found = re.findall(r'10\.\d{4,9}/[a-zA-Z0-9.()/\-]+', result.stdout)
            if found:
                p['doi'] = found[0].rstrip('.,)')
                p['metadata_source'] = p['pdf'] + ' (paper title page)'
                p['publisher_source'] = 'https://doi.org/' + p['doi']
    for p in pubs:
        if p['id'] == 'c21':
            p['doi'] = '10.1109/ICDCSW63273.2025.00130'
            p['publisher_source'] = 'https://doi.org/' + p['doi']
        if p['id'] == 'c15':
            p['pdf_candidates'] = ['https://arxiv.org/pdf/2606.08437v2'] + p.get('pdf_candidates', [])
        if p['id'] == 'c2':
            p['pdf_candidates'] = ['https://arxiv.org/pdf/2601.18386v1'] + p.get('pdf_candidates', [])
    path.write_text(json.dumps(pubs, indent=2, ensure_ascii=False) + '\n')
    # A paced lookup respects Crossref throttling. No retry loops on 429s.
    for p in pubs:
        if p['doi'] or p['kind'] == 'dataset':
            continue
        time.sleep(1.5)
        try:
            url = 'https://api.crossref.org/works?query.bibliographic=' + quote(p['title'] + ' Pai Chet Ng') + '&rows=3'
            works = json.loads(fetch(url))['message']['items']
            for work in works:
                title = work.get('title', [''])[0]
                names = ' '.join(a.get('family', '') for a in work.get('author', []))
                if SequenceMatcher(None, norm(title), norm(p['title'])).ratio() >= .93 and 'ng' in names.lower().split():
                    p['doi'] = work['DOI']
                    p['publisher_source'] = 'https://doi.org/' + work['DOI']
                    p['metadata_source'] = url
                    p['crossref'] = {k: work[k] for k in ['container-title', 'volume', 'issue', 'page', 'publisher', 'type', 'link'] if k in work}
                    break
            print('DOI', p['id'], p['doi'] or 'not found', flush=True)
        except Exception as exc:
            print('DOI', p['id'], str(exc)[:120], flush=True)
    def save(p):
        if p['pdf'] or p['kind'] == 'dataset':
            return p
        dest = ROOT / 'asset/paper' / (p['id'] + '-' + re.sub(r'[^a-z0-9]+', '-', p['title'].lower()).strip('-')[:100] + '.pdf')
        for u in p.get('pdf_candidates', []):
            if download(u, dest, p):
                break
        return p
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(save, p) for p in pubs]
        complete = {}
        for f in as_completed(futures):
            p = f.result(); complete[p['id']] = p
            print('Paper', p['id'], 'saved' if p['pdf'] else 'no local PDF', flush=True)
    pubs = [complete[p['id']] for p in pubs]
    for p in pubs:
        p['bibtex'] = bibtex(p)
    path.write_text(json.dumps(pubs, indent=2, ensure_ascii=False) + '\n')
    print('Papers', sum(bool(p['pdf']) for p in pubs), 'of', sum(p['kind'] != 'dataset' for p in pubs), 'DOIs', sum(bool(p['doi']) for p in pubs), flush=True)


if __name__ == '__main__':
    enrich()
