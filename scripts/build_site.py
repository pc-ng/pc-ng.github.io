"""Build the static academic site with Python's standard library.

Edit data/publications.json and data/projects.json, then run this script.
All pages and references are readable without JavaScript or a build server.
"""
from pathlib import Path
from html import escape
from datetime import date
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PUBLICATIONS = json.loads((ROOT / 'data/publications.json').read_text())
# Supplied manuscripts use stable filenames and are picked up on the next build.
for paper in PUBLICATIONS:
    if paper['kind'] != 'dataset':
        supplied = ROOT / 'asset/paper' / (paper['id'] + '.pdf')
        if supplied.exists() and supplied.read_bytes().lstrip().startswith(b'%PDF-'):
            paper['pdf'] = '/asset/paper/' + supplied.name
            paper['pdf_version'] = 'Author-supplied manuscript'
PAPERS = {p['id']: p for p in PUBLICATIONS}
PROJECTS = json.loads((ROOT / 'data/projects.json').read_text())
SELECTED = ['c15', 'c7', 'c40', 'j2', 'j4', 'j6', 'j11', 'c2']
DESCRIPTION = 'Trustworthy AI for multimodal sensing in mobile and pervasive systems.'
SITE = 'https://pc-ng.github.io'


def e(value):
    return escape(str(value), quote=True)


def link(label, url, css=''):
    if not url:
        return ''
    external = ' target="_blank" rel="noopener noreferrer"' if url.startswith('https://') else ''
    return f'<a href="{e(url)}"' + (f' class="{e(css)}"' if css else '') + external + f'>{e(label)}</a>'


def page(title, path, content, current='', description=None):
    # Use consistent title case for section headings, preserving acronyms and notes.
    if current != 'Notes':
        small_words = {'and', 'at', 'for', 'in', 'of', 'on', 'the', 'to', 'with'}
        def section_heading(match):
            words = match.group(2).split(' ')
            words = [word if not word or (index and word in small_words) else word[0].upper() + word[1:] for index, word in enumerate(words)]
            return match.group(1) + ' '.join(words) + match.group(3)
        content = re.sub(r'(<h2\b[^>]*>)([^<]+)(</h2>)', section_heading, content)
    nav = [('Home', '/'), ('Research', '/research/'), ('Publications', '/publications/'), ('Teaching', '/teaching/'), ('Service', '/service/')]
    navigation = ''.join(f'<a href="{url}"' + (' aria-current="page"' if name == current else '') + f'>{name}</a>' for name, url in nav)
    person = {'@context': 'https://schema.org', '@type': 'Person', 'name': 'Pai Chet Ng', 'givenName': 'Pai Chet', 'familyName': 'Ng', 'jobTitle': 'Assistant Professor', 'url': SITE, 'affiliation': {'@type': 'CollegeOrUniversity', 'name': 'Singapore Institute of Technology'}, 'sameAs': ['https://orcid.org/0000-0001-9153-5411', 'https://scholar.google.com/citations?user=WBieghIAAAAJ'], 'knowsAbout': ['Trustworthy artificial intelligence', 'Multimodal sensing', 'Federated learning', 'Hyperspectral imaging', 'Mobile and pervasive systems']}
    full_title = title + ' · Pai Chet Ng' if title != 'Pai Chet Ng' else 'Pai Chet Ng · Trustworthy AI & Multimodal Sensing'
    meta_description = description or DESCRIPTION + ' Assistant Professor at Singapore Institute of Technology.'
    math_assets = ''
    if 'class="note-body"' in content and (r'\(' in content or r'\[' in content):
        math_assets = '<link rel="stylesheet" href="/asset/vendor/katex/katex.min.css"><script src="/asset/vendor/katex/katex.min.js" defer></script><script src="/asset/vendor/katex/auto-render.min.js" defer></script><script src="/asset/note-math.js" defer></script>'
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(full_title)}</title><meta name="description" content="{e(meta_description)}">
<link rel="canonical" href="{SITE}{path}"><meta property="og:type" content="website"><meta property="og:title" content="{e(full_title)}"><meta property="og:description" content="{e(meta_description)}"><meta property="og:url" content="{SITE}{path}">
<meta name="theme-color" content="#fbfaf7"><link rel="icon" href="/asset/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="/asset/site.css">{math_assets}<script src="/asset/site.js" defer></script><script type="application/ld+json">{json.dumps(person).replace('<', chr(92)+'u003c')}</script></head>
<body><a class="skip-link" href="#main">Skip to content</a><header class="site-header"><div class="wrap header-inner"><nav class="nav" aria-label="Main navigation">{navigation}</nav></div></header>
<main id="main" class="wrap">{content}</main><footer class="site-footer"><div class="wrap footer-inner"><span>© 2026 Pai Chet Ng · Singapore</span><div class="footer-links">{link('Email', 'mailto:paichet.ng@singaporetech.edu.sg')}{link('ORCID', 'https://orcid.org/0000-0001-9153-5411')}{link('Notes', '/notes/')}</div></div></footer></body></html>'''


def write(path, html):
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(line.rstrip() for line in html.splitlines()) + '\n')


def paper_url(p):
    return p.get('publisher_source') or (p['source'] if p['source'] else '')


def authors(text):
    return re.sub(r'Pai Chet Ng|P\. C\. Ng', lambda m: '<strong>' + m.group() + '</strong>', e(text))


def publication(p, compact=False, id_prefix=''):
    ident = p['id']
    title_link = link(p['title'], paper_url(p)) if paper_url(p) else e(p['title'])
    # The complete author list and full venue detail are retained from the CV.
    venue = p['venue_note'].replace(' Source', '').strip()
    resources = []
    if paper_url(p):
        label = 'Data' if p['kind'] == 'dataset' else ('Paper' if p.get('doi') or 'arxiv.org/' in paper_url(p) or '/rec/' in paper_url(p) or '.pdf' in paper_url(p) or 'isca-archive' in paper_url(p) or 'view_paper' in paper_url(p) or 'openreview.net/forum' in paper_url(p) else 'Programme')
        resources.append(link(label + ' ↗', paper_url(p)))
    if p['pdf']:
        resources.append(f'<a href="{e(p["pdf"])}" download>PDF ↓</a>')
    elif p['kind'] != 'dataset':
        resources.append('<span class="pdf-pending" aria-label="PDF not yet available">PDF pending</span>')
    resources.extend(link(r['label'] + (' ↗' if r['url'].startswith('https://') else ' ↓'), r['url']) for r in p.get('resources', []) if 'singaporetech.edu.sg' not in r['url'] and (r['label'] != 'Repository' or not compact))
    status = p['status']
    pill = f'<span class="status {"preprint" if status == "Preprint" else ""}">{e(status)}</span>' if status != 'Published' else ''
    doi_label = 'arXiv DOI' if p['doi'].lower().startswith('10.48550/arxiv.') else 'DOI'
    doi = f'<div class="pub-doi">{doi_label}: {link(p["doi"], "https://doi.org/" + p["doi"])}</div>' if p['doi'] else ''
    version_note = ' Local PDF: ' + e(p['pdf_version']) + '.' if p.get('pdf_version') else ''
    metadata_note = ' ' + e(p['citation_note']) if p.get('citation_note') else ''
    citation = '' if compact else f'''<details class="citation"><summary>BibTeX &amp; citation notes</summary><div class="citation-box"><pre>{e(p['bibtex'])}</pre><div class="citation-actions"><button type="button" class="copy-btn js-only">Copy BibTeX</button><span role="status" aria-live="polite"></span><a href="/asset/bibtex/{ident}.bib" download>Download .bib</a></div><p class="citation-note">{e(status)}. {"DOI: " + e(p['doi']) if p['doi'] else 'No DOI verified for this record.'}{version_note}{metadata_note}</p></div></details>'''
    search_text = ' '.join(str(p.get(k, '')) for k in ['title', 'authors', 'reference', 'year', 'doi', 'status', 'kind']).lower()
    attrs = '' if compact else f' data-paper-id="{ident}" data-search="{e(search_text)}" data-year="{p["year"]}" data-type="{p["kind"]}" data-status="{e(status)}"'
    return f'''<article class="pub" id="{id_prefix}{ident}"{attrs}><h3>{title_link}</h3><p class="pub-authors">{authors(p['authors'])}.</p><p class="pub-venue">{e(venue)}</p><div class="pub-meta">{pill}{''.join(resources)}</div>{doi}{citation}</article>'''


def contact():
    return '''<aside class="contact" id="contact"><div><h2>Let’s connect.</h2><p>For research collaborations and academic enquiries.</p></div><a href="mailto:paichet.ng@singaporetech.edu.sg">paichet.ng@singaporetech.edu.sg ↗</a></aside>'''


def mini_project(p):
    return f'''<article class="project-mini"><a class="mini-image" href="/research/#{p['id']}" aria-label="Explore {e(p['title'])}"><img src="{p['image']}" alt="{e(p['alt'])}" width="640" height="330" loading="lazy"></a><p class="eyebrow">{e(p['theme'])}</p><h3><a href="/research/#{p['id']}">{e(p['title'])}</a></h3><p>{e(p['subtitle'])}</p><a class="mini-link" href="/research/#{p['id']}">Explore project →</a></article>'''


def home():
    projects = [next(p for p in PROJECTS if p['id'] == ident) for ident in ['hyper-skin', 'x-palm', 'armor']]
    content = f'''<section class="hero hero--compact" aria-labelledby="home-title"><div><div class="profile-heading"><div class="profile-copy"><div><h1 id="home-title">Pai Chet Ng</h1><p class="also-known">Also known as Pai, Pc Ng</p></div><p class="role">Assistant Professor, AI &amp; Data Science Division, Infocomm Technology Cluster, Singapore Institute of Technology</p></div><figure class="ai-avatar"><a href="/asset/images/pai-chet-ng-ai-avatar-smooth-v5.svg" aria-label="View full animated avatar of Pai Chet Ng"><picture><source media="(prefers-reduced-motion: reduce)" srcset="/asset/images/pai-chet-ng-ai-avatar-cartoon-v3.png"><img id="avatar-image" src="/asset/images/pai-chet-ng-ai-avatar-cartoon-v3.png" data-animated="/asset/images/pai-chet-ng-ai-avatar-smooth-v5.svg" data-still="/asset/images/pai-chet-ng-ai-avatar-cartoon-v3.png" alt="Cartoon avatar of Pai Chet Ng, with glasses, a relaxed ponytail and a professional smile." width="512" height="512" fetchpriority="high"></picture></a><figcaption>AI avatar generated by <a href="https://developers.openai.com/api/docs/guides/tools-image-generation" target="_blank" rel="noopener noreferrer">OpenAI imagegen</a></figcaption></figure></div><div class="link-row">{link('Email ↗', 'mailto:paichet.ng@singaporetech.edu.sg')}{link('Google Scholar ↗', 'https://scholar.google.com/citations?user=WBieghIAAAAJ')}{link('ORCID ↗', 'https://orcid.org/0000-0001-9153-5411')}{link('SIT profile ↗', 'https://www.singaporetech.edu.sg/directory/faculty/pai-chet-ng')}</div></div></section>
<section class="section home-research" id="research-interests"><div class="section-heading"><h2>Trustworthy AI For Multimodal Sensing In Mobile And Pervasive Systems</h2></div><p class="intro">I joined Singapore Institute of Technology in August 2023. Previously, I was a Postdoctoral Fellow at the University of Toronto and a Research Associate at the University of Guelph. I received my PhD in Electronic and Computer Engineering from the Hong Kong University of Science and Technology.</p><p class="intro">I develop trustworthy applied AI that learns from wireless, physiological, and hyperspectral signals, connecting privacy-preserving learning with mobile devices and IoT infrastructure for healthcare, authentication, and everyday environments.</p><div class="interests"><article class="interest"><span class="index">01</span><h3>Trustworthy Learning</h3><p>Federated and personalised learning, privacy-preserving AI, robust biometrics, and trustworthy generative and agentic systems.</p></article><article class="interest"><span class="index">02</span><h3>Multimodal Sensing</h3><p>Wireless and physiological signals, visual biometrics, and hyperspectral measurements to understand people and their environments.</p></article><article class="interest"><span class="index">03</span><h3>Mobile &amp; Pervasive Systems</h3><p>Intelligent sensing on mobile devices, wearables, and IoT infrastructure for accessible, deployable applications.</p></article></div></section>
<section class="section" id="selected-research"><div class="section-heading"><h2>Selected Research</h2>{link('Research portfolio →', '/research/')}</div><div class="project-grid">{''.join(mini_project(p) for p in projects)}</div></section>
<section class="section"><div class="section-heading"><h2>Recent updates</h2>{link('Academic service →', '/service/')}</div><ul class="news-list"><li><time datetime="2026-10">Oct 2026</time><span>Call for papers: <a href="https://tpad-2027.github.io/" target="_blank" rel="noopener noreferrer">TPAD@ICASSP2027</a>. We invite you to submit papers on trustworthy perception for autonomous driving, including adversarial robustness, multimodal fusion, and foundation models.</span></li><li><time datetime="2026-09">Sep 2026</time><span><a href="/publications/#c15">X-Palm</a> was accepted to the NeurIPS 2026 Evaluations &amp; Datasets Track.</span></li><li><time datetime="2026-09">Sep 2026</time><span>Chaired and organised <a href="https://2026.ieeeicip.org/satellite-workshops/">ARTI@IEEE ICIP2026</a> and co-organised <a href="https://fd301.github.io/PFATCV26ECCV/">PFATCV@ECCV2026</a>.</span></li><li><time datetime="2026-08">Aug 2026</time><span><a href="/publications/#c7">Open-World Meme Understanding</a> was accepted to the EMNLP 2026 main conference.</span></li><li><time datetime="2026-07">Jul 2026</time><span><a href="/publications/#c1">ARMOR++</a> was submitted to IEEE Transactions on Reliability.</span></li><li><time datetime="2026-05">May 2026</time><span>Organised the <a href="https://hyper-object.github.io/">Hyper-Object Challenge</a> at IEEE ICASSP 2026, exploring low-cost hyperspectral imaging.</span></li></ul></section>
<section class="section"><div class="section-heading"><h2>Selected publications</h2>{link('Full bibliography →', '/publications/')}</div><div class="compact-pubs">{''.join(publication(PAPERS[i], True) for i in ['c15', 'c7', 'c40', 'j2', 'j4', 'j6', 'j11'])}</div></section>{contact()}'''
    write('index.html', page('Pai Chet Ng', '/', content, 'Home'))


def funded_research():
    grants = [
        ('Federated-Auth', 'Principal Investigator · MOE AcRF Tier 1', '2024–2026 · S$150,000', 'Federated authentication on mobile devices with multiple biometric modalities.', ''),
        ('Location-aware LLM chatbot', 'Principal Investigator · SIT Ignition Grant', '2024–2026 · S$150,000', 'Service personalisation and seamless check-in, in collaboration with Neoma.', ''),
        ('Privacy-preserving healthcare', 'Principal Investigator · Academy of Medical Sciences, UK', '2024–2026 · £25,000 networking grant', 'Multimodal human behaviour analysis for privacy-preserving healthcare with federated learning.', 'Oversea Co-Lead: ' + link('Professor Fani Deligianni', 'https://www.gla.ac.uk/schools/computing/staff/fanideligianni/') + ', University of Glasgow.'),
        ('StrokeCircle', 'Co-Principal Investigator · MOE AcRF Tier 1', '2026–2028 · S$21,940', 'Digital peer support and social reconnection after stroke. Technical contribution: digital platform architecture and AI-assisted peer matching.', 'Principal Investigator: ' + link('Assistant Professor Sharon Fong Mei Toh', 'https://www.singaporetech.edu.sg/directory/faculty/fong-mei-toh') + '.')]
    content = '<section class="section" id="funding"><h2>Funded Research</h2><p class="section-intro">Selected projects supporting research, collaboration, and translation.</p><div class="grant-grid">'
    for title, role, dates, description, collaborators in grants:
        people = f'<p class="grant-collaborators">{collaborators}</p>' if collaborators else ''
        content += f'<article class="grant"><h3>{e(title)}</h3><p class="grant-role">{e(role)}</p><p>{e(dates)}</p><p>{e(description)}</p>{people}</article>'
    return content + '</div></section>'


def related_publications(ids):
    items = []
    for ident in ids:
        paper = PAPERS[ident]
        venue = paper.get('research_venue') or paper.get('bib_venue') or paper['venue_note']
        items.append(f'<li>{link(paper["title"], "/publications/#" + ident)} <span>({e(venue)})</span></li>')
    return '<details class="related"><summary>Related publications</summary><ul>' + ''.join(items) + '</ul></details>'


def research_project(project):
    source = paper_url(PAPERS[project['figure_source']]) if project['figure_source'] in PAPERS else project['figure_source']
    resources = []
    lead = PAPERS[project['publications'][0]]
    if paper_url(lead):
        resources.append(link('Paper ↗', paper_url(lead)))
    if lead['pdf']:
        resources.append(link('PDF ↓', lead['pdf']))
    resources.extend(link(resource['label'] + (' ↗' if resource['url'].startswith('https://') else ' ↓'), resource['url']) for resource in project['resources'])
    legacy_anchor = '<span id="vocare"></span>' if project['id'] == 'pcar' else ''
    return f'''<article class="project" id="{project['id']}"><figure><a class="figure-link" href="{e(project['image'])}" aria-label="View full figure for {e(project['title'])}"><img src="{project['image']}" alt="{e(project['alt'])}" width="800" height="420" loading="lazy"></a><figcaption>{link(project['figure_credit'], source)}</figcaption></figure><div>{legacy_anchor}<h3>{e(project['title'])}</h3><p class="subtitle">{e(project['subtitle'])}</p><p class="description">{e(project['description'])}</p><p class="contribution"><strong>Research contribution.</strong> {e(project['contribution'])}</p><div class="resource-links">{''.join(resources)}</div>{related_publications(project['publications'])}</div></article>'''


def research():
    content = f'''<header class="page-head"><p class="eyebrow">Research portfolio</p><h1>From Signals to Trustworthy Systems</h1><p class="lead">My research brings together multimodal sensing, privacy-preserving learning, and mobile and pervasive infrastructure. These projects connect advances in learning with the realities of devices, people, and everyday environments.</p><nav class="jump-links" aria-label="Research themes">{link('Funded Research', '#funding')}{link('Hyperspectral Imaging', '#hyperspectral')}{link('Biometrics & Federated Learning', '#federated')}{link('Trustworthy & Agentic AI', '#trustworthy')}{link('Mobile & IoT Foundations', '#mobile')}</nav></header>'''
    content += funded_research()
    for ident, title, themes in [('hyperspectral', 'Computational Hyperspectral Imaging', ['Hyperspectral imaging']), ('federated', 'Biometrics & Federated Learning', ['Biometrics & federated learning']), ('trustworthy', 'Trustworthy & Agentic AI', ['Trustworthy & agentic AI'])]:
        content += f'<section class="section" id="{ident}"><h2>{e(title)}</h2>'
        content += ''.join(research_project(project) for project in PROJECTS if project['theme'] in themes)
        content += '</section>'
    content += '''<section class="section" id="mobile"><h2>Mobile &amp; IoT Foundations</h2><p class="section-intro">My earlier research on Bluetooth Low Energy, proximity sensing, and wearable contact tracing established the wireless and mobile foundations of my current work. It connects reliable sensing in dense BLE networks with device-free occupancy detection and privacy-preserving exposure tracking.</p><div class="foundation-grid">'''
    figures = [
        ('BLE Proximity Sensing', '/asset/images/ble-compressive-sensing.png', 'Smartphone proximity sensing in an ideal BLE network compared with missing signals and faulty beacons.'),
        ('Wearable Contact Tracing', '/asset/images/wearable-contact-tracing.png', 'Two smartwatch users exchange anonymous signatures; the tracing phase downloads signatures and generates an alert.'),
        ('Device-Free Occupancy Detection', '/asset/images/device-free-occupancy.png', 'BLE measurements enter a denoising-contractive autoencoder and classifier for occupancy detection.')]
    for title, image, alt in figures:
        content += f'<figure class="foundation"><a href="{image}" aria-label="View full figure for {e(title)}"><img src="{image}" alt="{e(alt)}" width="800" height="420" loading="lazy"></a><figcaption><h3>{e(title)}</h3></figcaption></figure>'
    content += '''</div><div class="resource-links"><a href="/zblog/2021/contactTracing/">Contact Tracing: Research Impact &amp; Media Coverage →</a><a href="https://ieee-dataport.org/open-access/rssdatahumanhuman" target="_blank" rel="noopener noreferrer">Contact Tracing RSS Data · IEEE Dataport ↗</a><a href="https://ieee-dataport.org/documents/rss-fingerprint-data" target="_blank" rel="noopener noreferrer">RSS Fingerprint Localization Data · IEEE Dataport ↗</a></div>'''
    content += related_publications(['j13', 'j7', 'j17', 'j5', 'j8', 'j14', 'j10', 'j11']) + '</section>' + contact()
    write('research/index.html', page('Research', '/research/', content, 'Research'))


def publications():
    papers = [p for p in PUBLICATIONS if p['kind'] != 'dataset']
    years = sorted({p['year'] for p in papers}, reverse=True)
    content = '''<header class="page-head"><p class="eyebrow">Research outputs</p><h1>Publications</h1><p class="lead">Selected contributions to trustworthy AI, multimodal sensing, and mobile and pervasive systems, followed by the complete bibliography grouped by year. Each year previews two papers; expand it to browse all works. Author lists, paper sources, PDFs, and BibTeX accompany the references.</p></header>'''
    content += '<div class="pub-tools js-only"><div class="search-field"><label class="field-label" for="publication-search">Search publications</label><input id="publication-search" type="search" placeholder="Title, author, topic, or DOI…"></div><div><label class="field-label" for="publication-year">Year</label><select id="publication-year"><option value="">All years</option>' + ''.join(f'<option>{year}</option>' for year in years) + '</select></div><div><label class="field-label" for="publication-type">Type</label><select id="publication-type"><option value="">All types</option><option value="journal">Journal</option><option value="conference">Conference / workshop</option><option value="Preprint">Preprint</option><option value="Accepted">Accepted</option><option value="Submitted">Submitted</option><option value="dataset">Dataset</option></select></div><p class="result-status" id="result-status" role="status" aria-live="polite"></p></div>'
    content = content.replace('<option value="dataset">Dataset</option>', '')
    content += '<section class="pub-section" data-publication-group><h2>Selected Publications</h2><p class="section-intro">Representative contributions across my research programme.</p>' + ''.join(publication(PAPERS[i], id_prefix='selected-') for i in SELECTED) + '</section>'
    content += '<section class="pub-section" data-publication-group><h2>All Publications</h2><p class="section-intro">The complete bibliography, including the selected publications above.</p>'
    for year in years:
        pubs = [p for p in papers if p['year'] == year]
        content += f'<section class="pub-year" data-publication-group><h3 class="year-title">{year}</h3><div class="year-preview">' + ''.join(publication(p) for p in pubs[:2]) + '</div>'
        if len(pubs) > 2:
            content += f'<details class="year-more"><summary>Show All Publications from {year}</summary><div class="year-remaining">' + ''.join(publication(p) for p in pubs[2:]) + '</div></details>'
        content += '</section>'
    content += '</section><p class="no-results" id="no-results" hidden>No publications match these filters.</p>'
    content += '<div class="section"><a href="/asset/bibtex/pai-chet-ng.bib" download>Download complete bibliography (.bib) ↓</a></div>'
    write('publications/index.html', page('Publications', '/publications/', content, 'Publications'))
    for p in PUBLICATIONS:
        write('asset/bibtex/' + p['id'] + '.bib', p['bibtex'])
    write('asset/bibtex/pai-chet-ng.bib', '\n\n'.join(p['bibtex'] for p in papers))


def teaching():
    courses = [
        ('INF2007 · Module Coordinator · SIT', 'Mobile Application Development', 'Undergraduate teaching connecting software development with practical mobile applications. AY2025 Trimester 2: lectures and laboratories for 260 students.'),
        ('CEG2001 · Co-Module Coordinator · SIT', 'Sensors and Control', 'Sensing principles and practical systems through lectures, tutorials, and laboratories. AY2025 Trimester 1: 95 students.'),
        ('Teaching · SIT', 'Implementation of Advanced AI', 'Applied AI teaching, including federated learning and practical implementation.'),
        ('Supervision & Teaching · SIT', 'Independent Study Modules', 'Independent research in large language models, retrieval-augmented generation, text-to-image generation, and federated learning.'),
        ('Teaching · SIT', 'Computer Networks', 'Networking foundations supporting connected mobile and IoT systems.'),
        ('ELEC1020 · Teaching Assistant · HKUST', 'Media Production: Technology and Design', 'Hong Kong University of Science and Technology.'),
        ('ELEC2400 · Teaching Assistant · HKUST', 'Electronic Circuits', 'Hong Kong University of Science and Technology.'),
        ('ELEC2300 · Teaching Assistant · HKUST', 'Computer Organization', 'Hong Kong University of Science and Technology.'),
        ('ELEC6910Q · Teaching Assistant · HKUST', 'Analytics and Systems for Social Media and Big Data Applications', 'Hong Kong University of Science and Technology.')]
    content = '''<header class="page-head"><p class="eyebrow">Teaching &amp; mentoring</p><h1>Teaching</h1><p class="lead">My teaching brings together mobile development, sensing, networking, and applied AI. I mentor students through independent research, capstone projects, and industry work placements, connecting rigorous methods with deployable systems.</p></header>'''
    for group, indexes in [('Undergraduate', [0, 1, 4, 5, 6, 7]), ('Postgraduate', [2, 3, 8])]:
        content += f'<section class="section"><h2>{group}</h2><div class="course-list">'
        for index in indexes:
            code, title, description = courses[index]
            content += f'<article class="course"><span class="code">{e(code)}</span><h3>{e(title)}</h3><p>{e(description)}</p></article>'
        content += '</div></section>'
    content += '''<section class="section" id="student-projects"><h2>Research and project mentoring</h2><p class="section-intro">Selected areas of student research and applied project supervision.</p><div class="mentoring-grid"><article class="mentoring"><h3>Student research</h3><ul class="simple-list"><li>Federated learning for activity recognition and multimodal biometric verification.</li><li>Contactless palmprint identification with foundation models.</li><li>Deepfake robustness and agentic adversarial evaluation.</li><li>Instruction-guided speech synthesis and multimodal emotion recognition.</li></ul></article><article class="mentoring"><h3>Applied student projects</h3><ul class="simple-list"><li><a href="/research/#vocare">VoCare AI</a>: a multi-agent voice assistant for clinic operations.</li><li>StrokeCircle: digital peer support and matching for stroke survivors.</li><li>Agentic RAG and local language-model workflows.</li><li>Security analysis, scam detection, and automated knowledge retrieval.</li><li>Industry-linked software and connected-device projects.</li></ul></article></div></section><section class="section"><h2>Industry-Linked Learning</h2><p class="section-intro">I supervise capstone projects and Integrated Work Study Programme placements across software engineering, information security, computing, and DigiPen programmes. Industry placements and projects span organisations including GovTech, EY, Ensign InfoSecurity, ST Engineering, STMicroelectronics, Singtel, and Standard Chartered.</p><p class="section-intro">My research mentoring also includes collaborations with doctoral students across SIT and partner institutions in Canada, the United States, the United Kingdom, Greece, and Singapore.</p></section>''' + contact()
    write('teaching/index.html', page('Teaching & Mentoring', '/teaching/', content, 'Teaching'))


def service():
    events = [
        ('17 Sep 2026', 'ARTI@IEEE ICIP2026', 'Chair and organiser · Agentic Reasoning for Trustworthy Imagery · Tampere, Finland', 'https://2026.ieeeicip.org/satellite-workshops/'),
        ('9 Sep 2026', 'PFATCV@ECCV2026', 'Co-organiser · Privacy, Fairness, Accountability and Transparency in Computer Vision · Malmö, Sweden', 'https://fd301.github.io/PFATCV26ECCV/'),
        ('Jul 2026', 'PPT-HAC · IEEE ICHMS 2026', 'Special-session organiser · Privacy-Preserving and Trustworthy Human-Agent Collaboration · Singapore', ''),
        ('7 May 2026', 'Hyper-Object · IEEE ICASSP 2026', 'Challenge chair and organiser · Low-cost hyperspectral reconstruction · Barcelona, Spain', 'https://hyper-object.github.io/'),
        ('27 Nov 2025', 'PFATCV@BMVC2025', 'Co-organiser · Privacy, Fairness, Accountability and Transparency in Computer Vision · Sheffield, UK', 'https://sites.google.com/view/pfatcvbmvc25/program'),
        ('14 Sep 2025', 'SEEDS · IEEE ICIP 2025', 'Co-organiser · Edge Intelligence: Smart, Efficient, and Scalable Solutions for IoT, Wearables, and Embedded Devices · Anchorage, USA', 'https://sites.google.com/view/seeds2025'),
        ('20 Jul 2025', 'FPPAI · IEEE ICDCS 2025', 'Co-chair · Federated and Privacy Preserving AI in Biomedical Applications · Glasgow, UK', 'https://sites.google.com/view/fppaiicdcs25/program'),
        ('7 Feb 2025', 'Privacy-Preserving AI for Smart Healthcare', 'Chair · SIT and University of Glasgow Forum · Singapore', 'https://mhba-fl.github.io/'),
        ('28 Nov 2024', 'PFATCV@BMVC2024', 'Co-organiser · Privacy, Fairness, Accountability and Transparency in Computer Vision · Glasgow, UK', 'https://sites.google.com/view/pfatcvbmvc24/home'),
        ('Nov 2024', 'Multimodal Human Behaviour Analysis with Federated Learning', 'Workshop chair · IEEE World Forum on Internet of Things · Ottawa, Canada', 'https://mhba-fl.github.io/IEEE-WF-IoT_Workshop_backup/'),
        ('Apr 2024', 'Hyper-Skin · IEEE ICASSP 2024', 'Challenge chair · Hyperspectral Skin Vision · Seoul, South Korea', 'https://uoft-hyperskin.github.io/')]
    content = '''<section class="section"><h2>Conferences, Workshops &amp; Challenges</h2>'''
    for when, title, description, url in events:
        content += f'<article class="service-row"><time>{e(when)}</time><div><h3>{e(title)}</h3><p>{e(description)}</p>{link("Event website ↗", url, "service-link")}</div></article>'
    content += '''</section><section class="section"><h2>Editorial &amp; reviewing service</h2><ul class="simple-list"><li><strong>Programme Committee Member:</strong> AAAI 2026 Artificial Intelligence for Social Impact Track.</li><li><strong>Session Chair:</strong> IEEE ICASSP 2024 and 2026; IEEE ICIP 2026; AAAI 2026.</li><li><strong>Guest Editor:</strong> <a href="https://www.mdpi.com/journal/information/special_issues/0V446291TI">Blockchain and AI: Innovations and Applications in ICT</a>, <em>Information</em>, 2025.</li><li><strong>Conference reviewing:</strong> AAAI, NeurIPS, EMNLP, ACM Multimedia, IEEE ICASSP, IEEE ICIP, IEEE ICME, IEEE ICRA, IEEE ICHMS, IEEE TENCON, IEEE ICC, IEEE GLOBECOM, IEEE WCNC, IEEE PIMRC, IEEE MLSP, IEEE SOLI, IEEE World Forum on Internet of Things (WF-IoT), and related workshops.</li><li><strong>Journal reviewing:</strong> ACM Computing Surveys; ACM Transactions on Multimedia Computing, Communications, and Applications; Circuits, Systems, and Signal Processing; PLOS ONE; IEEE Access; IEEE Internet of Things Journal; IEEE Internet of Things Magazine; IEEE Open Journal of the Communications Society; IEEE Open Journal of Signal Processing; IEEE Sensors Journal; IEEE Sensors Letters; IEEE Transactions on Communications; IEEE Transactions on Consumer Electronics; IEEE Transactions on Information Forensics and Security; IEEE Transactions on Mobile Computing; IEEE Transactions on Multimedia; IEEE Transactions on Signal Processing; IEEE Transactions on Vehicular Technology; IEEE Transactions on Wireless Communications.</li><li><strong>Outstanding Reviewer Award:</strong> Special Sessions, IEEE ICHMS 2026.</li></ul></section><section class="section"><h2>Selected invited talks</h2><article class="service-row"><time>Jul 2026</time><div><h3>Multi-Modal Human Activity Recognition</h3><p>Federated Learning Across Heterogeneous Physiological and Wireless Signal · CSAN 2026, online.</p></div></article><article class="service-row"><time>Nov 2024</time><div><h3>Hyperspectral Skin Analysis on Consumer Devices</h3><p>Invited research talk · University of Glasgow, UK.</p></div></article><article class="service-row"><time datetime="2024-07-31">31 Jul 2024</time><div><h3>Hyperspectral Image Reconstruction for Skin Analysis</h3><p>ICT Research Seminar · Singapore Institute of Technology, Singapore.</p></div></article><article class="service-row"><time>Jul 2024</time><div><h3>Federated Learning for Human Behaviour Analysis</h3><p>Sengkang Hospital AI Forum · Singapore.</p></div></article><article class="service-row"><time datetime="2024-05-15">15 May 2024</time><div><h3>Multimodal Federated Learning for Human Behaviour Analysis</h3><p>SIT–University of Glasgow / UGS ICT Research Seminar · Singapore.</p></div></article><article class="service-row"><time>Jan 2023</time><div><h3>Bluetooth Low Energy for IoT Sensing</h3><p>Research seminar · Lakehead University, Canada.</p></div></article><article class="service-row"><time datetime="2021-07-30">30 Jul 2021</time><div><h3>Fighting COVID-19 with Bluetooth Low Energy-Based Contact Tracing</h3><p>Moscow Telecommunication Seminar · Online.</p></div></article></section>''' + contact()
    write('service/index.html', page('Service & Engagement', '/service/', content, 'Service'))


def notes():
    notes_path = ROOT / 'data/notes.json'
    records = json.loads(notes_path.read_text()) if notes_path.exists() else []
    content = '<header class="page-head"><p class="eyebrow">Research &amp; technical notes</p><h1>Notes</h1><p class="lead">Technical notes on physiological signals and time-series analysis, alongside research impact and public engagement.</p></header><section class="section">'
    for note in records:
        content += f'<article class="service-row"><time>{e(note["date"])}</time><div><h3>{link(note["title"], note["path"])}</h3><p>{e(note["description"])}</p></div></article>'
        updated = f'<p class="note-provenance">Updated {e(note["updated"])}</p>' if note.get('updated') else ''
        write(note['path'].lstrip('/') + 'index.html', page(note['title'], note['path'], f'<article class="note-article"><p class="eyebrow">Notes · {e(note["date"])}</p><h1>{e(note["title"])}</h1>{updated}<div class="note-body">{note["body"]}</div></article>', description=note['description']))
    content += '</section>'
    write('notes/index.html', page('Notes', '/notes/', content))
    redirect('zblog/2021/histogramVisualization/index.html', '/notes/')
    redirect('artificial_intelligence/timeseries/dtw/index.html', '/notes/dynamic-time-warping/')
    redirect('biometric_signals/aboutecg/aboutecg/index.html', '/notes/ecg-heart-rate/')


def redirect(path, destination):
    write(path, '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pai Chet Ng</title><meta http-equiv="refresh" content="0;url=' + e(destination) + '"><link rel="canonical" href="' + SITE + e(destination) + '"></head><body><p>This page has moved. <a href="' + e(destination) + '">Continue to Pai Chet Ng’s website</a>.</p></body></html>')


def handover():
    # Owner-only review notes live outside the publicly served repository.
    directory = ROOT.parent / 'pc-ng-webpage-review'
    directory.mkdir(exist_ok=True)
    missing = [p for p in PUBLICATIONS if not p['pdf'] and p['kind'] != 'dataset']
    lines = ['# Papers to supply', '', f'{len(missing)} papers do not yet have a verified local PDF. Source links remain available where verified; unavailable PDFs are marked “PDF pending”.', '', 'Save each paper at its exact destination below. Then rebuild the website; the supplied PDFs will be picked up automatically.', '', 'Destination folder: /home/pcng/pc-ng.github.io/asset/paper/', '']
    for p in missing:
        lines += [f'## {p["id"].upper()} · {p["title"]}', '', p['reference'], '', 'Source: ' + (paper_url(p) or p['source'] or 'No paper-specific public source located.')]
        lines.append('Save as: /home/pcng/pc-ng.github.io/asset/paper/' + p['id'] + '.pdf')
        if p['status'] != 'Published':
            lines.append('Status: ' + p['status'])
        errors = [a for a in p.get('attempts', []) if not a['url'].startswith('Crossref')]
        if errors:
            lines.append('Download result: ' + errors[-1]['error'])
        else:
            lines.append('Download result: no direct public PDF located in the checked sources.')
        lines.append('')
    (directory / 'papers-to-supply.md').write_text('\n'.join(lines))
    missing_doi = [p for p in PUBLICATIONS if not p['doi']]
    metadata_notes = '\n'.join('- ' + p['id'].upper() + ': ' + p['metadata_note'] for p in PUBLICATIONS if p.get('metadata_note'))
    (directory / 'metadata-to-review.md').write_text('# Citation metadata to review\n\nNo DOI was verified for the following records; none has been invented.\n\n' + '\n'.join(f'- {p["id"].upper()}: {p["title"]} ({p["status"]}). {p["source"]}' for p in missing_doi) + '\n\n## Publisher metadata corrections\n\n' + metadata_notes + '\n')
    (directory / 'project-resources.md').write_text('# Portfolio asset and resource provenance\n\n' + '\n\n'.join('## ' + p['title'] + '\n\nImage: ' + p['image'] + '\n\nCredit: ' + p['figure_credit'] + '\n\n' + '\n'.join(r['label'] + ': ' + r['url'] for r in p['resources']) for p in PROJECTS) + '\n\nSlides or code links are shown only where a public, project-specific source was found. Additional slides/code for ARMOR, FedGraph, X-Palm and VoCare may be supplied by the owner.\n')


def main():
    home(); research(); publications(); teaching(); service(); notes()
    for source, dest in {'zpublications/index.html': '/publications/', 'zresearch/index.html': '/research/', 'zresearch/physiological/index.html': '/research/#federated-sensing', 'zresearch/contactTracing/index.html': '/research/#mobile', 'zresearch/beacon/index.html': '/research/#mobile', 'zresearch/blank/index.html': '/research/', 'zdevelopments/index.html': '/teaching/#student-projects', 'zdevelopments/1_pythoncookbook/index.html': '/notes/', 'zdevelopments/2_mobileDevelopment/index.html': '/teaching/#student-projects', 'zdevelopments/3_embedded/index.html': '/research/#mobile', 'zdevelopments/blank/index.html': '/research/', 'zcv/index.html': '/', 'zblog/index.html': '/notes/'}.items():
        redirect(source, dest)
    write('404.html', page('Page not found', '/404.html', '<header class="page-head"><p class="eyebrow">404</p><h1>Page not found.</h1><p class="lead">Explore the <a href="/research/">research portfolio</a>, browse <a href="/publications/">publications</a>, or <a href="/">return home</a>.</p></header>'))
    write('.nojekyll', '')
    write('robots.txt', 'User-agent: *\nAllow: /\nSitemap: ' + SITE + '/sitemap.xml')
    urls = ['/', '/research/', '/publications/', '/teaching/', '/service/', '/notes/'] + [note['path'] for note in json.loads((ROOT / 'data/notes.json').read_text())]
    write('sitemap.xml', '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' + ''.join('<url><loc>' + SITE + url + '</loc></url>' for url in urls) + '</urlset>')
    handover()
    print('Built 6 pages and preserved legacy routes.')
    print('References:', len(PUBLICATIONS), '| Papers:', sum(p['kind'] != 'dataset' for p in PUBLICATIONS), '| Local PDFs:', sum(bool(p['pdf']) for p in PUBLICATIONS))


if __name__ == '__main__':
    main()
