const noteBody = document.querySelector('.note-body');
if (noteBody && window.renderMathInElement) {
  window.renderMathInElement(noteBody, {
    throwOnError: false,
    strict: 'ignore',
    trust: false
  });
}
