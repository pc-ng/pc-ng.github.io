$(document).ready(function() {
    $('a.abstract').click(function() {
        $(this).parent().parent().find(".abstract.hide").toggleClass('open');
    });
    $('a.bibtex').click(function() {
        $(this).parent().parent().find(".bibtex.hide").toggleClass('open');
    });
    $('.navbar-nav').find('a').removeClass('waves-effect waves-light');
});
