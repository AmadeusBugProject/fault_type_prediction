import re

from sklearn.base import BaseEstimator, TransformerMixin


class SpecialCharacterToWords(BaseEstimator, TransformerMixin):
    def __init__(self, repl_all_caps=True):
        self.repl_all_caps = repl_all_caps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.replace_special_char(i) for i in X]

    def replace_special_char(self, text):
        for rec in rec_map:
            text = rec[0].sub(rec[1], text)
        if self.repl_all_caps:
            text = rec_all_caps[0].sub(rec_all_caps[1], text)
        for repl in char_map:
            text = text.replace(repl[0], repl[1])
        for repl in unic_char_map:
            text = text.replace(repl[0], repl[1])
        return ' JJJnewline ' + text + ' JJJendline '


char_map = [
    ('    ', ' JJJquadspace '),
    ('  ', ' JJJdoublespace '),
    ('\n', ' JJJendline \n JJJnewline '),
    ('\t', ' JJJtabulator '),
    ('~', ' JJJtilde '),
    ('!', ' JJJexclamation '),
    ('@', ' JJJat '),
    ('####', ' JJJquadhash '),
    ('###', ' JJJtriplehash '),
    ('##', ' JJJdoublehash '),
    ('#', ' JJJsinglehash '),
    ('$', ' JJJdollar '),
    ('%', ' JJJpercent '),
    ('^', ' JJJhat '),
    ('&', ' JJJampersand '),
    ('*', ' JJJasterisk '),
    ('(', ' JJJroundbracketopen '),
    (')', ' JJJroundbracketclose '),
    ('_', ' JJJunderscore '),
    ('-', ' JJJminus '),
    ('=', ' JJJequals '),
    ('+', ' JJJplus '),
    ('[', ' JJJsqarebracketopen '),
    (']', ' JJJsqarebracketclose '),
    ('{', ' JJJcurlybracketopen '),
    ('}', ' JJJcurlybracketclose '),
    (';', ' JJJsemicolon '),
    (':', ' JJJcolon '),
    ("'", ' JJJsinglequote '),
    ('"', ' JJJquote '),
    ('\\', ' JJJbackslash '),
    ('|', ' JJJpipe '),
    (',', ' JJJcomma '),
    ('<', ' JJJsmaller '),
    ('.', ' JJJdot '),
    ('>', ' JJJlarger '),
    ('/', ' JJJslash '),
    ('`', ' JJJbacktick '),
    ('?', ' JJJquestion ')]

unic_char_map = [
    ('โ', ' JJJtick '),
    ('ยด', ' JJJtick '),
    ('โ', ' JJJtick '),
    ('โ', ' JJJsquote '),
    ('โ', ' JJJsquote '),
    ('๏ผ', ' JJJcolon '),
    ('๏น', ' JJJcolon '),
    ('๏ผ', ' JJJcomma '),
    ('โฆ', ' JJJellipsis '),
    ('โ', ' JJJminus '),
    ('โ', ' JJJminus '),
    ('ยซ', ' JJJpointybracketopen '),
    ('โบ', ' JJJpointybracketopen '),
    ('ยป', ' JJJpointybracketclose '),
    ('โน', ' JJJpointybracketclose '),
    ('๐', ' JJJunicodearrow '),
    ('โ', ' JJJunicodearrow '),
    ('โ', ' JJJunicodearrow '),
    ('โ', ' JJJunicodebox '),
    ('โ', ' JJJunicodebox '),
    ('โ', ' JJJunicodebox '),
    ('โ', ' JJJunicodebox '),
    ('โค', ' JJJunicodebox '),
    ('โ', ' JJJunicodebox '),
    ('โ', ' JJJunicodebox '),
    ('โฌ', ' JJJunicodebox '),
    ('โ', ' JJJunicodebox '),
    ('โ', ' JJJunicodebox '),
    ('โ', ' JJJunicodebox '),
    ('โ', ' JJJunicodebox '),
    ('โ', ' JJJunicodebox '),
    ('โ', ' JJJunicodebox '),
    ('โ', ' JJJunicodebox '),
    ('ยง', ' JJJparagraph '),
    ('ยท', ' JJJitemize '),
    ('โข', ' JJJitemize '),
    ('โ', ' JJJitemize '),
    ('โ', ' JJJCheckmark '),
    ('โ', ' JJJCheckmark ')]

rec_map = [
    (re.compile(r"(?:(?:[A-Z][a-z0-9]*)+(?:Exception|Error))"), ' JJJexception '),
    (re.compile(r"(?:(?:[A-Z]?[a-z0-9]+)(?:[A-Z][a-z0-9]*)+)"), ' JJJcamelcased '),
    (re.compile(r"(?:(?:\w+_)+\w+)"), ' JJJunderscored '),
    (re.compile(r"(?:0x[a-f0-9]+)"), ' JJJhex '),
    (re.compile(r"\d+"), ' JJJnumber '),
]

rec_all_caps = (re.compile(r"(?:[A-Z]{3,})"), ' JJJallcaps ')



