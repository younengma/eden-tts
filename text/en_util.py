""" from https://github.com/keithito/tacotron  we make small modifications"""
import re
from text import cleaners
import os
import json
from utils.log_util import get_logger
from g2p_en import G2p
from string import punctuation

log = get_logger(__name__)

symbol_to_id_file = "resource/symbol_to_id_en.json"
if os.path.exists(symbol_to_id_file) is False:
    from text.symbols import symbols
    # Mappings from symbol to numeric ID and vice versa:
    _symbol_to_id = {s: i for i, s in enumerate(symbols)}
    _id_to_symbol = {i: s for i, s in enumerate(symbols)}
    with open(symbol_to_id_file, "w") as f:
        json.dump(_symbol_to_id, f, indent=2, ensure_ascii=False)
        log.info(f"save symbol2id to {symbol_to_id_file}.")
else:
    with open(symbol_to_id_file, "r") as f:
        _symbol_to_id = json.load(f)
        _id_to_symbol = {i: s for i, s in _symbol_to_id.items()}
        log.info(f"load symbol2id from {symbol_to_id_file}.")

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")
# add a special token at the begging and the end.
START = _symbol_to_id[" "]
END = _symbol_to_id["."]


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


lexicon = read_lexicon("resource/librispeech-lexicon.txt")
g2p = G2p()


def get_phonemes(text):
    text = text.rstrip(punctuation)
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))

    return phones


def text_to_sequence(text, cleaner_names=["english_cleaners"], token_type="char"):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
      token_type:  if "char": use character as tokens esle: use phonemes as tokens

    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []
    if token_type == "char":
        sequence += symbols_to_sequence(_clean_text(text, cleaner_names))
    else:
        # phonemes as inputs
        # Check for curly braces and treat their contents as ARPAbet:
        text = get_phonemes(text)
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence += symbols_to_sequence(_clean_text(text, cleaner_names))
                break
            sequence += symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
            sequence += _arpabet_to_sequence(m.group(2))
            text = m.group(3)
    if sequence[0] != START:
        sequence = [START] + sequence
    if sequence[-1] != END:
        sequence += [END]
    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return symbols_to_sequence(["@" + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != "_" and s != "~"
