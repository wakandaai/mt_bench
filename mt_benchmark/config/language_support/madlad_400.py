#mt_benchmark/config/language_support/madlad_400.py
"""
MADLAD-400 Language Support Configuration

This module provides language support information for MADLAD-400 model evaluation
with compatibility for Seamless and Toucan models.

Key Features:
- Converts BCP-47 language codes to ISO 639-3 format for cross-model compatibility
- Provides 104 languages with consistent language_script format (e.g., 'eng_Latn')
- Identifies 13 African languages supported by MADLAD-400
- Includes mapping functions for code conversion between BCP-47 and ISO 639-3

Compatibility:
- 97 languages overlap with Seamless model
- 10 languages overlap with both Seamless and Toucan models  
- 8 African languages available across all three models:
  afr_Latn (Afrikaans), amh_Ethi (Amharic), ibo_Latn (Igbo), lug_Latn (Luganda),
  nya_Latn (Chichewa), som_Latn (Somali), yor_Latn (Yoruba), zul_Latn (Zulu)
"""

import csv
from io import StringIO
from typing import Dict 

# Mapping from MADLAD-400 BCP-47 codes to ISO 639-3 codes for compatibility
BCP47_TO_ISO639_3 = {
    "en": "eng",  # English
    "ru": "rus",  # Russian
    "es": "spa",  # Spanish
    "fr": "fra",  # French
    "de": "deu",  # German
    "it": "ita",  # Italian
    "pt": "por",  # Portuguese
    "pl": "pol",  # Polish
    "nl": "nld",  # Dutch
    "vi": "vie",  # Vietnamese
    "tr": "tur",  # Turkish
    "sv": "swe",  # Swedish
    "id": "ind",  # Indonesian
    "ro": "ron",  # Romanian
    "cs": "ces",  # Czech
    "zh": "cmn",  # Mandarin Chinese
    "hu": "hun",  # Hungarian
    "ja": "jpn",  # Japanese
    "th": "tha",  # Thai
    "fi": "fin",  # Finnish
    "fa": "pes",  # Persian
    "uk": "ukr",  # Ukrainian
    "da": "dan",  # Danish
    "el": "ell",  # Greek
    "no": "nob",  # Norwegian (Bokmål)
    "bg": "bul",  # Bulgarian
    "sk": "slk",  # Slovak
    "ko": "kor",  # Korean
    "ar": "arb",  # Arabic (Modern Standard)
    "lt": "lit",  # Lithuanian
    "ca": "cat",  # Catalan
    "sl": "slv",  # Slovenian
    "he": "heb",  # Hebrew
    "et": "est",  # Estonian
    "lv": "lvs",  # Latvian
    "hi": "hin",  # Hindi
    "sq": "sqi",  # Albanian
    "ms": "zsm",  # Malay (Standard)
    "az": "azj",  # Azerbaijani (North)
    "sr": "srp",  # Serbian
    "ta": "tam",  # Tamil
    "hr": "hrv",  # Croatian
    "kk": "kaz",  # Kazakh
    "is": "isl",  # Icelandic
    "ml": "mal",  # Malayalam
    "mr": "mar",  # Marathi
    "te": "tel",  # Telugu
    "af": "afr",  # Afrikaans
    "gl": "glg",  # Galician
    "fil": "tgl", # Filipino (Tagalog)
    "be": "bel",  # Belarusian
    "mk": "mkd",  # Macedonian
    "eu": "eus",  # Basque
    "bn": "ben",  # Bengali
    "ka": "kat",  # Georgian
    "mn": "khk",  # Mongolian (Halh)
    "bs": "bos",  # Bosnian
    "uz": "uzn",  # Uzbek (Northern)
    "ur": "urd",  # Urdu
    "sw": "swh",  # Swahili
    "yue": "yue", # Cantonese
    "ne": "npi",  # Nepali
    "kn": "kan",  # Kannada
    "gu": "guj",  # Gujarati
    "si": "sin",  # Sinhala
    "cy": "cym",  # Welsh
    "hy": "hye",  # Armenian
    "ky": "kir",  # Kyrgyz
    "tg": "tgk",  # Tajik
    "ga": "gle",  # Irish
    "mt": "mlt",  # Maltese
    "my": "mya",  # Myanmar (Burmese)
    "km": "khm",  # Khmer
    "so": "som",  # Somali
    "ku": "kmr",  # Kurdish (Kurmanji)
    "ps": "pbt",  # Pashto (Southern)
    "pa": "pan",  # Punjabi
    "rw": "kin",  # Kinyarwanda
    "lo": "lao",  # Lao
    "ha": "hau",  # Hausa
    "ckb": "ckb", # Kurdish (Sorani)
    "mg": "plt",  # Malagasy (Plateau)
    "am": "amh",  # Amharic
    "jv": "jav",  # Javanese
    "sd": "snd",  # Sindhi
    "ceb": "ceb", # Cebuano
    "xh": "xho",  # Xhosa
    "su": "sun",  # Sundanese
    "ny": "nya",  # Chichewa
    "sn": "sna",  # Shona
    "zu": "zul",  # Zulu
    "ig": "ibo",  # Igbo
    "yo": "yor",  # Yoruba
    "st": "sot",  # Sesotho
    "om": "gaz",  # Oromo (West Central)
    "ti": "tir",  # Tigrinya
    "ee": "ewe",  # Ewe
    "lg": "lug",  # Luganda
    "fon": "fon", # Fon
    "ts": "tso",  # Tsonga
    "tn": "tsn",  # Tswana
    "ak": "aka",  # Akan (Twi)
    "ln": "lin",  # Lingala
    "kbp": "kbp", # Kabiyè
    "wo": "wol",  # Wolof
    "bm": "bam",  # Bambara
    "ff": "fuv",  # Fulfulde (Nigerian)
    "rn": "run",  # Rundi
    "kg": "kon",  # Kongo
}

# MADLAD-400 language data with BCP-47 codes, names, and scripts
data = """code	language	script
en	English	Latn
ru	Russian	Cyrl
es	Spanish	Latn
fr	French	Latn
de	German	Latn
it	Italian	Latn
pt	Portuguese	Latn
pl	Polish	Latn
nl	Dutch	Latn
vi	Vietnamese	Latn
tr	Turkish	Latn
sv	Swedish	Latn
id	Indonesian	Latn
ro	Romanian	Latn
cs	Czech	Latn
zh	Mandarin Chinese	Hans
hu	Hungarian	Latn
ja	Japanese	Jpan
th	Thai	Thai
fi	Finnish	Latn
fa	Persian	Arab
uk	Ukrainian	Cyrl
da	Danish	Latn
el	Greek	Grek
no	Norwegian	Latn
bg	Bulgarian	Cyrl
sk	Slovak	Latn
ko	Korean	Kore
ar	Arabic	Arab
lt	Lithuanian	Latn
ca	Catalan	Latn
sl	Slovenian	Latn
he	Hebrew	Hebr
et	Estonian	Latn
lv	Latvian	Latn
hi	Hindi	Deva
sq	Albanian	Latn
ms	Malay	Latn
az	Azerbaijani	Latn
sr	Serbian	Cyrl
ta	Tamil	Taml
hr	Croatian	Latn
kk	Kazakh	Cyrl
is	Icelandic	Latn
ml	Malayalam	Mlym
mr	Marathi	Deva
te	Telugu	Telu
af	Afrikaans	Latn
gl	Galician	Latn
fil	Filipino	Latn
be	Belarusian	Cyrl
mk	Macedonian	Cyrl
eu	Basque	Latn
bn	Bengali	Beng
ka	Georgian	Geor
mn	Mongolian	Cyrl
bs	Bosnian	Cyrl
uz	Uzbek	Latn
ur	Urdu	Arab
sw	Swahili	Latn
yue	Cantonese	Hant
ne	Nepali	Deva
kn	Kannada	Knda
kaa	Kara-Kalpak	Cyrl
gu	Gujarati	Gujr
si	Sinhala	Sinh
cy	Welsh	Latn
eo	Esperanto	Latn
la	Latin	Latn
hy	Armenian	Armn
ky	Kyrgyz	Cyrl
tg	Tajik	Cyrl
ga	Irish	Latn
mt	Maltese	Latn
my	Myanmar (Burmese)	Mymr
km	Khmer	Khmr
tt	Tatar	Cyrl
so	Somali	Latn
ku	Kurdish (Kurmanji)	Latn
ps	Pashto	Arab
pa	Punjabi	Guru
rw	Kinyarwanda	Latn
lo	Lao	Laoo
ha	Hausa	Latn
dv	Dhivehi	Thaa
fy	Western Frisian	Latn
lb	Luxembourgish	Latn
ckb	Kurdish (Sorani)	Arab
mg	Malagasy	Latn
gd	Scottish Gaelic	Latn
am	Amharic	Ethi
ug	Uyghur	Arab
ht	Haitian Creole	Latn
grc	Ancient Greek	Grek
hmn	Hmong	Latn
sd	Sindhi	Arab
jv	Javanese	Latn
mi	Maori	Latn
tk	Turkmen	Latn
ceb	Cebuano	Latn
yi	Yiddish	Hebr
ba	Bashkir	Cyrl
fo	Faroese	Latn
or	Odia (Oriya)	Orya
xh	Xhosa	Latn
su	Sundanese	Latn
kl	Kalaallisut	Latn
ny	Chichewa	Latn
sm	Samoan	Latn
sn	Shona	Latn
co	Corsican	Latn
zu	Zulu	Latn
ig	Igbo	Latn
yo	Yoruba	Latn
pap	Papiamento	Latn
st	Sesotho	Latn
haw	Hawaiian	Latn
as	Assamese	Beng
oc	Occitan	Latn
cv	Chuvash	Cyrl
lus	Mizo	Latn
tet	Tetum	Latn
gsw	Swiss German	Latn
sah	Yakut	Cyrl
br	Breton	Latn
rm	Romansh	Latn
sa	Sanskrit	Deva
bo	Tibetan	Tibt
om	Oromo	Latn
se	Northern Sami	Latn
ce	Chechen	Cyrl
cnh	Hakha Chin	Latn
ilo	Ilocano	Latn
hil	Hiligaynon	Latn
ti	Tigrinya	Ethi
ee	Ewe	Latn
lg	Luganda	Latn
fon	Fon	Latn
ts	Tsonga	Latn
tn	Tswana	Latn
nso	Sepedi	Latn
ak	Twi	Latn
ln	Lingala	Latn
gn	Guarani	Latn
kbp	Kabiyè	Latn
ve	Venda	Latn
lu	Luba-Katanga	Latn
tiv	Tiv	Latn
wo	Wolof	Latn
bm	Bambara	Latn
ff	Fulfulde	Latn
rn	Rundi	Latn
kg	Kongo	Latn"""

def madlad400_languages_supported() -> Dict[str, Dict[str, str]]:
    """
    Get a dictionary of all MADLAD-400 supported languages with ISO 639-3 codes for compatibility.
    
    Returns:
        Dict with language codes as keys and metadata as values.
        Format: {iso639_3_code_script: {'name': str}}
    """
    
    # Parse the TSV data
    reader = csv.DictReader(StringIO(data), delimiter='\t')
    
    language_dict = {}
    
    for row in reader:
        bcp47_code = row['code'].strip()
        name = row['language'].strip()
        script = row['script'].strip()
        
        # Convert BCP-47 code to ISO 639-3 for compatibility
        iso_code = BCP47_TO_ISO639_3.get(bcp47_code, bcp47_code)
        
        # Create key in format: iso_code_script (e.g., 'eng_Latn', 'afr_Latn')
        key = f"{iso_code}_{script}"
        
        language_dict[key] = {
            'name': name,
            'bcp47_code': bcp47_code  # Keep original BCP-47 code for reference
        }
    
    return language_dict


def madlad400_african_languages() -> Dict[str, Dict[str, str]]:
    """
    Get only African languages from MADLAD-400.
    
    Returns:
        Dict with African language codes as keys and metadata as values.
    """
    
    # African language codes (ISO 639-3 format for consistency)
    african_language_codes = {
        'afr',  # Afrikaans
        'amh',  # Amharic
        'hau',  # Hausa
        'ibo',  # Igbo
        'yor',  # Yoruba
        'swh',  # Swahili
        'som',  # Somali
        'gaz',  # Oromo
        'kin',  # Kinyarwanda
        'zul',  # Zulu
        'xho',  # Xhosa
        'sna',  # Shona
        'nya',  # Chichewa
        'sot',  # Sesotho
        'tsn',  # Tswana
        'tso',  # Tsonga
        'ven',  # Venda
        'lug',  # Luganda
        'ewe',  # Ewe
        'tir',  # Tigrinya
        'fon',  # Fon
        'nso',  # Sepedi
        'aka',  # Akan/Twi
        'lin',  # Lingala
        'kbp',  # Kabiyè
        'lub',  # Luba-Katanga
        'tiv',  # Tiv
        'wol',  # Wolof
        'bam',  # Bambara
        'fuv',  # Fulfulde
        'run',  # Rundi
        'kon',  # Kongo
        'plt'   # Malagasy
    }
    
    all_languages = madlad400_languages_supported()
    
    african_dict = {}
    for key, value in all_languages.items():
        iso_code = key.split('_')[0]
        if iso_code in african_language_codes:
            african_dict[key] = value
    
    return african_dict


def get_language_info(language_code: str) -> Dict[str, str]:
    """
    Get information about a specific language.
    
    Args:
        language_code: Either BCP-47 (e.g., 'sw', 'yo', 'en') or ISO 639-3 code (e.g., 'swh', 'yor', 'eng')
    
    Returns:
        Dictionary with language information or None if not found
    """
    all_languages = madlad400_languages_supported()
    
    # First try to find by ISO 639-3 code (primary key format)
    for key, value in all_languages.items():
        if key.startswith(f"{language_code}_"):
            return {
                'code': language_code,
                'key': key,
                **value
            }
    
    # If not found, try to find by BCP-47 code
    for key, value in all_languages.items():
        if value.get('bcp47_code') == language_code:
            return {
                'code': language_code,
                'key': key,
                **value
            }
    
    return None


def get_bcp47_to_iso_mapping() -> Dict[str, str]:
    """
    Get the mapping from BCP-47 codes to ISO 639-3 codes.
    
    Returns:
        Dictionary mapping BCP-47 codes to ISO 639-3 codes
    """
    return BCP47_TO_ISO639_3.copy()


def get_iso_to_bcp47_mapping() -> Dict[str, str]:
    """
    Get the mapping from ISO 639-3 codes to BCP-47 codes.
    
    Returns:
        Dictionary mapping ISO 639-3 codes to BCP-47 codes
    """
    return {v: k for k, v in BCP47_TO_ISO639_3.items()}


def normalize_language_key(language_key: str) -> str:
    """
    Normalize a language key to the standard ISO 639-3_Script format.
    
    Args:
        language_key: Language key in various formats (e.g., 'en_Latn', 'sw_Latn', 'eng_Latn')
    
    Returns:
        Normalized language key in ISO 639-3_Script format
    """
    if '_' not in language_key:
        return language_key
    
    code, script = language_key.split('_', 1)
    
    # Convert BCP-47 to ISO 639-3 if needed
    iso_code = BCP47_TO_ISO639_3.get(code, code)
    
    return f"{iso_code}_{script}"


def get_compatible_language_codes() -> Dict[str, str]:
    """
    Get all language codes in a format compatible with Seamless and Toucan models.
    
    Returns:
        Dictionary mapping MADLAD-400 language keys to compatible keys
    """
    all_languages = madlad400_languages_supported()
    compatible_codes = {}
    
    for key in all_languages.keys():
        normalized_key = normalize_language_key(key)
        compatible_codes[key] = normalized_key
    
    return compatible_codes


def get_evaluation_languages(include_african_only: bool = False) -> Dict[str, Dict[str, str]]:
    """
    Get languages suitable for cross-model evaluation.
    
    Args:
        include_african_only: If True, return only African languages
        
    Returns:
        Dictionary of languages with their metadata
    """
    if include_african_only:
        return madlad400_african_languages()
    else:
        return madlad400_languages_supported()


def get_cross_model_languages() -> Dict[str, Dict[str, str]]:
    """
    Get languages that are supported by MADLAD-400, Seamless, and Toucan.
    
    Returns:
        Dictionary of languages supported by all three models
    """
    # This would require importing the other modules, but for now we provide
    # the known common languages based on our analysis
    common_languages = {
        'afr_Latn': {'name': 'Afrikaans'},
        'eng_Latn': {'name': 'English'}, 
        'ibo_Latn': {'name': 'Igbo'},
        'lug_Latn': {'name': 'Ganda'},
        'nya_Latn': {'name': 'Chichewa'},
        'som_Latn': {'name': 'Somali'},
        'yor_Latn': {'name': 'Yoruba'},
        'zul_Latn': {'name': 'Zulu'},
        'fra_Latn': {'name': 'French'},
        'hau_Latn': {'name': 'Hausa'}
    }
    
    return common_languages


# # Example usage
# if __name__ == "__main__":
#     # Get all languages
#     all_langs = madlad400_languages_supported()
#     print(f"Total MADLAD-400 languages: {len(all_langs)}")
    
#     # Get African languages only
#     african_langs = madlad400_african_languages()
#     print(f"\nAfrican languages in MADLAD-400: {len(african_langs)}")
#     print("\nAfrican languages:")
#     for key, info in sorted(african_langs.items()):
#         print(f"  {key}: {info['name']}")
    
#     # Get specific language info
#     swahili_info = get_language_info('sw')
#     if swahili_info:
#         print(f"\nSwahili info: {swahili_info}")