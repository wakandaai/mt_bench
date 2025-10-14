#mt_benchmark/config/language_support/madlad_400.py

import csv
from io import StringIO
from typing import Dict 

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
    Get a dictionary of all MADLAD-400 supported languages.
    
    Returns:
        Dict with language codes as keys and metadata as values.
        Format: {code_script: {'name': str}}
    """
    
    # Parse the TSV data
    reader = csv.DictReader(StringIO(data), delimiter='\t')
    
    language_dict = {}
    
    for row in reader:
        code = row['code'].strip()
        name = row['language'].strip()
        script = row['script'].strip()
        
        # Create key in format: code_script (e.g., 'en_Latn')
        key = f"{code}_{script}"
        
        language_dict[key] = {
            'name': name
        }
    
    return language_dict


def madlad400_african_languages() -> Dict[str, Dict[str, str]]:
    """
    Get only African languages from MADLAD-400.
    
    Returns:
        Dict with African language codes as keys and metadata as values.
    """
    
    african_language_codes = {
        'af', 'am', 'ha', 'ig', 'yo', 'sw', 'so', 'om', 'rw', 
        'zu', 'xh', 'sn', 'ny', 'st', 'tn', 'ts', 've', 'lg',
        'ee', 'ti', 'fon', 'nso', 'ak', 'ln', 'kbp', 'lu',
        'tiv', 'wo', 'bm', 'ff', 'rn', 'kg', 'mg'
    }
    
    all_languages = madlad400_languages_supported()
    
    african_dict = {}
    for key, value in all_languages.items():
        code = key.split('_')[0]
        if code in african_language_codes:
            african_dict[key] = value
    
    return african_dict


def get_language_info(language_code: str) -> Dict[str, str]:
    """
    Get information about a specific language.
    
    Args:
        language_code: The BCP-47 language code (e.g., 'sw', 'yo', 'en')
    
    Returns:
        Dictionary with language information or None if not found
    """
    all_languages = madlad400_languages_supported()
    
    for key, value in all_languages.items():
        if key.startswith(f"{language_code}_"):
            return {
                'code': language_code,
                'key': key,
                **value
            }
    
    return None


# Example usage
if __name__ == "__main__":
    # Get all languages
    all_langs = madlad400_languages_supported()
    print(f"Total MADLAD-400 languages: {len(all_langs)}")
    
    # Get African languages only
    african_langs = madlad400_african_languages()
    print(f"\nAfrican languages in MADLAD-400: {len(african_langs)}")
    print("\nAfrican languages:")
    for key, info in sorted(african_langs.items()):
        print(f"  {key}: {info['name']}")
    
    # Get specific language info
    swahili_info = get_language_info('sw')
    if swahili_info:
        print(f"\nSwahili info: {swahili_info}")