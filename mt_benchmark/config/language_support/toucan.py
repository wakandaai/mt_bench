# mt_benchmark/config/language_support/toucan.py

from typing import Dict

lang_names={
    "aar": "Afar",
    "ach": "Acholi",
    "afr": "Afrikaans",
    "aka": "Akan",
    "amh": "Amharic",
    "bam": "Bambara",
    "bas": "Basaa",
    "bem": "Bemba",
    "btg": "Bete Gagnoa",
    "eng": "English",
    "ewe": "Ewe",
    "fon": "Fon",
    "fra": "French",
    "hau": "Hausa",
    "ibo": "Igbo",
    "kbp": "Kabiye",
    "lgg": "Lugbara",
    "lug": "Luganda",
    "mlg": "Malagasy",
    "nyn": "Nyakore",
    "orm": "Oromo",
    "som": "Somali",
    "sot": "Sesotho",
    "swa": "Swahili",
    "tir": "Tigrinya",
    "yor": "Yoruba",
    "teo": "Ateso",
    "gez": "Geez",
    "wal": "Wolaytta",
    "fan": "Fang",
    "kau": "Kanuri",
    "kin": "Kinyawanda",
    "kon": "Kongo",
    "lin": "Lingala",
    "nya": "Chichewa",
    "pcm": "Nigerian Pidgin",
    "ssw": "Siswati",
    "tsn": "Setswana",
    "tso": "Tsonga",
    "twi": "Twi",
    "wol": "Wolof",
    "xho": "Xhosa",
    "zul": "Zulu",
    "nnb": "Nande",
    "swc": "Swahili Congo",
    "ara": "Arabic"
}

script_mappings = {
    "aar": "Latn",  # Afar
    "ach": "Latn",  # Acholi
    "afr": "Latn",  # Afrikaans
    "aka": "Latn",  # Akan
    "amh": "Ethi",  # Amharic
    "bam": "Latn",  # Bambara
    "bas": "Latn",  # Basaa
    "bem": "Latn",  # Bemba
    "btg": "Latn",  # Bete Gagnoa
    "eng": "Latn",  # English
    "ewe": "Latn",  # Ewe
    "fon": "Latn",  # Fon
    "fra": "Latn",  # French
    "hau": "Latn",  # Hausa
    "ibo": "Latn",  # Igbo
    "kbp": "Latn",  # Kabiye
    "lgg": "Latn",  # Lugbara
    "lug": "Latn",  # Luganda
    "mlg": "Latn",  # Malagasy
    "nyn": "Latn",  # Nyakore
    "orm": "Latn",  # Oromo
    "som": "Latn",  # Somali
    "sot": "Latn",  # Sesotho
    "swa": "Latn",  # Swahili
    "tir": "Ethi",  # Tigrinya
    "yor": "Latn",  # Yoruba
    "teo": "Latn",  # Ateso
    "gez": "Ethi",  # Geez
    "wal": "Latn",  # Wolaytta
    "fan": "Latn",  # Fang
    "kau": "Latn",  # Kanuri
    "kin": "Latn",  # Kinyawanda
    "kon": "Latn",  # Kongo
    "lin": "Latn",  # Lingala
    "nya": "Latn",  # Chichewa
    "pcm": "Latn",  # Nigerian Pidgin
    "ssw": "Latn",  # Siswati
    "tsn": "Latn",  # Setswana
    "tso": "Latn",  # Tsonga
    "twi": "Latn",  # Twi
    "wol": "Latn",  # Wolof
    "xho": "Latn",  # Xhosa
    "zul": "Latn",  # Zulu
    "nnb": "Latn",  # Nande
    "swc": "Latn",  # Swahili Congo
    "ara": "Arab"   # Arabic
}

TOUCAN_TO_FLORES_MAPPING = {
    "swa_Latn": "swh_Latn", # Swahili
    "twi_Latn": "twi_Latn_akua1239", # Akuapem Twi
    "ara_Arab": "arb_Arab",  # Arabic (Modern Standard Arabic)
    "orm_Latn": "gaz_Latn",  # Oromo (West Central Oromo)
    "mlg_Latn": "plt_Latn",  # Malagasy (Plateau Malagasy)
    "kau_Latn": "knc_Latn",  # Kanuri (Central Kanuri)
}

def toucan_languages_supported() -> Dict[str, Dict[str, str]]:
    """Get a dictionary of all supported languages with ISO 693-3 + script code keys."""
    
    language_dict = {}
    
    for iso_code, name in lang_names.items():
        script = script_mappings.get(iso_code, "Latn")  # Default to Latin if not specified
        key = f"{iso_code}_{script}"
        
        language_dict[key] = {
            'name': name
        }
    
    return language_dict