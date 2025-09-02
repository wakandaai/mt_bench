# mt_benchmark/config/language_support/code_mapping.py

import csv
from io import StringIO
from typing import Dict, List, Set
from collections import defaultdict

data = """ISO-639-1	ISO-639-3	Script	Glottocode	Name	Notes
	ace	Arab	achi1257	Acehnese (Jawi script)	
	ace	Latn	achi1257	Acehnese (Latin script)	
	acm	Arab	meso1252	Mesopotamian Arabic	
	acq	Arab	taiz1242	Taʽizzi-Adeni Arabic	
	aeb	Arab	tuni1259	Tunisian Arabic	
af	afr	Latn	afri1274	Afrikaans	
sq	als	Latn	tosk1239	Albanian (Tosk)	
am	amh	Ethi	amha1245	Amharic	
	apc	Arab	nort3139	Levantine Arabic (North)	
	apc	Arab	sout3123	Levantine Arabic (South)	
ar	arb	Arab	stan1318	Modern Standard Arabic	
ar	arb	Latn	stan1318	Modern Standard Arabic (Romanized)	
an	arg	Latn	arag1245	Aragonese	
	ars	Arab	najd1235	Najdi Arabic	
	ary	Arab	moro1292	Moroccan Arabic	
	arz	Arab	egyp1253	Egyptian Arabic	
as	asm	Beng	assa1263	Assamese	
	ast	Latn	astu1245	Asturian	
	awa	Deva	awad1243	Awadhi	
ay	ayr	Latn	cent2142	Central Aymara	
	azb	Arab	sout2697	South Azerbaijani	
az	azj	Latn	nort2697	North Azerbaijani	
ba	bak	Cyrl	bash1264	Bashkir	
bm	bam	Latn	bamb1269	Bambara	
	ban	Latn	bali1278	Balinese	
be	bel	Cyrl	bela1254	Belarusian	
	bem	Latn	bemb1257	Bemba	
bn	ben	Beng	beng1280	Bengali	
	bho	Deva	bhoj1244	Bhojpuri	
	bjn	Arab	banj1239	Banjar (Jawi script)	
	bjn	Latn	banj1239	Banjar (Latin script)	
bo	bod	Tibt	utsa1239	Lhasa Tibetan	
bs	bos	Latn	bosn1245	Bosnian	
	brx	Deva	bodo1269	Bodo	dev only
	bug	Latn	bugi1244	Buginese	
bg	bul	Cyrl	bulg1262	Bulgarian	
ca	cat	Latn	stan1289	Catalan	
ca	cat	Latn	vale1252	Valencian	
	ceb	Latn	cebu1242	Cebuano	
cs	ces	Latn	czec1258	Czech	
cv	chv	Cyrl	chuv1255	Chuvash	
	cjk	Latn	chok1245	Chokwe	
	ckb	Arab	cent1972	Central Kurdish	
zh	cmn	Hans	beij1234	Mandarin Chinese (Standard Beijing)	
zh	cmn	Hant	taib1240	Mandarin Chinese (Taiwanese)	
	crh	Latn	crim1257	Crimean Tatar	
cy	cym	Latn	wels1247	Welsh	
da	dan	Latn	dani1285	Danish	
	dar	Cyrl	darg1241	Dargwa	dev only
de	deu	Latn	stan1295	German	
	dgo	Deva	dogr1250	Dogri	dev only
	dik	Latn	sout2832	Southwestern Dinka	
	dyu	Latn	dyul1238	Dyula	
dz	dzo	Tibt	dzon1239	Dzongkha	
et	ekk	Latn	esto1258	Estonian	
el	ell	Grek	mode1248	Greek	
en	eng	Latn	stan1293	English	
eo	epo	Latn	espe1235	Esperanto	
eu	eus	Latn	basq1248	Basque	
ee	ewe	Latn	ewee1241	Ewe	
fo	fao	Latn	faro1244	Faroese	
fj	fij	Latn	fiji1243	Fijian	
	fil	Latn	fili1244	Filipino	
fi	fin	Latn	finn1318	Finnish	
	fon	Latn	fonn1241	Fon	
fr	fra	Latn	stan1290	French	
	fur	Latn	east2271	Friulian	
ff	fuv	Latn	nige1253	Nigerian Fulfulde	
	gaz	Latn	west2721	West Central Oromo	
gd	gla	Latn	scot1245	Scottish Gaelic	
ga	gle	Latn	iris1253	Irish	
gl	glg	Latn	gali1258	Galician	
	gom	Deva	goan1235	Goan Konkani	
	gug	Latn	para1311	Paraguayan Guaraní	
gu	guj	Gujr	guja1252	Gujarati	
ht	hat	Latn	hait1244	Haitian Creole	
ha	hau	Latn	haus1257	Hausa	
he	heb	Hebr	hebr1245	Hebrew	
hi	hin	Deva	hind1269	Hindi	
	hne	Deva	chha1249	Chhattisgarhi	
hr	hrv	Latn	croa1245	Croatian	
hu	hun	Latn	hung1274	Hungarian	
hy	hye	Armn	nucl1235	Armenian	
ig	ibo	Latn	nucl1417	Igbo	
	ilo	Latn	ilok1237	Ilocano	
id	ind	Latn	indo1316	Indonesian	
is	isl	Latn	icel1247	Icelandic	
it	ita	Latn	ital1282	Italian	
jv	jav	Latn	java1254	Javanese	
ja	jpn	Jpan	nucl1643	Japanese	
	kaa	Latn	kara1467	Karakalpak	devtest only
	kab	Latn	kaby1243	Kabyle	
	kac	Latn	kach1280	Jingpho	
	kam	Latn	kamb1297	Kamba	
kn	kan	Knda	nucl1305	Kannada	
ks	kas	Arab	kash1277	Kashmiri (Arabic script)	
ks	kas	Deva	kash1277	Kashmiri (Devanagari script)	
ka	kat	Geor	nucl1302	Georgian	
kk	kaz	Cyrl	kaza1248	Kazakh	
	kbp	Latn	kabi1261	Kabiyè	
	kea	Latn	kabu1256	Kabuverdianu	
	khk	Cyrl	halh1238	Halh Mongolian	
km	khm	Khmr	cent1989	Khmer (Central)	
ki	kik	Latn	kiku1240	Kikuyu	
rw	kin	Latn	kiny1244	Kinyarwanda	
ky	kir	Cyrl	kirg1245	Kyrgyz	
	kmb	Latn	kimb1241	Kimbundu	
	kmr	Latn	nort2641	Northern Kurdish	
	knc	Arab	cent2050	Central Kanuri (Arabic script)	
	knc	Latn	cent2050	Central Kanuri (Latin script)	
ko	kor	Hang	kore1280	Korean	
	ktu	Latn	kitu1246	Kituba (DRC)	
lo	lao	Laoo	laoo1244	Lao	
	lij	Latn	geno1240	Ligurian (Genoese)	
li	lim	Latn	limb1263	Limburgish	
ln	lin	Latn	ling1263	Lingala	
lt	lit	Latn	lith1251	Lithuanian	
	lld	Latn	ladi1250	Ladin (Val Badia)	
	lmo	Latn	lomb1257	Lombard	[1]
	ltg	Latn	east2282	Latgalian	
lb	ltz	Latn	luxe1241	Luxembourgish	
	lua	Latn	luba1249	Luba-Kasai	
lg	lug	Latn	gand1255	Ganda	
	luo	Latn	luok1236	Luo	
	lus	Latn	lush1249	Mizo	
lv	lvs	Latn	stan1325	Standard Latvian	
	mag	Deva	maga1260	Magahi	
	mai	Deva	mait1250	Maithili	
ml	mal	Mlym	mala1464	Malayalam	
mr	mar	Deva	mara1378	Marathi	
	mhr	Cyrl	gras1239	Meadow Mari	dev only
	min	Arab	mina1268	Minangkabau (Jawi script)	
	min	Latn	mina1268	Minangkabau (Latin script)	
mk	mkd	Cyrl	mace1250	Macedonian	
mt	mlt	Latn	malt1254	Maltese	
	mni	Beng	mani1292	Meitei (Manipuri, Bengali script)	
	mni	Mtei	mani1292	Meitei (Manipuri, Meitei script)	dev only
	mos	Latn	moss1236	Mossi	
mi	mri	Latn	maor1246	Maori	
my	mya	Mymr	nucl1310	Burmese	
	myv	Cyrl	erzy1239	Erzya	
nl	nld	Latn	dutc1256	Dutch	
nn	nno	Latn	norw1262	Norwegian Nynorsk	
nb	nob	Latn	norw1259	Norwegian Bokmål	
ne	npi	Deva	nepa1254	Nepali	
	nqo	Nkoo	nkoa1234	Nko	
	nso	Latn	pedi1238	Northern Sotho	
	nus	Latn	nuer1246	Nuer	
ny	nya	Latn	nyan1308	Nyanja	
oc	oci	Latn	occi1239	Occitan	
oc	oci	Latn	aran1260	Aranese	
or	ory	Orya	oriy1255	Odia	
	pag	Latn	pang1290	Pangasinan	
pa	pan	Guru	panj1256	Eastern Panjabi	
	pap	Latn	papi1253	Papiamento	
	pbt	Arab	sout2649	Southern Pashto	
fa	pes	Arab	west2369	Western Persian	
	plt	Latn	plat1254	Plateau Malagasy	
pl	pol	Latn	poli1260	Polish	
pt	por	Latn	braz1246	Portuguese (Brazilian)	
	prs	Arab	dari1249	Dari	
qu	quy	Latn	ayac1239	Ayacucho Quechua	
ro	ron	Latn	roma1327	Romanian	
rn	run	Latn	rund1242	Rundi	
ru	rus	Cyrl	russ1263	Russian	
sg	sag	Latn	sang1328	Sango	
sa	san	Deva	sans1269	Sanskrit	
	sat	Olck	sant1410	Santali	
	scn	Latn	sici1248	Sicilian	
	shn	Mymr	shan1277	Shan	
si	sin	Sinh	sinh1246	Sinhala	
sk	slk	Latn	slov1269	Slovak	
sl	slv	Latn	slov1268	Slovenian	
sm	smo	Latn	samo1305	Samoan	
sn	sna	Latn	shon1251	Shona	
sd	snd	Arab	sind1272	Sindhi (Arabic script)	
sd	snd	Deva	sind1272	Sindhi (Devanagari script)	dev only
so	som	Latn	soma1255	Somali	
st	sot	Latn	sout2807	Southern Sotho	
es	spa	Latn	amer1254	Spanish (Latin American)	
	srd	Latn	sard1257	Sardinian	[1]
sr	srp	Cyrl	serb1264	Serbian	
ss	ssw	Latn	swat1243	Swati	
su	sun	Latn	sund1252	Sundanese	
sv	swe	Latn	swed1254	Swedish	
sw	swh	Latn	swah1253	Swahili	
	szl	Latn	sile1253	Silesian	
ta	tam	Taml	tami1289	Tamil	
	taq	Latn	tama1365	Tamasheq (Latin script)	
	taq	Tfng	tama1365	Tamasheq (Tifinagh script)	
tt	tat	Cyrl	tata1255	Tatar	
te	tel	Telu	telu1262	Telugu	
tg	tgk	Cyrl	taji1245	Tajik	
th	tha	Thai	thai1261	Thai	
ti	tir	Ethi	tigr1271	Tigrinya	
	tpi	Latn	tokp1240	Tok Pisin	
tn	tsn	Latn	tswa1253	Tswana	
ts	tso	Latn	tson1249	Tsonga	
tk	tuk	Latn	turk1304	Turkmen	
	tum	Latn	tumb1250	Tumbuka	
tr	tur	Latn	nucl1301	Turkish	
tw	twi	Latn	akua1239	Akuapem Twi	
tw	twi	Latn	asan1239	Asante Twi	
	tyv	Cyrl	tuvi1240	Tuvan	
ug	uig	Arab	uigh1240	Uyghur	
uk	ukr	Cyrl	ukra1253	Ukrainian	
	umb	Latn	umbu1257	Umbundu	
ur	urd	Arab	urdu1245	Urdu	
uz	uzn	Latn	nort2690	Northern Uzbek	
	vec	Latn	vene1259	Venetian	
vi	vie	Latn	viet1252	Vietnamese	
	vmw	Latn	cent2033	Emakhuwa (Central)	
	war	Latn	wara1300	Waray	
wo	wol	Latn	nucl1347	Wolof	
	wuu	Hans	suhu1238	Wu Chinese	dev only
xh	xho	Latn	xhos1239	Xhosa	
yi	ydd	Hebr	east2295	Eastern Yiddish	
yo	yor	Latn	yoru1245	Yoruba	
	yue	Hant	xian1255	Yue Chinese (Hong Kong Cantonese)	
	zgh	Tfng	stan1324	Standard Moroccan Tamazight	
ms	zsm	Latn	stan1306	Standard Malay	
zu	zul	Latn	zulu1248	Zulu
aa	aar	Latn	afar1241	Afar	
	ach	Latn	acho1249	Acholi	
ak	aka	Latn	akan1250	Akan	
	bas	Latn	basa1284	Basaa	
	btg	Latn	bete1243	Bete Gagnoa	
	fan	Latn	fang1246	Fang	
	gez	Ethi	geez1241	Geez	
kg	kon	Latn	kiko1239	Kongo	
	lgg	Latn	lugb1240	Lugbara	
	nnb	Latn	nand1266	Nande	
	nyn	Latn	nyan1309	Nyakore	
	pcm	Latn	nige1257	Nigerian Pidgin	
	swc	Latn	kong1291	Swahili Congo	
	teo	Latn	ates1237	Ateso	
	wal	Ethi	wola1242	Wolaytta	"""

def iso639_3_to_iso639_1() -> Dict[str, str]:
    """Get a mapping from ISO 639-3 codes to ISO 639-1 codes.
    
    Returns:
        Dict mapping ISO 639-3 codes to their corresponding ISO 639-1 codes.
        Only includes entries where both codes exist.
    """
    mapping = {}
    reader = csv.reader(StringIO(data), delimiter='\t')
    next(reader)  # Skip header
    
    for row in reader:
        if len(row) >= 6:  # Use >= to handle potential extra columns
            iso_639_1 = row[0].strip()  # Use index instead of key
            iso_639_3 = row[1].strip()  # Use index instead of key
            if iso_639_1 and iso_639_3:  # Both must be non-empty
                mapping[iso_639_3] = iso_639_1
        # Don't raise error for short rows - just skip them
    
    return mapping

def iso639_1_to_iso639_3() -> Dict[str, List[str]]:
    """Get a mapping from ISO 639-1 codes to ISO 639-3 codes.
    
    Note: Returns a list because one ISO 639-1 code can map to multiple 
    ISO 639-3 codes (e.g., 'zh' maps to 'cmn', 'yue', 'wuu', etc.)
    
    Returns:
        Dict mapping ISO 639-1 codes to lists of corresponding ISO 639-3 codes.
    """
    mapping = defaultdict(list)
    reader = csv.reader(StringIO(data), delimiter='\t')
    next(reader)  # Skip header
    
    for row in reader:
        if len(row) >= 6:
            iso_639_1 = row[0].strip()
            iso_639_3 = row[1].strip()
            if iso_639_1 and iso_639_3:
                if iso_639_3 not in mapping[iso_639_1]:  # Avoid duplicates
                    mapping[iso_639_1].append(iso_639_3)
    
    # Convert defaultdict to regular dict
    return dict(mapping)

def iso639_1_to_iso639_3_and_script() -> Dict[str, List[str]]:
    """Get a mapping from ISO 639-1 codes to ISO 639-3 and script combinations.
    
    Returns:
        Dict mapping ISO 639-1 codes to lists of strings in format 'iso639_3_Script'.
        Example: {'en': ['eng_Latn'], 'zh': ['cmn_Hans', 'cmn_Hant'], ...}
    """
    mapping = defaultdict(list)
    reader = csv.reader(StringIO(data), delimiter='\t')
    next(reader)  # Skip header
    
    for row in reader:
        if len(row) >= 6:
            iso_639_1 = row[0].strip()
            iso_639_3 = row[1].strip() 
            script = row[2].strip()
            if iso_639_1 and iso_639_3 and script:
                combined = f"{iso_639_3}_{script}"
                if combined not in mapping[iso_639_1]:  # Avoid duplicates
                    mapping[iso_639_1].append(combined)
    
    return dict(mapping)

def get_all_iso639_3_codes() -> Set[str]:
    """Get all unique ISO 639-3 codes from the dataset.
    
    Returns:
        Set of all ISO 639-3 codes present in the data.
    """
    codes = set()
    reader = csv.reader(StringIO(data), delimiter='\t')
    next(reader)  # Skip header
    
    for row in reader:
        if len(row) >= 2:
            iso_639_3 = row[1].strip()
            if iso_639_3:
                codes.add(iso_639_3)
    
    return codes

def get_all_iso639_1_codes() -> Set[str]:
    """Get all unique ISO 639-1 codes from the dataset.
    
    Returns:
        Set of all ISO 639-1 codes present in the data.
    """
    codes = set()
    reader = csv.reader(StringIO(data), delimiter='\t')
    next(reader)  # Skip header
    
    for row in reader:
        if len(row) >= 1:
            iso_639_1 = row[0].strip()
            if iso_639_1:
                codes.add(iso_639_1)
    
    return codes

def get_languages_by_script(script_code: str) -> List[Dict[str, str]]:
    """Get all languages that use a specific script.
    
    Args:
        script_code: The script code to filter by (e.g., 'Latn', 'Arab', 'Cyrl')
    
    Returns:
        List of dicts containing language information for the specified script.
    """
    languages = []
    reader = csv.reader(StringIO(data), delimiter='\t')
    header = next(reader)  # Get header for column names
    
    for row in reader:
        if len(row) >= 6:
            if row[2].strip() == script_code:  # Script column
                lang_info = {
                    'iso639_1': row[0].strip(),
                    'iso639_3': row[1].strip(),
                    'script': row[2].strip(),
                    'glottocode': row[3].strip(),
                    'name': row[4].strip(),
                    'notes': row[5].strip() if len(row) > 5 else ''
                }
                languages.append(lang_info)
    return languages