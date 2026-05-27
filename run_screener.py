import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import logging
import time
from pathlib import Path
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================
CACHE_DATA = True
CACHE_DIR = Path("stock_cache")
CACHE_DIR.mkdir(exist_ok=True)

STATIC_DATA_FILE = Path("static_data.csv")
BATCH_SIZE = 50
MAX_RETRIES = 3

# Market Cap Classification Thresholds (in INR Crores)
LARGE_CAP_THRESHOLD = 20000
MID_CAP_THRESHOLD = 5000

# Custom Score Configuration
SCORE_CONFIG = {
    'short_term': {
        'early_detection_weight': 0.5,
        'momentum_weight': 0.3,
        'trend_weight': 0.2
    },
    'balanced': {
        'early_detection_weight': 0.3,
        'momentum_weight': 0.35,
        'trend_weight': 0.35
    },
    'long_term': {
        'early_detection_weight': 0.2,
        'momentum_weight': 0.3,
        'trend_weight': 0.5
    }
}

# ============================================
# SYMBOL LIST
# ============================================

RAW_SYMBOLS = [
    "20MICRONS", "21STCENMGM", "360ONE", "3MINDIA", "3PLAND", "5PAISA", "63MOONS",
    "AAATECH", "AADHARHFC", "AAREYDRUGS", "AARON", "AARTECH", "AARTIDRUGS", "AARTIIND",
    "AARTIPHARM", "AARTISURF", "AARVI", "AAVAS", "ABB", "ABBOTINDIA", "ABCAPITAL",
    "ABCOTS", "ABDL", "ABFRL", "ABLBL", "ABMINTLLTD", "ABREL", "ABSLAMC", "ACC",
    "ACCELYA", "ACE", "ACI", "ACL", "ACMESOLAR", "ACUTAAS", "ADANIENSOL", "ADANIENT",
    "ADANIGREEN", "ADANIPORTS", "ADANIPOWER", "ADFFOODS", "ADL", "ADOR", "ADSL",
    "ADVANCE", "ADVANIHOTR", "ADVENTHTL", "ADVENZYMES", "AEGISLOG", "AEGISVOPAK",
    "AEQUS", "AEROENTER", "AEROFLEX", "AERONEU", "AETHER", "AFCONS", "AFFLE",
    "AFFORDABLE", "AFSL", "AGARIND", "AGARWALEYE", "AGI", "AGIIL", "AGRITECH",
    "AGROPHOS", "AHCL", "AHLADA", "AHLEAST", "AHLUCONT", "AIAENG", "AIIL",
    "AIROLAM", "AJANTPHARM", "AJAXENGG", "AJMERA", "AKSHARCHEM", "AKUMS",
    "AKZOINDIA", "ALBERTDAVD", "ALEMBICLTD", "ALICON", "ALIVUS", "ALKALI",
    "ALKEM", "ALKYLAMINE", "ALLDIGI", "ALLTIME", "ALOKINDS", "ALPA",
    "ALPHAGEO", "AMANTA", "AMBER", "AMBIKCO", "AMBUJACEM", "AMDIND",
    "AMJLAND", "AMNPLST", "AMRUTANJAN", "ANANDRATHI", "ANANTRAJ", "ANDHRAPAP",
    "ANDHRSUGAR", "ANGELONE", "ANIKINDS", "ANTELOPUS", "ANTHEM", "ANUHPHR",
    "ANUP", "ANURAS", "APARINDS", "APCL", "APCOTEXIND", "APEX", "APLAPOLLO",
    "APLLTD", "APOLLO", "APOLLOHOSP", "APOLLOPIPE", "APOLLOTYRE", "APOLSINHOT",
    "APTECHT", "APTUS", "ARCHIDPLY", "ARE&M", "ARENTERP", "ARFIN", "ARIES",
    "ARIHANTCAP", "ARIHANTSUP", "ARISINFRA", "ARKADE", "ARMANFIN", "AROGRANITE",
    "ARROWGREEN", "ARSSBL", "ARTEMISMED", "ARTNIRMAN", "ARVEE", "ARVIND",
    "ARVINDFASN", "ARVSMART", "ASAHIINDIA", "ASAHISONG", "ASAL", "ASALCBR",
    "ASHAPURMIN", "ASHIANA", "ASHOKA", "ASHOKLEY", "ASIANENE", "ASIANHOTNR",
    "ASIANPAINT", "ASIANTILES", "ASKAUTOLTD", "ASPINWALL", "ASTEC", "ASTERDM",
    "ASTRAL", "ASTRAMICRO", "ASTRAZEN", "ATALREAL", "ATAM", "ATGL", "ATHERENERG",
    "ATLANTAA", "ATLANTAELE", "ATLASCYCLE", "ATUL", "ATULAUTO", "AUBANK",
    "AURIONPRO", "AUROPHARMA", "AURUM", "AUSOMENT", "AUTOAXLES", "AUTOIND",
    "AVADHSUGAR", "AVALON", "AVANTEL", "AVANTIFEED", "AVG", "AVL", "AVROIND",
    "AVTNPL", "AWFIS", "AWHCL", "AWL", "AXISBANK", "AXISCADES", "AYMSYNTEX",
    "AZAD", "BAFNAPH", "BAJAJ-AUTO", "BAJAJCON", "BAJAJELEC", "BAJAJFINSV",
    "BAJAJHCARE", "BAJAJHFL", "BAJAJHLDNG", "BAJAJINDEF", "BAJEL", "BAJFINANCE",
    "BALAJEE", "BALAJITELE", "BALAMINES", "BALAXI", "BALKRISIND", "BALMLAWRIE",
    "BALPHARMA", "BALRAMCHIN", "BALUFORGE", "BANARBEADS", "BANARISUG",
    "BANCOINDIA", "BANDHANBNK", "BANG", "BANKA", "BANKBARODA", "BANKINDIA",
    "BANSALWIRE", "BANSWRAS", "BASF", "BATAINDIA", "BAYERCROP", "BBL", "BBOX",
    "BBTC", "BBTCL", "BCLIND", "BCONCEPTS", "BDL", "BEARDSELL", "BECTORFOOD",
    "BEDMUTHA", "BEL", "BELLACASA", "BELRISE", "BEML", "BEPL", "BERGEPAINT",
    "BETA", "BFINVEST", "BFUTILITIE", "BGRENERGY", "BHAGCHEM", "BHAGERIA",
    "BHAGYANGR", "BHARATCOAL", "BHARATFORG", "BHARATGEAR", "BHARATRAS",
    "BHARATSE", "BHARATWIRE", "BHARTIARTL", "BHARTIHEXA", "BHEL", "BIGBLOC",
    "BIKAJI", "BIL", "BIOCON", "BIOFILCHEM", "BIRLACABLE", "BIRLACORPN",
    "BIRLAMONEY", "BIRLANU", "BLACKBUCK", "BLAL", "BLISSGVS", "BLKASHYAP",
    "BLS", "BLSE", "BLUECOAST", "BLUEDART", "BLUEJET", "BLUESTARCO",
    "BLUESTONE", "BLUSPRING", "BMWVENTLTD", "BODALCHEM", "BOMDYEING", "BOROLTD",
    "BORORENEW", "BOROSCI", "BOSCHLTD", "BPCL", "BPL", "BRIGADE",
    "BRIGHOTEL", "BRITANNIA", "BROOKS", "BSE", "BSHSL", "BSL", "BSOFT",
    "BUILDPRO", "BUTTERFLY", "BVCL", "BYKE", "CAMLINFINE", "CAMPUS", "CAMS",
    "CANBK", "CANFINHOME", "CANHLIFE", "CANTABIL", "CAPACITE", "CAPILLARY",
    "CAPITALSFB", "CAPLIPOINT", "CARBORUNIV", "CARERATING", "CARRARO",
    "CARTRADE", "CARYSIL", "CASTROLIND", "CCL", "CDSL", "CEATLTD", "CEIGALL",
    "CELLO", "CEMPRO", "CENTENKA", "CENTRALBK", "CENTUM", "CENTURYPLY", "CERA",
    "CESC", "CEWATER", "CGCL", "CGPOWER", "CHALET", "CHAMBLFERT", "CHEMBOND",
    "CHEMBONDCH", "CHEMCON", "CHEMFAB", "CHEMPLASTS", "CHENNPETRO", "CHEVIOT",
    "CHOICEIN", "CHOLAFIN", "CHOLAHLDNG", "CIEINDIA", "CIFL", "CIGNITITEC",
    "CINELINE", "CIPLA", "CLEAN", "CLEDUCATE", "CLSEL", "CMSINFO", "CNL",
    "COALINDIA", "COASTCORP", "COCHINSHIP", "COFFEEDAY", "COFORGE", "COHANCE",
    "COLPAL", "COMSYN", "CONCOR", "CONCORDBIO", "CONFIPET", "CONSOFINVT",
    "CONTROLPR", "CORALFINAC", "CORDSCABLE", "COROMANDEL", "CORONA",
    "COSMOFIRST", "CPCAP", "CPEDU", "CPPLUS", "CRAFTSMAN", "CRAMC", "CREDITACC",
    "CREST", "CRISIL", "CRIZAC", "CROMPTON", "CROWN", "CSBBANK", "CSLFINANCE",
    "CTE", "CUB", "CUBEXTUB", "CUMMINSIND", "CUPID", "CURAA", "CYBERTECH",
    "CYIENT", "CYIENTDLM", "DABUR", "DALBHARAT", "DALMIASUG", "DAMCAPITAL",
    "DAMODARIND", "DATAMATICS", "DATAPATTNS", "DBCORP", "DBEIL", "DBL", "DBOL",
    "DBREALTY", "DCAL", "DCBBANK", "DCI", "DCM", "DCMNVL", "DCMSHRIRAM",
    "DCMSRIND", "DCW", "DCXINDIA", "DDEVPLSTIK", "DECCANCE", "DEEDEV",
    "DEEPAKFERT", "DEEPAKNTR", "DEEPINDS", "DELHIVERY", "DELPHIFX", "DELTACORP",
    "DELTAMAGNT", "DEN", "DENORA", "DENTA", "DEVIT", "DEVX", "DEVYANI",
    "DGCONTENT", "DHAMPURSUG", "DHANUKA", "DHARMAJ", "DHRUV", "DHUNINV",
    "DIACABS", "DIAMINESQ", "DIAMONDYD", "DICIND", "DIFFNKG", "DIGITIDE",
    "DIGJAMLMTD", "DIVGIITTS", "DIVISLAB", "DIXON", "DJML", "DLF", "DLINKINDIA",
    "DMART", "DMCC", "DODLA", "DOLATALGO", "DOLLAR", "DOLPHIN", "DOMS", "DONEAR",
    "DPABHUSHAN", "DPWIRES", "DREAMFOLKS", "DREDGECORP", "DRREDDY", "DSSL",
    "DTIL", "DVL", "DWARKESH", "DYCL", "DYNAMATECH", "DYNPRO", "E2E", "EBGNG",
    "ECLERX", "ECOSMOBLTY", "EDELWEISS", "EFCIL", "EICHERMOT", "EIDPARRY",
    "EIEL", "EIFFL", "EIHAHOTELS", "EIHOTEL", "EIMCOELECO", "EKC", "ELECON",
    "ELECTCAST", "ELECTHERM", "ELGIEQUIP", "ELGIRUBCO", "ELIN", "ELLEN",
    "EMAMILTD", "EMAMIPAP", "EMAMIREAL", "EMBDL", "EMCURE", "EMIL", "EMKAY",
    "EMMBI", "EMMVEE", "EMSLIMITED", "EMUDHRA", "ENDURANCE", "ENGINERSIN",
    "ENIL", "ENRIN", "ENTERO", "EPACK", "EPACKPEB", "EPIGRAL", "EPL",
    "EQUITASBNK", "ERIS", "ESABINDIA", "ESAFSFB", "ESCORTS", "ESTER", "ETERNAL",
    "ETHOSLTD", "EUREKAFORB", "EUROBOND", "EUROPRATIK", "EVEREADY", "EVERESTIND",
    "EXCELINDUS", "EXCELSOFT", "EXICOM", "EXIDEIND", "EXPLEOSOL", "FABTECH",
    "FACT", "FAIRCHEMOR", "FAZE3Q", "FDC", "FEDERALBNK", "FEDFINA", "FIBERWEB",
    "FIEMIND", "FILATEX", "FINCABLES", "FINEORG", "FINKURVE", "FINOPB",
    "FINPIPE", "FIRSTCRY", "FISCHER", "FIVESTAR", "FLAIR", "FLUOROCHEM",
    "FMGOETZE", "FOCUS", "FOODSIN", "FORCEMOT", "FORTIS", "FOSECOIND", "FSL",
    "FUSION", "GABRIEL", "GAEL", "GAIL", "GALAPREC", "GALAXYSURF", "GALLANTT",
    "GANDHAR", "GANDHITUBE", "GANECOS", "GANESHBE", "GANESHCP", "GANESHHOU",
    "GANGESSECU", "GARFIBRES", "GARUDA", "GATEWAY", "GCSL", "GEECEE",
    "GEEKAYWIRE", "GEMAROMA", "GENCON", "GENESYS", "GENSOL", "GENUSPOWER",
    "GEOJITFSL", "GESHIP", "GFLLIMITED", "GHCL", "GHCLTEXTIL", "GICHSGFIN",
    "GICRE", "GILLANDERS", "GILLETTE", "GINNIFILA", "GIPCL", "GKENERGY",
    "GKSL", "GLAND", "GLAXO", "GLENMARK", "GLOBAL", "GLOBALVECT", "GLOBECIVIL",
    "GLOBUSSPR", "GLOTTIS", "GMBREW", "GMDCLTD", "GMMPFAUDLR", "GMRAIRPORT",
    "GMRP&UI", "GNA", "GNFC", "GOACARBON", "GOCLCORP", "GOCOLORS", "GODAVARIB",
    "GODFRYPHLP", "GODIGIT", "GODREJAGRO", "GODREJCP", "GODREJIND", "GODREJPROP",
    "GOKEX", "GOKUL", "GOKULAGRO", "GOLDENTOBC", "GOLDIAM", "GOLDTECH",
    "GOODLUCK", "GOPAL", "GPIL", "GPPL", "GPTHEALTH", "GPTINFRA", "GRANULES",
    "GRAPHITE", "GRASIM", "GRAVITA", "GREAVESCOT", "GREENLAM", "GREENPANEL",
    "GREENPLY", "GRINDWELL", "GRINFRA", "GRMOVER", "GROBTEA", "GROWW", "GRPLTD",
    "GRSE", "GRWRHITECH", "GSFC", "GSLSU", "GSPL", "GTPL", "GUFICBIO",
    "GUJALKALI", "GUJAPOLLO", "GUJGASLTD", "GUJRAFFIA", "GUJTHEM", "GULFOILLUB",
    "GULFPETRO", "GULPOLY", "GVPIL", "GVT&D", "HAL", "HAPPSTMNDS", "HAPPYFORGE",
    "HARIOMPIPE", "HARRMALAYA", "HARSHA", "HATSUN", "HAVELLS", "HBLENGINE",
    "HBSL", "HCG", "HCLTECH", "HDBFS", "HDFCAMC", "HDFCBANK", "HDFCLIFE",
    "HECPROJECT", "HEG", "HEIDELBERG", "HEMIPROP", "HERANBA", "HERITGFOOD",
    "HEROMOTOCO", "HESTERBIO", "HEXATRADEX", "HEXT", "HFCL", "HGINFRA", "HGM",
    "HGS", "HIKAL", "HILINFRA", "HILTON", "HIMATSEIDE", "HINDALCO", "HINDCOMPOS",
    "HINDCOPPER", "HINDOILEXP", "HINDPETRO", "HINDUNILVR", "HINDWAREAP",
    "HINDZINC", "HIRECT", "HISARMETAL", "HITECH", "HITECHCORP", "HITECHGEAR",
    "HLEGLAS", "HMAAGRO", "HMT", "HMVL", "HNDFDS", "HOMEFIRST", "HONASA",
    "HONAUT", "HONDAPOWER", "HPAL", "HPIL", "HPL", "HSCL", "HUBTOWN", "HUDCO",
    "HUHTAMAKI", "HYUNDAI", "ICDSLTD", "ICEMAKE", "ICICIAMC", "ICICIBANK",
    "ICICIGI", "ICICIPRULI", "ICIL", "ICRA", "IDBI", "IDEAFORGE", "IDFCFIRSTB",
    "IEX", "IFBAGRO", "IFBIND", "IFCI", "IFGLEXPOR", "IGARASHI", "IGCL", "IGIL",
    "IGL", "IGPL", "IIFL", "IIFLCAPS", "IITL", "IKIO", "IKS", "IMAGICAA",
    "IMFA", "IMPAL", "INCREDIBLE", "INDBANK", "INDGN", "INDHOTEL", "INDIACEM",
    "INDIAGLYCO", "INDIAMART", "INDIANB", "INDIANCARD", "INDIANHUME",
    "INDIASHLTR", "INDIGO", "INDIGOPNTS", "INDIQUBE", "INDNIPPON", "INDOAMIN",
    "INDOBORAX", "INDOCO", "INDOFARM", "INDORAMA", "INDOSTAR", "INDOTECH",
    "INDOTHAI", "INDOUS", "INDRAMEDCO", "INDSWFTLAB", "INDTERRAIN", "INDUSINDBK",
    "INDUSTOWER", "INFOBEAN", "INFY", "INGERRAND", "INNOVACAP", "INNOVANA",
    "INOXGREEN", "INOXINDIA", "INOXWIND", "INSECTICID", "INSPIRISYS", "INTELLECT",
    "INTENTECH", "INTERARCH", "INTLCONV", "IOB", "IOC", "IOLCP", "IONEXCHANG",
    "IPCALAB", "IPL", "IRB", "IRCON", "IRCTC", "IREDA", "IRFC", "IRIS",
    "IRISDOREME", "IRMENERGY", "ISFT", "ISGEC", "ISHANCH", "ITC", "ITCHOTELS",
    "ITDC", "ITI", "IVALUE", "IVP", "IXIGO", "IZMO", "J&KBANK", "JAGRAN",
    "JAGSNPHARM", "JAIBALAJI", "JAICORPLTD", "JAINREC", "JAIPURKURT", "JAMNAAUTO",
    "JARO", "JASH", "JAYAGROGN", "JAYBARMARU", "JAYKAY", "JAYNECOIND",
    "JAYSREETEA", "JBCHEPHARM", "JBMA", "JGCHEM", "JINDALPHOT", "JINDALPOLY",
    "JINDALSAW", "JINDALSTEL", "JINDRILL", "JINDWORLD", "JIOFIN", "JISLJALEQS",
    "JITFINFRA", "JKCEMENT", "JKIL", "JKLAKSHMI", "JKPAPER", "JKTYRE", "JLHL",
    "JMA", "JMFINANCIL", "JNKINDIA", "JOCIL", "JPOLYINVST", "JPPOWER", "JSFB",
    "JSL", "JSLL", "JSWCEMENT", "JSWENERGY", "JSWHL", "JSWINFRA", "JSWSTEEL",
    "JTEKTINDIA", "JTLIND", "JUBLCPL", "JUBLFOOD", "JUBLINGREA", "JUBLPHARMA",
    "JUNIPER", "JUSTDIAL", "JWL", "JYOTHYLAB", "JYOTICNC", "KABRAEXTRU",
    "KAJARIACER", "KAKATCEM", "KALAMANDIR", "KALPATARU", "KALYANIFRG",
    "KALYANKJIL", "KAMATHOTEL", "KANORICHEM", "KANPRPLA", "KANSAINER",
    "KAPSTON", "KARMAENG", "KARURVYSYA", "KAVDEFENCE", "KAYA", "KAYNES", "KCP",
    "KDDL", "KEC", "KECL", "KEI", "KERNEX", "KEYFINSERV", "KFINTECH", "KHADIM",
    "KHAICHEM", "KHAITANLTD", "KICL", "KIMS", "KINGFA", "KIOCL", "KIRIINDUS",
    "KIRLOSBROS", "KIRLOSENG", "KIRLOSIND", "KIRLPNU", "KITEX", "KKCL", "KMEW",
    "KNAGRI", "KNRCON", "KOKUYOCMLN", "KOLTEPATIL", "KOPRAN", "KOTAKBANK",
    "KOTHARIPET", "KOTHARIPRO", "KPEL", "KPIGREEN", "KPIL", "KPITTECH",
    "KPRMILL", "KRBL", "KREBSBIO", "KRISHANA", "KRISHIVAL", "KRISHNADEF",
    "KRITI", "KRITINUT", "KRN", "KRONOX", "KROSS", "KRSNAA", "KRYSTAL", "KSB",
    "KSCL", "KSHINTL", "KSL", "KSOLVES", "KSR", "KTKBANK", "KUANTUM", "LAGNAM",
    "LALPATHLAB", "LAMBODHARA", "LANCORHOL", "LANDMARK", "LAOPALA", "LATENTVIEW",
    "LAURUSLABS", "LAXMIDENTL", "LAXMIINDIA", "LEMERITE", "LEMONTREE",
    "LENSKART", "LEXUS", "LFIC", "LGBBROSLTD", "LGEINDIA", "LGHL", "LIBERTSHOE",
    "LICHSGFIN", "LICI", "LIKHITHA", "LINC", "LINCOLN", "LINDEINDIA", "LLOYDSENGG",
    "LLOYDSENT", "LLOYDSME", "LMW", "LODHA", "LOKESHMACH", "LORDSCHLO",
    "LOTUSDEV", "LOTUSEYE", "LOVABLE", "LOYALTEX", "LT", "LTF", "LTFOODS",
    "LTIM", "LTTS", "LUMAXIND", "LUMAXTECH", "LUPIN", "LUXIND", "LXCHEM",
    "LYKALABS", "M&M", "M&MFIN", "MAANALU", "MACPOWER", "MADHAV", "MADRASFERT",
    "MAGADSUGAR", "MAHABANK", "MAHAPEXLTD", "MAHEPC", "MAHESHWARI", "MAHLIFE",
    "MAHLOG", "MAHSCOOTER", "MAHSEAMLES", "MAITHANALL", "MALLCOM", "MALUPAPER",
    "MAMATA", "MANAKALUCO", "MANAKCOAT", "MANAKSIA", "MANAKSTEEL", "MANALIPETC",
    "MANAPPURAM", "MANBA", "MANCREDIT", "MANGALAM", "MANGLMCEM", "MANINDS",
    "MANINFRA", "MANKIND", "MANOMAY", "MANORAMA", "MANORG", "MANYAVAR",
    "MAPMYINDIA", "MARALOVER", "MARATHON", "MARICO", "MARINE", "MARKOLINES",
    "MARKSANS", "MARUTI", "MASFIN", "MASKINVEST", "MASTEK", "MASTERTR",
    "MATRIMONY", "MAWANASUG", "MAXESTATES", "MAXHEALTH", "MAXIND", "MAYURUNIQ",
    "MAZDA", "MAZDOCK", "MBAPL", "MBEL", "MBLINFRA", "MCL", "MCLEODRUSS", "MCX",
    "MEDANTA", "MEDIASSIST", "MEDICAMEQ", "MEDICO", "MEDPLUS", "MEESHO",
    "MEGASOFT", "MEGASTAR", "MEIL", "MENONBE", "METROBRAND", "METROPOLIS",
    "MFSL", "MGL", "MHLXMIRU", "MHRIL", "MICEL", "MIDHANI", "MIDWESTLTD",
    "MINDACORP", "MINDTECK", "MIRCELECTR", "MIRZAINT", "MITCON", "MMFL", "MMP",
    "MMTC", "MOBIKWIK", "MODIRUBBER", "MODIS", "MODISONLTD", "MODTHREAD",
    "MOIL", "MOL", "MOLDTECH", "MOLDTKPAC", "MONARCH", "MONEYBOXX", "MONTECARLO",
    "MOREPENLAB", "MOSCHIP", "MOTHERSON", "MOTILALOFS", "MPHASIS", "MPSLTD",
    "MRF", "MRPL", "MSPL", "MSTCLTD", "MSUMI", "MTARTECH", "MTNL", "MUFIN",
    "MUFTI", "MUKANDLTD", "MUKTAARTS", "MUNJALAU", "MUNJALSHOW", "MURUDCERA",
    "MUTHOOTCAP", "MUTHOOTFIN", "MUTHOOTMF", "MVGJL", "MWL", "NACLIND",
    "NAGREEKCAP", "NAHARCAP", "NAHARINDUS", "NAHARPOLY", "NAHARSPING",
    "NAM-INDIA", "NARMADA", "NATCAPSUQ", "NATCOPHARM", "NATHBIOGEN",
    "NATIONALUM", "NAUKRI", "NAVA", "NAVINFLUOR", "NAVKARCORP", "NAVNETEDUL",
    "NAZARA", "NBCC", "NBIFIN", "NCC", "NCLIND", "NDGL", "NDLVENTURE",
    "NDRAUTO", "NDTV", "NELCAST", "NELCO", "NEOGEN", "NEPHROPLUS", "NESCO",
    "NESTLEIND", "NETWEB", "NETWORK18", "NEULANDLAB", "NEWGEN", "NFL", "NGIL",
    "NGLFINE", "NH", "NHPC", "NIACL", "NIBE", "NIBL", "NIITLTD", "NIITMTS",
    "NILKAMAL", "NINSYS", "NIPPOBATRY", "NIRAJ", "NIRAJISPAT", "NITCO",
    "NITINSPIN", "NITIRAJ", "NIVABUPA", "NKIND", "NLCINDIA", "NMDC", "NOCIL",
    "NORBTEAEXP", "NORTHARC", "NOVAAGRI", "NPST", "NRAIL", "NRBBEARING",
    "NRL", "NSIL", "NSLNISP", "NTPC", "NTPCGREEN", "NUCLEUS", "NURECA",
    "NUVAMA", "NUVOCO", "NYKAA", "OAL", "OBCL", "OBEROIRLTY", "OCCLLTD",
    "ODIGMA", "OFSS", "OIL", "OILCOUNTUB", "OLAELEC", "OLECTRA", "OMAXAUTO",
    "OMAXE", "OMFREIGHT", "OMINFRAL", "ONEPOINT", "ONESOURCE", "ONGC",
    "ONMOBILE", "ONWARDTEC", "OPTIEMUS", "ORBTEXP", "ORCHPHARMA", "ORICONENT",
    "ORIENTBELL", "ORIENTCEM", "ORIENTCER", "ORIENTELEC", "ORIENTHOT",
    "ORIENTLTD", "ORIENTTECH", "ORISSAMINE", "ORKLAINDIA", "OSWALAGRO",
    "OSWALGREEN", "OSWALPUMPS", "PACEDIGITK", "PAGEIND", "PAISALO", "PAKKA",
    "PALASHSECU", "PALREDTEC", "PANACEABIO", "PANAMAPET", "PAR", "PARACABLES",
    "PARADEEP", "PARAGMILK", "PARAS", "PARKHOSPS", "PARKHOTELS", "PASUPTAC",
    "PATANJALI", "PATELENG", "PATELRMART", "PAUSHAKLTD", "PAYTM", "PCBL",
    "PDMJEPAPER", "PDSL", "PENIND", "PERSISTENT", "PETRONET", "PFC", "PFIZER",
    "PFOCUS", "PFS", "PGEL", "PGHH", "PGHL", "PGIL", "PHOENIXLTD", "PICCADIL",
    "PIDILITIND", "PIGL", "PIIND", "PILANIINVS", "PINELABS", "PIONEEREMB",
    "PIRAMALFIN", "PITTIENG", "PIXTRANS", "PKTEA", "PLASTIBLEN", "PLATIND",
    "PLAZACABLE", "PNB", "PNBGILTS", "PNBHOUSING", "PNCINFRA", "PNGJL", "POCL",
    "PODDARMENT", "POKARNA", "POLICYBZR", "POLYCAB", "POLYMED", "POLYPLEX",
    "PONNIERODE", "POONAWALLA", "POWERGRID", "POWERINDIA", "POWERMECH", "PPAP",
    "PPL", "PPLPHARMA", "PRABHA", "PRAJIND", "PRAKASH", "PRECAM", "PRECOT",
    "PRECWIRE", "PREMEXPLN", "PREMIERENE", "PREMIERPOL", "PRESTIGE", "PRICOLLTD",
    "PRIMESECU", "PRINCEPIPE", "PRITI", "PRIVISCL", "PROSTARM", "PROTEAN",
    "PROZONER", "PRSMJOHNSN", "PRUDENT", "PSB", "PSPPROJECT", "PTC", "PTCIL",
    "PTL", "PUNJABCHEM", "PURVA", "PVP", "PVRINOX", "PVSL", "PWL", "PYRAMID",
    "QPOWER", "QUADFUTURE", "QUESS", "QUICKHEAL", "RACE", "RACLGEAR",
    "RADHIKAJWE", "RADIANTCMS", "RADICO", "RAILTEL", "RAIN", "RAINBOW",
    "RAJESHEXPO", "RAJOOENG", "RAJRATAN", "RAJSREESUG", "RAJTV", "RALLIS",
    "RAMANEWS", "RAMAPHO", "RAMCOCEM", "RAMCOIND", "RAMCOSYS", "RAMKY",
    "RAMRAT", "RANEHOLDIN", "RATEGAIN", "RATNAMANI", "RATNAVEER", "RAYMOND",
    "RAYMONDLSL", "RAYMONDREL", "RBA", "RBLBANK", "RBZJEWEL", "RCF", "RECLTD",
    "REDINGTON", "REDTAPE", "REFEX", "REGAAL", "REGENCERAM", "RELAXO",
    "RELCHEMQ", "RELIABLE", "RELIANCE", "RELIGARE", "RELTD", "REMSONSIND",
    "RENUKA", "REPCOHOME", "REPL", "REPRO", "RESPONIND", "RETAIL", "RGL",
    "RHETAN", "RHIM", "RHL", "RICOAUTO", "RIIL", "RISHABH", "RITCO", "RITES",
    "RKEC", "RKFORGE", "RKSWAMY", "RMDRIP", "RML", "RNBDENIMS", "ROHLTD",
    "ROLEXRINGS", "ROML", "ROSSARI", "ROSSELLIND", "ROSSTECH", "ROTO", "ROUTE",
    "RPEL", "RPGLIFE", "RPOWER", "RPPINFRA", "RPSGVENT", "RPTECH", "RRKABEL",
    "RSSOFTWARE", "RSWM", "RSYSTEMS", "RTNINDIA", "RUBFILA", "RUBICON",
    "RUBYMILLS", "RUCHIRA", "RUPA", "RUSTOMJEE", "RVHL", "RVNL", "RVTH",
    "S&SPOWER", "SAATVIKGL", "SAFARI", "SAGARDEEP", "SAGCEM", "SAGILITY",
    "SAHYADRI", "SAIL", "SAILIFE", "SAKAR", "SAKSOFT", "SALONA", "SALSTEEL",
    "SALZERELEC", "SAMBHV", "SAMHI", "SAMMAANCAP", "SAMPANN", "SANATHAN",
    "SANDHAR", "SANDUMA", "SANGAMIND", "SANGHIIND", "SANGHVIMOV", "SANOFI",
    "SANOFICONR", "SANSERA", "SANSTAR", "SAPPHIRE", "SARDAEN", "SAREGAMA",
    "SARLAPOLY", "SASKEN", "SASTASUNDR", "SATIA", "SATIN", "SAURASHCEM", "SBC",
    "SBCL", "SBFC", "SBGLP", "SBICARD", "SBILIFE", "SBIN", "SCHAEFFLER",
    "SCHAND", "SCHNEIDER", "SCI", "SCILAL", "SCODATUBES", "SCPL", "SDBL",
    "SEAMECLTD", "SECMARK", "SELMC", "SEMAC", "SENCO", "SENORES", "SERVOTECH",
    "SESHAPAPER", "SFL", "SGFIN", "SGIL", "SGMART", "SHAHALLOYS", "SHAILY",
    "SHAKTIPUMP", "SHALBY", "SHALPAINTS", "SHANKARA", "SHANTIGEAR", "SHANTIGOLD",
    "SHARDACROP", "SHARDAMOTR", "SHAREINDIA", "SHEMAROO", "SHILCTECH",
    "SHILPAMED", "SHIVALIK", "SHIVAMILLS", "SHIVATEX", "SHK", "SHOPERSTOP",
    "SHRADHA", "SHREDIGCEM", "SHREECEM", "SHREEJISPG", "SHREEPUSHK",
    "SHREERAMA", "SHREYANIND", "SHRINGARMS", "SHRIPISTON", "SHRIRAMFIN",
    "SHRIRAMPPS", "SHYAMMETL", "SICALLOG", "SIEMENS", "SIGIND", "SIGMA",
    "SIGNATURE", "SIGNPOST", "SILGO", "SILINV", "SILVERTUC", "SIMPLEXINF",
    "SINCLAIR", "SINTERCOM", "SIRCA", "SIS", "SIYSIL", "SJS", "SJVN", "SKFINDIA",
    "SKFINDUS", "SKIPPER", "SKMEGGPROD", "SKYGOLD", "SMARTLINK", "SMARTWORKS",
    "SMCGLOBAL", "SMLMAH", "SMLT", "SMSPHARMA", "SNOWMAN", "SOBHA", "SOFTTECH",
    "SOLARA", "SOLARINDS", "SOLARWORLD", "SOLEX", "SOMANYCERA", "SOMATEX",
    "SOMICONVEY", "SONACOMS", "SONAMLTD", "SONATSOFTW", "SOTL", "SOUTHBANK",
    "SOUTHWEST", "SPAL", "SPANDANA", "SPARC", "SPECIALITY", "SPECTRUM",
    "SPENCERS", "SPIC", "SPLIL", "SPLPETRO", "SPMLINFRA", "SPORTKING", "SRD",
    "SREEL", "SRF", "SRHHYPOLTD", "SRM", "SSDL", "SSWL", "STALLION", "STANLEY",
    "STAR", "STARCEMENT", "STARHEALTH", "STARPAPER", "STARTECK", "STCINDIA",
    "STEELCAS", "STEELCITY", "STEL", "STERTOOLS", "STLTECH", "STOVEKRAFT",
    "STUDDS", "STYL", "STYLAMIND", "STYLEBAAZA", "STYRENIX", "SUBROS",
    "SUDARSCHEM", "SUDEEPPHRM", "SUKHJITS", "SULA", "SUMEETINDS", "SUMICHEM",
    "SUMIT", "SUMMITSEC", "SUNCLAY", "SUNDARMFIN", "SUNDRMBRAK", "SUNDRMFAST",
    "SUNDROP", "SUNFLAG", "SUNPHARMA", "SUNTECK", "SUNTV", "SUPERHOUSE",
    "SUPRAJIT", "SUPREME", "SUPREMEIND", "SUPREMEINF", "SUPRIYA", "SURAJEST",
    "SURAJLTD", "SURAKSHA", "SURYALAXMI", "SURYAROSNI", "SURYODAY", "SUTLEJTEX",
    "SUVEN", "SUYOG", "SUZLON", "SWANCORP", "SWARAJENG", "SWELECTES", "SWIGGY",
    "SWSOLAR", "SYMPHONY", "SYNGENE", "SYRMA", "SYSTMTXC", "TAINWALCHM",
    "TAJGVK", "TAKE", "TALBROAUTO", "TANLA", "TARACHAND", "TARAPUR", "TARC",
    "TARIL", "TARMAT", "TARSONS", "TASTYBITE", "TATACAP", "TATACHEM",
    "TATACOMM", "TATACONSUM", "TATAELXSI", "TATAINVEST", "TATAPOWER",
    "TATASTEEL", "TATATECH", "TATVA", "TBOTEK", "TBZ", "TCI", "TCIEXP",
    "TCPLPACK", "TCS", "TDPOWERSYS", "TEAMLEASE", "TECHM", "TECHNOE", "TEGA",
    "TEJASNET", "TEMBO", "TENNIND", "TEXINFRA", "TEXMOPIPES", "TEXRAIL",
    "TFCILTD", "THANGAMAYL", "THEINVEST", "THEJO", "THELEELA", "THEMISMED",
    "THERMAX", "THOMASCOOK", "THOMASCOTT", "THYROCARE", "TI", "TIGERLOGS",
    "TIIL", "TIINDIA", "TIL", "TIMETECHNO", "TIMKEN", "TINNARUBR", "TIPSFILMS",
    "TIPSMUSIC", "TIRUMALCHM", "TIRUPATIFL", "TITAGARH", "TITAN", "TMB",
    "TMCV", "TMPV", "TNPETRO", "TNPL", "TOKYOPLAST", "TOLINS", "TORNTPHARM",
    "TORNTPOWER", "TOTAL", "TOUCHWOOD", "TPLPLASTEH", "TRACXN", "TRANSRAILL",
    "TRANSWORLD", "TRAVELFOOD", "TREJHARA", "TREL", "TRENT", "TRF", "TRIDENT",
    "TRIGYN", "TRITURBINE", "TRIVENI", "TSFINV", "TTKHLTCARE", "TTKPRESTIG",
    "TTML", "TVSELECT", "TVSHLTD", "TVSMOTOR", "TVSSCS", "TVSSRICHAK", "TVTODAY",
    "UBL", "UCAL", "UCOBANK", "UDS", "UFBL", "UFLEX", "UFO", "UGARSUGAR",
    "UGROCAP", "UJJIVANSFB", "ULTRACEMCO", "UMAEXPORTS", "UMIYA-MRO",
    "UNICHEMLAB", "UNIDT", "UNIECOM", "UNIENTER", "UNIMECH", "UNIONBANK",
    "UNIPARTS", "UNITDSPR", "UNITEDPOLY", "UNITEDTEA", "UNIVASTU", "UNIVCABLES",
    "UNIVPHOTO", "UNOMINDA", "UPL", "URAVIDEF", "URBANCO", "USHAMART", "UTIAMC",
    "UTLSOLAR", "UTTAMSUGAR", "V2RETAIL", "VADILALIND", "VAIBHAVGBL",
    "VALIANTLAB", "VALIANTORG", "VARDHACRLC", "VARROC", "VASCONEQ", "VASWANI",
    "VBL", "VEDL", "VEEDOL", "VENKEYS", "VENTIVE", "VENUSPIPES", "VENUSREM",
    "VERANDA", "VERTOZ", "VESUVIUS", "VETO", "VGL", "VGUARD", "VHL", "VHLTD",
    "VIDHIING", "VIDYAWIRES", "VIJAYA", "VIKRAMSOLR", "VIKRAN", "VIMTALABS",
    "VINATIORGA", "VINCOFE", "VINDHYATEL", "VINEETLAB", "VINYLINDIA", "VIPIND",
    "VISAKAIND", "VISASTEEL", "VISHNU", "VLSFINANCE", "VMART", "VMM", "VMSTMT",
    "VOLTAMP", "VOLTAS", "VPRPL", "VRAJ", "VRLLOG", "VSSL", "VSTIND", "VSTL",
    "VSTTILLERS", "VTL", "WAAREEENER", "WAAREEINDO", "WAAREERTL", "WABAG",
    "WAKEFIT", "WALCHANNAG", "WANBURY", "WCIL", "WEALTH", "WEBELSOLAR",
    "WEIZMANIND", "WEL", "WELCORP", "WELENT", "WELINV", "WELSPUNLIV", "WENDT",
    "WESTLIFE", "WEWIN", "WEWORK", "WHEELS", "WHIRLPOOL", "WILLAMAGOR",
    "WINDLAS", "WINDMACHIN", "WIPL", "WIPRO", "WOCKPHARMA", "WONDERLA",
    "WORTHPERI", "WSI", "WSTCSTPAPR", "XCHANGING", "XELPMOC", "XPROINDIA",
    "XTGLOBAL", "YASHO", "YATHARTH", "YATRA", "YESBANK", "YUKEN", "ZAGGLE",
    "ZEEL", "ZENITHEXPO", "ZENSARTECH", "ZENTEC", "ZFCVINDIA", "ZIMLAB",
    "ZODIAC", "ZODIACLOTH", "ZOTA", "ZUARI", "ZUARIIND", "ZYDUSLIFE", "ZYDUSWELL"
]

# ============================================
# STATIC DATA MANAGEMENT
# ============================================

def load_static_data():
    """Load static data from CSV file"""
    if STATIC_DATA_FILE.exists():
        logger.info(f"Loading static data from {STATIC_DATA_FILE}")
        df = pd.read_csv(STATIC_DATA_FILE)
        
        # Check for required columns (now Cap_Class instead of MarketCap)
        required_cols = ['Symbol', 'Sector', 'Industry', 'Cap_Class']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Static data missing columns: {missing_cols}")
            logger.info("Required columns: Symbol, Sector, Industry, Cap_Class")
            return None
        
        # Report missing symbols
        all_symbols = set(RAW_SYMBOLS)
        static_symbols = set(df['Symbol'].values)
        missing_symbols = all_symbols - static_symbols
        
        if missing_symbols:
            logger.warning(f"⚠️ {len(missing_symbols)} symbols missing from static_data.csv:")
            for sym in list(missing_symbols)[:10]:
                logger.warning(f"   - {sym}")
            if len(missing_symbols) > 10:
                logger.warning(f"   ... and {len(missing_symbols) - 10} more")
        
        # Validate Cap_Class values
        valid_caps = ['Large Cap', 'Mid Cap', 'Small Cap']
        invalid_caps = df[~df['Cap_Class'].isin(valid_caps)]
        if not invalid_caps.empty:
            logger.warning(f"⚠️ {len(invalid_caps)} symbols have invalid Cap_Class values")
            logger.warning("Valid values: 'Large Cap', 'Mid Cap', 'Small Cap'")
            for sym in invalid_caps['Symbol'].head(5):
                logger.warning(f"   - {sym}")
        
        return df
    else:
        logger.error(f"❌ static_data.csv not found! Please create it with columns: Symbol, Sector, Industry, Cap_Class")
        return None

# ============================================
# BATCH DOWNLOAD ENGINE
# ============================================

def safe_batch_download(symbols_batch):
    """Download a batch of symbols using yf.download + fallback for missing"""
    
    nse_symbols = [f"{s}.NS" for s in symbols_batch]
    
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(
                nse_symbols,
                period="1y",
                interval="1d",
                progress=False,
                auto_adjust=True,
                repair=True,
                group_by='ticker',
                threads=False
            )
            
            if df is None or df.empty:
                return {}
            
            result = {}
            returned_symbols = set()

            # =========================
            # HANDLE MULTI-INDEX DATA
            # =========================
            if isinstance(df.columns, pd.MultiIndex):
                available = set(df.columns.get_level_values(0))

                for symbol in symbols_batch:
                    nse_symbol = f"{symbol}.NS"

                    if nse_symbol in available:
                        stock_df = df[nse_symbol].copy()

                        if isinstance(stock_df.columns, pd.MultiIndex):
                            stock_df.columns = stock_df.columns.get_level_values(0)

                        stock_df = stock_df.loc[:, ~stock_df.columns.duplicated()]

                        if not stock_df.empty:
                            result[symbol] = stock_df
                            returned_symbols.add(symbol)

            # =========================
            # HANDLE SINGLE SYMBOL CASE
            # =========================
            else:
                if not df.empty:
                    result[symbols_batch[0]] = df
                    returned_symbols.add(symbols_batch[0])

            # =========================
            # 🔥 FALLBACK FOR MISSING
            # =========================
            missing_symbols = set(symbols_batch) - returned_symbols

            if missing_symbols:
                logger.debug(f"Retrying {len(missing_symbols)} missing symbols individually")

                for symbol in missing_symbols:
                    try:
                        single = yf.download(
                            f"{symbol}.NS",
                            period="1y",
                            interval="1d",
                            progress=False,
                            auto_adjust=True,
                            repair=True,
                            threads=False
                        )

                        if single is not None and not single.empty:
                            result[symbol] = single

                    except Exception as e:
                        logger.debug(f"Fallback failed for {symbol}: {e}")

                    time.sleep(0.05)  # small delay to avoid rate limits

            return result

        except Exception as e:
            logger.debug(f"Batch attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    return {}



def download_universe(symbols):
    """Download all symbols in batches with individual fallback"""
    all_data = {}
    total_batches = (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} symbols)")
        
        # Check cache first
        symbols_to_download = []
        for symbol in batch:
            cache_file = CACHE_DIR / f"{symbol}_1y.pkl"
            if cache_file.exists():
                try:
                    cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if cache_age < timedelta(hours=24):
                        all_data[symbol] = pd.read_pickle(cache_file)
                        continue
                except:
                    pass
            symbols_to_download.append(symbol)
        
        if symbols_to_download:
            logger.info(f"Downloading {len(symbols_to_download)} symbols from batch {batch_num}")
            
            # Try batch download
            batch_results = safe_batch_download(symbols_to_download)
            
            # Check which symbols failed batch download
            failed_symbols = [s for s in symbols_to_download if s not in batch_results]
            
            if failed_symbols:
                logger.info(f"Retrying {len(failed_symbols)} symbols individually...")
                # Retry failed symbols individually
                for symbol in failed_symbols:
                    try:
                        ticker = yf.Ticker(f"{symbol}.NS")
                        hist = ticker.history(period="1y", auto_adjust=True, repair=True)
                        
                        if hist is not None and not hist.empty:
                            # Clean NaN values
                            hist = hist[hist['Close'].notna()]
                            if 'Volume' in hist.columns:
                                hist = hist[hist['Volume'] > 0]
                            
                            if len(hist) >= 50:
                                last_close = hist['Close'].iloc[-1]
                                if last_close > 0 and not pd.isna(last_close):
                                    all_data[symbol] = hist
                                    cache_file = CACHE_DIR / f"{symbol}_1y.pkl"
                                    hist.to_pickle(cache_file)
                                    logger.debug(f"Individual download succeeded for {symbol}")
                                else:
                                    logger.debug(f"Invalid last close for {symbol}")
                            else:
                                logger.debug(f"Insufficient data for {symbol}: {len(hist)} rows after cleaning")
                        else:
                            logger.debug(f"No data for {symbol}")
                    except Exception as e:
                        logger.debug(f"Individual download failed for {symbol}: {e}")
            
            # Add successfully downloaded from batch
            for symbol, hist in batch_results.items():
                if hist is not None and symbol not in all_data:
                    all_data[symbol] = hist
                    cache_file = CACHE_DIR / f"{symbol}_1y.pkl"
                    hist.to_pickle(cache_file)
            
            # Small pause between batches
            if i + BATCH_SIZE < len(symbols):
                time.sleep(0.5)
    
    return all_data
# ============================================
# FETCH NIFTY INDEX
# ============================================

def fetch_nifty_index():
    """Fetch Nifty 50 index data for beta calculation"""
    try:
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="1y")
        if hist.empty:
            nifty = yf.Ticker("NSEI")
            hist = nifty.history(period="1y")
        return hist if not hist.empty else None
    except Exception as e:
        logger.debug(f"Could not fetch Nifty index: {e}")
        return None

def calculate_beta(stock_hist, nifty_hist):
    """Calculate beta for a stock"""
    try:
        if nifty_hist is None or len(nifty_hist) < 50:
            return 1.0
        
        stock_returns = stock_hist['Close'].pct_change().dropna()
        nifty_returns = nifty_hist['Close'].pct_change().dropna()
        
        min_len = min(len(stock_returns), len(nifty_returns))
        if min_len < 30:
            return 1.0
        
        stock_returns_aligned = stock_returns.iloc[-min_len:]
        nifty_returns_aligned = nifty_returns.iloc[-min_len:]
        
        covariance = np.cov(stock_returns_aligned, nifty_returns_aligned)[0][1]
        variance = np.var(nifty_returns_aligned)
        
        beta = covariance / variance if variance > 0 else 1.0
        return max(0.5, min(2.0, beta))
    except:
        return 1.0

# ============================================
# PROCESS STOCK DATA
# ============================================

def process_stock_data(symbol, hist, nifty_hist, static_df):
    """Process raw historical data into stock metrics"""
    try:
        if hist is None or hist.empty:
            return None
        
        # CRITICAL FIX: Remove any rows after the last valid Close price
        # This handles Saturday/Sunday placeholder rows with NaN prices
        last_valid_idx = hist['Close'].last_valid_index()
        if last_valid_idx is not None:
            hist = hist.loc[:last_valid_idx]
        
        if len(hist) < 50:
            logger.debug(f"{symbol}: Only {len(hist)} rows after cleaning (need 50)")
            return None
        
        # Now use the last row (which has valid Close)
        current = hist.iloc[-1]
        prev_close = hist.iloc[-2]['Close'] if len(hist) > 1 else current['Close']
        
        # Check for zero price (just in case)
        if current['Close'] <= 0 or pd.isna(current['Close']):
            logger.debug(f"{symbol}: Zero or NaN price after cleaning")
            return None
        
        # Get static data
        static_row = static_df[static_df['Symbol'] == symbol]
        if not static_row.empty:
            sector = static_row.iloc[0]['Sector']
            industry = static_row.iloc[0]['Industry']
            cap_class = static_row.iloc[0]['Cap_Class']
        else:
            logger.debug(f"No static data for {symbol}, skipping")
            return None
        
        # Timeframe indices
        idx_3d = max(0, len(hist) - 4)
        idx_1w = max(0, len(hist) - 6)
        idx_1m = max(0, len(hist) - 21)
        idx_3m = max(0, len(hist) - 63)
        idx_6m = max(0, len(hist) - 126)
        idx_1y = max(0, len(hist) - 252)
        
        # Calculate returns
        change_pct = ((current['Close'] - prev_close) / prev_close) * 100 if prev_close > 0 else 0
        three_day_return = ((current['Close'] - hist.iloc[idx_3d]['Close']) / hist.iloc[idx_3d]['Close']) * 100 if idx_3d < len(hist) and hist.iloc[idx_3d]['Close'] > 0 else 0
        week_return = ((current['Close'] - hist.iloc[idx_1w]['Close']) / hist.iloc[idx_1w]['Close']) * 100 if idx_1w < len(hist) and hist.iloc[idx_1w]['Close'] > 0 else 0
        month_return = ((current['Close'] - hist.iloc[idx_1m]['Close']) / hist.iloc[idx_1m]['Close']) * 100 if idx_1m < len(hist) and hist.iloc[idx_1m]['Close'] > 0 else 0
        three_month_return = ((current['Close'] - hist.iloc[idx_3m]['Close']) / hist.iloc[idx_3m]['Close']) * 100 if idx_3m < len(hist) and hist.iloc[idx_3m]['Close'] > 0 else 0
        six_month_return = ((current['Close'] - hist.iloc[idx_6m]['Close']) / hist.iloc[idx_6m]['Close']) * 100 if idx_6m < len(hist) and hist.iloc[idx_6m]['Close'] > 0 else 0
        one_year_return = ((current['Close'] - hist.iloc[idx_1y]['Close']) / hist.iloc[idx_1y]['Close']) * 100 if idx_1y < len(hist) and hist.iloc[idx_1y]['Close'] > 0 else 0
        
        # Moving averages
        sma_21 = hist['Close'].rolling(window=21).mean().iloc[-1]
        sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        
        # Safeguard for NaN moving averages
        if pd.isna(sma_21) or sma_21 <= 0:
            sma_21 = current['Close']
        if pd.isna(sma_50) or sma_50 <= 0:
            sma_50 = current['Close']
        if pd.isna(sma_200) or sma_200 <= 0:
            sma_200 = current['Close']
        
        # 52-week high/low
        high_52w = hist['High'].rolling(window=252).max().iloc[-1]
        low_52w = hist['Low'].rolling(window=252).min().iloc[-1]
        
        if pd.isna(high_52w) or high_52w == 0:
            high_52w = hist['High'].max()
        if pd.isna(low_52w) or low_52w == 0:
            low_52w = hist['Low'].min()
        
        pct_from_high = ((current['Close'] - high_52w) / high_52w) * 100 if high_52w > 0 else 0
        pct_from_low = ((current['Close'] - low_52w) / low_52w) * 100 if low_52w > 0 else 0
        
        # Volume
        current_volume = current['Volume'] if 'Volume' in current else 0
        avg_volume_20 = hist['Volume'].tail(20).mean() if 'Volume' in hist.columns else 1
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
        
        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1] if not pd.isna(rs.iloc[-1]) else 50
        
        # MACD
        exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_histogram = (macd - macd_signal).iloc[-1] if len(macd) > 0 else 0
        
        # Bollinger Bands
        bb_mid = hist['Close'].rolling(window=20).mean().iloc[-1]
        bb_std = hist['Close'].rolling(window=20).std().iloc[-1]
        bb_upper = bb_mid + (bb_std * 2)
        bb_lower = bb_mid - (bb_std * 2)
        bb_position = ((current['Close'] - bb_lower) / (bb_upper - bb_lower)) * 100 if bb_upper != bb_lower else 50
        
        # Beta and Volatility
        beta = calculate_beta(hist, nifty_hist)
        volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100
        
        return {
            'Symbol': symbol,
            'Sector': sector,
            'Industry': industry,
            'Market_Cap': 0,
            'Cap_Class': cap_class,
            'CMP': round(current['Close'], 2),
            'OPEN': round(current['Open'] if 'Open' in current else current['Close'], 2),
            'HIGH': round(current['High'] if 'High' in current else current['Close'], 2),
            'LOW': round(current['Low'] if 'Low' in current else current['Close'], 2),
            'CHANGEPCT': round(change_pct, 2),
            'YEST CLOSE': round(prev_close, 2),
            'Close 1 Week': round(hist.iloc[idx_1w]['Close'], 2),
            'Close 3 days ago': round(hist.iloc[idx_3d]['Close'], 2),
            '3 day %': round(three_day_return, 2),
            '200 DMA': round(sma_200, 2),
            '% from 200 DMA': round(((current['Close'] - sma_200) / sma_200) * 100 if sma_200 > 0 else 0, 2),
            '50 day SMA': round(sma_50, 2),
            '21 Day SMA': round(sma_21, 2),
            '52w high': round(high_52w, 2),
            '52w low': round(low_52w, 2),
            '1 week % change': round(week_return, 2),
            '1m % change': round(month_return, 2),
            '3m % change': round(three_month_return, 2),
            '6m % change': round(six_month_return, 2),
            '1 year return': round(one_year_return, 2),
            '% from 52 low': round(pct_from_low, 2),
            '% from 52w high': round(pct_from_high, 2),
            'Volume Ratio': round(volume_ratio, 2),
            'RSI': round(rsi, 2),
            'MACD Hist': round(macd_histogram, 4),
            'BB Position': round(bb_position, 2),
            'Beta': round(beta, 2),
            'Volatility': round(volatility, 2)
        }
    except Exception as e:
        logger.debug(f"Error processing {symbol}: {e}")
        return None
# ============================================
# SCORING FUNCTIONS
# ============================================

def calculate_trend_score(row):
    try:
        ma_short = 1.8 if row['CMP'] > row['21 Day SMA'] else 0
        ma_mid = 2.4 if row['CMP'] > row['50 day SMA'] else 0
        ma_long = 2.2 if row['CMP'] > row['200 DMA'] else 0
        
        ret_6m = row['6m % change']
        if ret_6m > 30: ret_6m_score = 1.9
        elif ret_6m > 15: ret_6m_score = 1.3
        elif ret_6m > 0: ret_6m_score = 0.7
        elif ret_6m < -15: ret_6m_score = -1.3
        else: ret_6m_score = -0.7
        
        ret_1y = row['1 year return']
        if ret_1y > 45: ret_1y_score = 2.3
        elif ret_1y > 30: ret_1y_score = 1.6
        elif ret_1y > 15: ret_1y_score = 1
        elif ret_1y > 0: ret_1y_score = 0.5
        elif ret_1y < -30: ret_1y_score = -2
        else: ret_1y_score = -1.2
        
        near_high = 1 if row['% from 52w high'] >= -10 else 0
        low_penalty = -1 if row['% from 52 low'] <= 12 else 0
        breakout = 0.9 if row['CMP'] > row['200 DMA'] * 0.97 else 0
        
        raw = ma_short + ma_mid + ma_long + ret_6m_score + ret_1y_score + near_high + low_penalty + breakout
        return round(min(10, max(0, raw * 10 / 13.2)), 2)
    except:
        return 0

def calculate_momentum_score(row):
    try:
        s2 = row['1m % change']
        s2_score = 2.8 if s2 > 16 else (2 if s2 > 9 else (1.2 if s2 > 3.5 else 0))
        
        u2 = row['3m % change']
        u2_score = 2.4 if u2 > 14 else (1.7 if u2 > 7 else (0.9 if u2 > 2.5 else 0))
        
        l2 = row['1 week % change']
        l2_score = 2 if l2 > 1.8 else (1.1 if l2 > 0.9 else 0)
        
        h2 = row['3 day %']
        h2_score = 1.7 if h2 > 1.3 else (0.9 if h2 > 0.6 else 0)
        
        t2 = row['6m % change']
        t2_score = 2.1 if t2 > 24 else (1.4 if t2 > 12 else 0)
        
        condition1 = 1.1 if row['1m % change'] > row['6m % change'] * 0.65 else 0
        condition2 = 0.9 if (row['CMP'] > row['50 day SMA'] and row['1m % change'] > 0) else 0
        
        raw = (s2_score + u2_score + l2_score + h2_score + t2_score + condition1 + condition2) * 10 / 13
        return round(min(10, max(0, raw)), 2)
    except:
        return 0

def calculate_early_detection_score(row):
    try:
        h2 = row['3 day %']
        h2_score = 4.8 if h2 > 11 else (3.8 if h2 > 7 else (2.6 if h2 > 3.5 else (1.4 if h2 > 1.2 else 0)))
        
        l2 = row['1 week % change']
        l2_score = 4.8 if l2 > 11 else (3.8 if l2 > 7 else (2.6 if l2 > 3.5 else (1.4 if l2 > 1.2 else 0)))
        
        s2 = row['1m % change']
        s2_score = 4.2 if s2 > 20 else (3.2 if s2 > 11 else (1.8 if s2 > 4.5 else 0))
        
        t2 = row['6m % change']
        t2_score = 2.2 if t2 > 28 else (1.5 if t2 > 14 else 0)
        
        condition1 = 1.3 if row['1m % change'] > row['6m % change'] * 0.75 else 0
        condition2 = 1.8 if (row['% from 52w high'] > -8 and row['3 day %'] > 1.8) else 0
        condition3 = 1.6 if (row['% from 52 low'] < 20 and row['1 week % change'] > 2.8) else 0
        
        raw = (h2_score + l2_score + s2_score + t2_score + condition1 + condition2 + condition3) * 10 / 19.5
        return round(min(10, max(0, raw)), 2)
    except:
        return 0

def calculate_composite_score(row, weights):
    return round(
        row['Early Detection Score'] * weights['early_detection_weight'] +
        row['Momentum Score'] * weights['momentum_weight'] +
        row['Trend Score'] * weights['trend_weight'], 2
    )

def calculate_trend_status(row):
    try:
        if row['CMP'] > row['200 DMA'] and row['CMP'] > row['50 day SMA'] and row['CMP'] > row['21 Day SMA']:
            if row['Momentum Score'] > 7: return "Strong Uptrend"
            elif row['Momentum Score'] > 4: return "Uptrend"
            else: return "Weak Uptrend"
        elif row['CMP'] < row['200 DMA'] and row['CMP'] < row['50 day SMA']:
            if row['% from 52 low'] < 15: return "Oversold"
            else: return "Downtrend"
        elif row['CMP'] > row['200 DMA'] and row['CMP'] < row['50 day SMA']: 
            return "Consolidation"
        elif row['CMP'] < row['200 DMA'] and row['CMP'] > row['50 day SMA']: 
            return "Recovery"
        else: 
            return "Neutral"
    except:
        return "Neutral"

def calculate_action(trend_status, trend_score, momentum_score, early_score):
    try:
        if trend_status in ["Strong Uptrend", "Uptrend"] and trend_score > 6 and momentum_score > 5: 
            return "STRONG BUY"
        elif trend_status in ["Strong Uptrend", "Uptrend"] and trend_score > 4: 
            return "BUY"
        elif trend_status == "Recovery" and early_score > 6: 
            return "BUY"
        elif trend_status == "Oversold" and momentum_score > 5: 
            return "ACCUMULATE"
        elif trend_status == "Consolidation" and early_score > 7: 
            return "ACCUMULATE"
        elif trend_status == "Downtrend": 
            return "AVOID"
        else: 
            return "HOLD"
    except:
        return "HOLD"

def calculate_tier(trend_status, composite_score):
    try:
        if trend_status in ["Strong Uptrend", "Uptrend"] and composite_score > 7: 
            return "TIER 1"
        elif trend_status in ["Strong Uptrend", "Uptrend", "Recovery"] and composite_score > 5: 
            return "TIER 2"
        elif composite_score > 3: 
            return "TIER 3"
        else: 
            return "TIER 4"
    except:
        return "TIER 4"

# ============================================
# ANALYSIS FUNCTIONS
# ============================================

def calculate_comparative_metrics(df):
    sector_avg_1w = df.groupby('Sector')['1 week % change'].transform('mean')
    sector_avg_1m = df.groupby('Sector')['1m % change'].transform('mean')
    
    df['vs_Sector_1w'] = round(df['1 week % change'] - sector_avg_1w, 2)
    df['vs_Sector_1m'] = round(df['1m % change'] - sector_avg_1m, 2)
    
    industry_avg_1w = df.groupby('Industry')['1 week % change'].transform('mean')
    industry_avg_1m = df.groupby('Industry')['1m % change'].transform('mean')
    
    df['vs_Industry_1w'] = round(df['1 week % change'] - industry_avg_1w, 2)
    df['vs_Industry_1m'] = round(df['1m % change'] - industry_avg_1m, 2)
    
    df['Rank_in_Sector'] = df.groupby('Sector')['1 week % change'].rank(ascending=False).fillna(0).astype(int)
    df['Rank_in_Industry'] = df.groupby('Industry')['1 week % change'].rank(ascending=False).fillna(0).astype(int)
    
    return df

def scan_breakouts(df):
    df['Breakout_Type'] = 'None'
    
    volume_breakout = (df['Volume Ratio'] > 1.5) & (df['CHANGEPCT'] > 2)
    df.loc[volume_breakout, 'Breakout_Type'] = 'Volume Breakout'
    
    ma_crossover = (df['CMP'] > df['21 Day SMA']) & (df['21 Day SMA'] > df['50 day SMA'])
    df.loc[ma_crossover & (df['Breakout_Type'] == 'None'), 'Breakout_Type'] = 'MA Crossover'
    
    high_breakout = df['% from 52w high'] > -2
    df.loc[high_breakout & (df['Breakout_Type'] == 'None'), 'Breakout_Type'] = '52W High Alert'
    
    support_bounce = (abs(df['% from 200 DMA']) < 2) & (df['Volume Ratio'] > 1.2) & (df['CHANGEPCT'] > 0)
    df.loc[support_bounce & (df['Breakout_Type'] == 'None'), 'Breakout_Type'] = 'Support Bounce'
    
    rsi_breakout = (df['RSI'] > 70) & (df['Volume Ratio'] > 1.3)
    df.loc[rsi_breakout & (df['Breakout_Type'] == 'None'), 'Breakout_Type'] = 'RSI Overbought'
    
    return df

def create_sector_industry_drilldown(df):
    drilldown = []
    
    for sector in df['Sector'].unique():
        sector_df = df[df['Sector'] == sector]
        sector_avg_1w = sector_df['1 week % change'].mean()
        
        for industry in sector_df['Industry'].unique():
            industry_df = sector_df[sector_df['Industry'] == industry]
            if len(industry_df) > 0:
                industry_avg_1w = industry_df['1 week % change'].mean()
                industry_count = len(industry_df)
                
                drilldown.append({
                    'Sector': sector,
                    'Industry': industry,
                    'Industry_Count': industry_count,
                    'Industry_1W_Return': round(industry_avg_1w, 2),
                    'Industry_vs_Sector': round(industry_avg_1w - sector_avg_1w, 2),
                    'Top_Stock': industry_df.nlargest(1, '1 week % change')['Symbol'].iloc[0] if len(industry_df) > 0 else 'N/A',
                    'Top_Stock_Return': round(industry_df['1 week % change'].max(), 2) if len(industry_df) > 0 else 0
                })
    
    return pd.DataFrame(drilldown).sort_values(['Sector', 'Industry_1W_Return'], ascending=[True, False])

def calculate_sector_rotation_timeline(sector_metrics):
    timeline = sector_metrics.copy()
    
    timeline['Rank_1W'] = timeline['1W_avg'].rank(ascending=False).astype(int)
    timeline['Rank_1M'] = timeline['1M_avg'].rank(ascending=False).astype(int)
    timeline['Rank_3M'] = timeline['3M_avg'].rank(ascending=False).astype(int)
    
    timeline['Rank_Change_1W_vs_1M'] = timeline['Rank_1M'] - timeline['Rank_1W']
    timeline['Rank_Change_1M_vs_3M'] = timeline['Rank_3M'] - timeline['Rank_1M']
    
    def get_direction(change):
        if change > 0: return "↑ Improving"
        elif change < 0: return "↓ Declining"
        else: return "→ Stable"
    
    timeline['Direction_1W_vs_1M'] = timeline['Rank_Change_1W_vs_1M'].apply(get_direction)
    timeline['Direction_1M_vs_3M'] = timeline['Rank_Change_1M_vs_3M'].apply(get_direction)
    
    return timeline

def generate_summary(df, sector_metrics, breakout_df):
    summary = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'recommendations': []
    }
    
    if len(sector_metrics) > 0:
        top_sector = sector_metrics.index[0]
        summary['recommendations'].append(f"PRIMARY FOCUS: {top_sector} sector with {sector_metrics.iloc[0]['1W_avg']:.2f}% weekly return")
    
    emerging = sector_metrics[sector_metrics['Rotation_Phase'] == 'Emerging']
    if not emerging.empty:
        summary['recommendations'].append(f"EMERGING OPPORTUNITY: {', '.join(emerging.index[:2])} sectors showing early momentum")
    
    volume_breakouts = breakout_df[breakout_df['Breakout_Type'] == 'Volume Breakout']
    if not volume_breakouts.empty:
        stocks = [f"{row['Symbol']} (+{row['CHANGEPCT']:.1f}%)" for _, row in volume_breakouts.head(3).iterrows()]
        summary['recommendations'].append(f"VOLUME BREAKOUTS: {', '.join(stocks)}")
    
    high_breakouts = breakout_df[breakout_df['Breakout_Type'] == '52W High Alert']
    if not high_breakouts.empty:
        stocks = [f"{row['Symbol']}" for _, row in high_breakouts.head(3).iterrows()]
        summary['recommendations'].append(f"52W HIGH ALERTS: {', '.join(stocks)}")
    
    early_boomers = df[df['Early Detection Score'] > 7].nlargest(3, 'Early Detection Score')
    if not early_boomers.empty:
        stocks = [f"{row['Symbol']} ({row['Sector']})" for _, row in early_boomers.iterrows()]
        summary['recommendations'].append(f"EARLY BOOMERS: {', '.join(stocks)}")
    
    return summary

# ============================================
# EXCEL OUTPUT - COMPLETE WITH ALL 10 SHEETS
# ============================================

def create_excel_output(df, sector_metrics, industry_metrics, cap_metrics, drilldown_df, breakout_df, 
                        rotation_timeline, summary, data_quality_report, filename="stock_market_analysis.xlsx"):
    """Create comprehensive Excel output with all 10 sheets"""
    
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Format definitions
        green_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        red_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        dark_green_format = workbook.add_format({'bg_color': '#006100', 'font_color': '#FFFFFF', 'bold': True})
        dark_red_format = workbook.add_format({'bg_color': '#FF0000', 'font_color': '#FFFFFF'})
        header_format = workbook.add_format({'bold': True, 'bg_color': '#4472C4', 'color': 'white', 'font_size': 12})
        
        # ============ SHEET 1: Main Analysis ============
        main_cols = [
            'Symbol', 'Cap_Class', 'Sector', 'Industry', 'CMP', 'CHANGEPCT', '3 day %', '1 week % change', 
            '1m % change', '3m % change', '6m % change', '1 year return', 
            '52w high', '52w low', '% from 52w high', '% from 52 low',
            '200 DMA', '50 day SMA', '21 Day SMA', '% from 200 DMA',
            'Trend Score', 'Momentum Score', 'Early Detection Score', 'Composite Score', 
            'Action', 'Tier', 'Volume Ratio', 'RSI', 'Beta', 'Volatility',
            'Breakout_Type', 'vs_Sector_1w', 'Rank_in_Sector'
        ]
        
        available_cols = [col for col in main_cols if col in df.columns]
        df_main = df[available_cols].copy().sort_values('Symbol').reset_index(drop=True)
        df_main.to_excel(writer, sheet_name='1. Main Analysis', index=False)
        
        worksheet = writer.sheets['1. Main Analysis']
        
        # 3-color scale for returns
        return_cols = ['CHANGEPCT', '3 day %', '1 week % change', '1m % change', '3m % change', '6m % change', '1 year return']
        for col_name in return_cols:
            if col_name in df_main.columns:
                col_idx = df_main.columns.get_loc(col_name)
                worksheet.conditional_format(1, col_idx, len(df_main), col_idx, {
                    'type': '3_color_scale',
                    'min_color': '#F8696B',
                    'mid_color': '#FFEB84',
                    'max_color': '#63BE7B'
                })
        
        # Score formatting
        score_cols = ['Trend Score', 'Momentum Score', 'Early Detection Score', 'Composite Score']
        for col_name in score_cols:
            if col_name in df_main.columns:
                col_idx = df_main.columns.get_loc(col_name)
                worksheet.conditional_format(1, col_idx, len(df_main), col_idx, {
                    'type': 'cell',
                    'criteria': '>=',
                    'value': 7,
                    'format': green_format
                })
                worksheet.conditional_format(1, col_idx, len(df_main), col_idx, {
                    'type': 'cell',
                    'criteria': '<',
                    'value': 4,
                    'format': red_format
                })
        
        # Action coloring
        if 'Action' in df_main.columns:
            col_idx = df_main.columns.get_loc('Action')
            worksheet.conditional_format(1, col_idx, len(df_main), col_idx, {
                'type': 'text',
                'criteria': 'containing',
                'value': 'STRONG BUY',
                'format': dark_green_format
            })
            worksheet.conditional_format(1, col_idx, len(df_main), col_idx, {
                'type': 'text',
                'criteria': 'containing',
                'value': 'AVOID',
                'format': dark_red_format
            })
        
        # ============ SHEET 2: Sector Analysis ============
        sector_metrics.to_excel(writer, sheet_name='2. Sector Analysis')
        
        # ============ SHEET 3: Industry Analysis ============
        industry_metrics.to_excel(writer, sheet_name='3. Industry Analysis')
        
        # ============ SHEET 4: Sector-Industry Drilldown ============
        drilldown_df.to_excel(writer, sheet_name='4. Sector-Industry Drilldown', index=False)
        
        # ============ SHEET 5: Breakout Scanner ============
        breakout_df.to_excel(writer, sheet_name='5. Breakout Scanner', index=False)
        
        # ============ SHEET 6: Sector Rotation Timeline ============
        rotation_timeline.to_excel(writer, sheet_name='6. Sector Rotation Timeline')
        
        # ============ SHEET 7: Interactive Dashboard ============
        dashboard = workbook.add_worksheet('7. Interactive Dashboard')
        
        # Create helper sheet for filtering
        helper = workbook.add_worksheet('_FilterData')
        helper.hide()
        
        filter_cols = ['Symbol', 'Cap_Class', 'Sector', 'Industry', 'CHANGEPCT', '3 day %', '1 week % change', 
                       '1m % change', '3m % change', '6m % change', '1 year return']
        filter_data = df[filter_cols].copy()
        
        for col_idx, col_name in enumerate(filter_data.columns):
            helper.write(0, col_idx, col_name)
        for row_idx, row in enumerate(filter_data.values, 1):
            for col_idx, value in enumerate(row):
                helper.write(row_idx, col_idx, value)
        
        # Create named ranges for dropdowns
        caps = ['All', 'Large Cap', 'Mid Cap', 'Small Cap']
        sectors = sorted(df['Sector'].unique())
        industries = sorted(df['Industry'].unique())
        timeframes = ['1D (Today)', '3D', '1W', '1M', '3M', '6M', '1Y']
        
        for i, cap in enumerate(caps):
            helper.write(100 + i, 0, cap)
        workbook.define_name('Caps', f'=_FilterData!$A$101:$A${100 + len(caps)}')
        
        for i, sector in enumerate(sectors):
            helper.write(100 + i, 1, sector)
        workbook.define_name('Sectors', f'=_FilterData!$B$101:$B${100 + len(sectors)}')
        
        for i, industry in enumerate(industries):
            helper.write(100 + i, 2, industry)
        workbook.define_name('Industries', f'=_FilterData!$C$101:$C${100 + len(industries)}')
        
        for i, tf in enumerate(timeframes):
            helper.write(100 + i, 3, tf)
        workbook.define_name('Timeframes', f'=_FilterData!$D$101:$D${100 + len(timeframes)}')
        
        # Dashboard layout
        dashboard.write(0, 0, 'INTERACTIVE FILTER DASHBOARD', header_format)
        
        dashboard.write(2, 0, 'FILTERS', workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'}))
        
        dashboard.write(4, 0, 'Select Market Cap:', workbook.add_format({'bold': True}))
        dashboard.write(4, 1, 'All')
        dashboard.data_validation('B5', {'validate': 'list', 'source': '=Caps'})
        dashboard.write(5, 1, 'All')
        
        dashboard.write(7, 0, 'Select Sector:', workbook.add_format({'bold': True}))
        dashboard.write(7, 1, '')
        dashboard.data_validation('B8', {'validate': 'list', 'source': '=Sectors'})
        
        dashboard.write(9, 0, 'Select Industry:', workbook.add_format({'bold': True}))
        dashboard.write(9, 1, '')
        dashboard.data_validation('B10', {'validate': 'list', 'source': '=Industries'})
        
        dashboard.write(11, 0, 'Select Timeframe:', workbook.add_format({'bold': True}))
        dashboard.write(11, 1, '')
        dashboard.data_validation('B12', {'validate': 'list', 'source': '=Timeframes'})
        dashboard.write(12, 1, '1D (Today)')
        
        dashboard.write(14, 0, 'SELECTION SUMMARY', workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'}))
        dashboard.write(15, 0, 'Selected Cap:')
        dashboard.write(16, 0, 'Selected Sector:')
        dashboard.write(17, 0, 'Selected Industry:')
        dashboard.write(18, 0, 'Selected Timeframe:')
        
        dashboard.write_formula(15, 1, '=IF(B5="", "All Caps", B5)')
        dashboard.write_formula(16, 1, '=IF(B8="", "All Sectors", B8)')
        dashboard.write_formula(17, 1, '=IF(B10="", "All Industries", B10)')
        dashboard.write_formula(18, 1, '=IF(B12="", "1D (Today)", B12)')
        
        dashboard.write(20, 0, 'AVERAGE RETURN:', workbook.add_format({'bold': True, 'font_size': 12}))
        avg_formula = '=IF(AND(B5="", B8="", B10=""), "", AVERAGE(FILTER(INDEX(_FilterData!$A$2:$K$1000, 0, MATCH(B12, {"1D (Today)","3D","1W","1M","3M","6M","1Y"}, 0)+4), IF(OR(B5="All", B5=""), TRUE, _FilterData!$B$2:$B$1000=B5) * IF(OR(B8="", B8="All Sectors"), TRUE, _FilterData!$C$2:$C$1000=B8) * IF(OR(B10="", B10="All Industries"), TRUE, _FilterData!$D$2:$D$1000=B10))))'
        dashboard.write_formula(20, 1, avg_formula)
        
        dashboard.write(22, 0, 'TOP 10 GAINERS', workbook.add_format({'bold': True, 'bg_color': '#C6EFCE'}))
        dashboard.write(23, 0, 'Rank')
        dashboard.write(23, 1, 'Stock')
        dashboard.write(23, 2, 'Return (%)')
        dashboard.write(22, 4, 'TOP 10 LOSERS', workbook.add_format({'bold': True, 'bg_color': '#FFC7CE'}))
        dashboard.write(23, 4, 'Rank')
        dashboard.write(23, 5, 'Stock')
        dashboard.write(23, 6, 'Return (%)')
        
        for i in range(10):
            rank = i + 1
            dashboard.write(24 + i, 0, rank)
            dashboard.write(24 + i, 4, rank)
            
            base_filter = f'IF(AND(OR(B5="All", B5=""), OR(B8="", B8="All Sectors"), OR(B10="", B10="All Industries")), FILTER(HSTACK(_FilterData!$A$2:$A$1000, INDEX(_FilterData!$A$2:$K$1000, 0, MATCH(B12, {{"1D (Today)","3D","1W","1M","3M","6M","1Y"}}, 0)+4)), IF(OR(B5="All", B5=""), TRUE, _FilterData!$B$2:$B$1000=B5) * IF(OR(B8="", B8="All Sectors"), TRUE, _FilterData!$C$2:$C$1000=B8) * IF(OR(B10="", B10="All Industries"), TRUE, _FilterData!$D$2:$D$1000=B10)), FILTER(HSTACK(_FilterData!$A$2:$A$1000, INDEX(_FilterData!$A$2:$K$1000, 0, MATCH(B12, {{"1D (Today)","3D","1W","1M","3M","6M","1Y"}}, 0)+4)), IF(B5="All", TRUE, _FilterData!$B$2:$B$1000=B5) * IF(B8="", TRUE, _FilterData!$C$2:$C$1000=B8) * IF(B10="", TRUE, _FilterData!$D$2:$D$1000=B10)))'
            
            gainer_stock = f'=IF(AND(B5="All", B8="", B10=""), "", IFERROR(INDEX(SORT({base_filter}, 2, -1), {rank}, 1), ""))'
            gainer_return = f'=IF(AND(B5="All", B8="", B10=""), "", IFERROR(INDEX(SORT({base_filter}, 2, -1), {rank}, 2), ""))'
            loser_stock = f'=IF(AND(B5="All", B8="", B10=""), "", IFERROR(INDEX(SORT({base_filter}, 2, 1), {rank}, 1), ""))'
            loser_return = f'=IF(AND(B5="All", B8="", B10=""), "", IFERROR(INDEX(SORT({base_filter}, 2, 1), {rank}, 2), ""))'
            
            dashboard.write_formula(24 + i, 1, gainer_stock)
            dashboard.write_formula(24 + i, 2, gainer_return)
            dashboard.write_formula(24 + i, 5, loser_stock)
            dashboard.write_formula(24 + i, 6, loser_return)
        
        dashboard.conditional_format('C25:C34', {'type': '3_color_scale', 'min_color': '#F8696B', 'mid_color': '#FFEB84', 'max_color': '#63BE7B'})
        dashboard.conditional_format('G25:G34', {'type': '3_color_scale', 'min_color': '#F8696B', 'mid_color': '#FFEB84', 'max_color': '#63BE7B'})
        
        dashboard.write(36, 0, 'INSTRUCTIONS:', workbook.add_format({'bold': True}))
        dashboard.write(37, 0, '1. Select Market Cap at B5 (default: All)')
        dashboard.write(38, 0, '2. Select Sector at B8 (leave blank for all)')
        dashboard.write(39, 0, '3. Select Industry at B10 (leave blank for all)')
        dashboard.write(40, 0, '4. Select Timeframe at B12 (default: 1D)')
        
        # ============ SHEET 8: Score Configuration ============
        config_sheet = workbook.add_worksheet('8. Score Configuration')
        
        config_sheet.write(0, 0, 'CUSTOM SCORE WEIGHTS', workbook.add_format({'bold': True, 'font_size': 14}))
        config_sheet.write(2, 0, 'Select Mode:', workbook.add_format({'bold': True}))
        config_sheet.write(3, 0, '')
        config_sheet.data_validation('A4', {'validate': 'list', 'source': ['short_term', 'balanced', 'long_term']})
        
        config_sheet.write(5, 0, 'CURRENT ACTIVE CONFIGURATION:', workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'}))
        config_sheet.write(6, 0, 'Mode:')
        config_sheet.write(7, 0, 'Early Detection Weight:')
        config_sheet.write(8, 0, 'Momentum Weight:')
        config_sheet.write(9, 0, 'Trend Weight:')
        
        config_sheet.write(6, 1, '=A4')
        config_sheet.write_formula(7, 1, '=VLOOKUP(A4, {"short_term",0.5;"balanced",0.3;"long_term",0.2}, 2, 0)')
        config_sheet.write_formula(8, 1, '=VLOOKUP(A4, {"short_term",0.3;"balanced",0.35;"long_term",0.3}, 2, 0)')
        config_sheet.write_formula(9, 1, '=VLOOKUP(A4, {"short_term",0.2;"balanced",0.35;"long_term",0.5}, 2, 0)')
        
        # ============ SHEET 9: Summary ============
        summary_sheet = workbook.add_worksheet('9. Summary')
        
        summary_sheet.write(0, 0, 'STOCK MARKET ANALYSIS SUMMARY', header_format)
        summary_sheet.write(1, 0, f'Analysis Date: {summary["date"]}')
        summary_sheet.write(2, 0, f'Total Stocks Analyzed: {len(df)}')
        
        # Market Cap Distribution & Performance
        row = 4
        summary_sheet.write(row, 0, 'MARKET CAP SUMMARY', workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'}))
        summary_sheet.write(row+1, 0, 'Cap Class')
        summary_sheet.write(row+1, 1, 'Count')
        summary_sheet.write(row+1, 2, '1D')
        summary_sheet.write(row+1, 3, '1W')
        summary_sheet.write(row+1, 4, '1M')
        summary_sheet.write(row+1, 5, '3M')
        
        for i, cap in enumerate(['Large Cap', 'Mid Cap', 'Small Cap']):
            cap_df = df[df['Cap_Class'] == cap]
            if len(cap_df) > 0:
                summary_sheet.write(row+2+i, 0, cap)
                summary_sheet.write(row+2+i, 1, len(cap_df))
                summary_sheet.write(row+2+i, 2, f"{cap_df['CHANGEPCT'].mean():.2f}%")
                summary_sheet.write(row+2+i, 3, f"{cap_df['1 week % change'].mean():.2f}%")
                summary_sheet.write(row+2+i, 4, f"{cap_df['1m % change'].mean():.2f}%")
                summary_sheet.write(row+2+i, 5, f"{cap_df['3m % change'].mean():.2f}%")
        
        # Top Sectors with all timeframes
        row += 6
        summary_sheet.write(row, 0, 'TOP SECTORS', workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'}))
        summary_sheet.write(row+1, 0, 'Sector')
        summary_sheet.write(row+1, 1, '1D')
        summary_sheet.write(row+1, 2, '3D')
        summary_sheet.write(row+1, 3, '1W')
        summary_sheet.write(row+1, 4, '1M')
        summary_sheet.write(row+1, 5, '3M')
        summary_sheet.write(row+1, 6, 'Momentum')
        summary_sheet.write(row+1, 7, 'Phase')
        
        for i, (sector, row_data) in enumerate(sector_metrics.head(10).iterrows()):
            summary_sheet.write(row+2+i, 0, sector)
            summary_sheet.write(row+2+i, 1, f"{row_data['1D_avg']:.2f}%")
            summary_sheet.write(row+2+i, 2, f"{row_data['3D_avg']:.2f}%")
            summary_sheet.write(row+2+i, 3, f"{row_data['1W_avg']:.2f}%")
            summary_sheet.write(row+2+i, 4, f"{row_data['1M_avg']:.2f}%")
            summary_sheet.write(row+2+i, 5, f"{row_data['3M_avg']:.2f}%")
            summary_sheet.write(row+2+i, 6, f"{row_data['Momentum_Shift']:.2f}%")
            summary_sheet.write(row+2+i, 7, row_data['Rotation_Phase'])
        
        # Top Industries with all timeframes
        row += len(sector_metrics.head(10)) + 5
        summary_sheet.write(row, 0, 'TOP INDUSTRIES', workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'}))
        summary_sheet.write(row+1, 0, 'Industry')
        summary_sheet.write(row+1, 1, 'Sector')
        summary_sheet.write(row+1, 2, '1D')
        summary_sheet.write(row+1, 3, '3D')
        summary_sheet.write(row+1, 4, '1W')
        summary_sheet.write(row+1, 5, '1M')
        summary_sheet.write(row+1, 6, '3M')
        
        top_industries = df.groupby(['Industry', 'Sector']).agg({
            'CHANGEPCT': 'mean',
            '3 day %': 'mean',
            '1 week % change': 'mean',
            '1m % change': 'mean',
            '3m % change': 'mean'
        }).round(2).reset_index().sort_values('1 week % change', ascending=False).head(15)
        
        for i, row_data in enumerate(top_industries.iterrows()):
            summary_sheet.write(row+2+i, 0, row_data[1]['Industry'])
            summary_sheet.write(row+2+i, 1, row_data[1]['Sector'])
            summary_sheet.write(row+2+i, 2, f"{row_data[1]['CHANGEPCT']:.2f}%")
            summary_sheet.write(row+2+i, 3, f"{row_data[1]['3 day %']:.2f}%")
            summary_sheet.write(row+2+i, 4, f"{row_data[1]['1 week % change']:.2f}%")
            summary_sheet.write(row+2+i, 5, f"{row_data[1]['1m % change']:.2f}%")
            summary_sheet.write(row+2+i, 6, f"{row_data[1]['3m % change']:.2f}%")
        
        # Breakout Summary
        row += 20
        summary_sheet.write(row, 0, 'BREAKOUT SUMMARY', workbook.add_format({'bold': True, 'bg_color': '#FFC000'}))
        breakout_summary = breakout_df.groupby('Breakout_Type').size().reset_index(name='Count')
        for i, row_data in enumerate(breakout_summary.iterrows()):
            summary_sheet.write(row+1+i, 0, row_data[1]['Breakout_Type'])
            summary_sheet.write(row+1+i, 1, row_data[1]['Count'])
        
        # Recommendations
        row += len(breakout_summary) + 5
        summary_sheet.write(row, 0, 'RECOMMENDATIONS', workbook.add_format({'bold': True, 'bg_color': '#FFC000'}))
        for i, rec in enumerate(summary['recommendations']):
            summary_sheet.write(row+1+i, 0, rec)
        
        # ============ SHEET 10: Data Quality Report ============
        quality_sheet = workbook.add_worksheet('10. Data Quality Report')
        
        quality_sheet.write(0, 0, 'DATA QUALITY REPORT', header_format)
        quality_sheet.write(1, 0, f'Analysis Date: {summary["date"]}')
        quality_sheet.write(2, 0, f'Total Symbols in Universe: {len(RAW_SYMBOLS)}')
        quality_sheet.write(3, 0, f'Successfully Processed: {len(df)}')
        quality_sheet.write(4, 0, f'Failed/Missing: {len(RAW_SYMBOLS) - len(df)}')
        
        if data_quality_report:
            row = 6
            quality_sheet.write(row, 0, 'MISSING SYMBOLS REPORT', workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'}))
            quality_sheet.write(row+1, 0, 'Symbol')
            quality_sheet.write(row+1, 1, 'Reason')
            
            for i, (symbol, reason) in enumerate(data_quality_report.items()):
                if i < 100:  # Show first 100 missing
                    quality_sheet.write(row+2+i, 0, symbol)
                    quality_sheet.write(row+2+i, 1, reason)
            
            if len(data_quality_report) > 100:
                quality_sheet.write(row+2+100, 0, f'... and {len(data_quality_report) - 100} more')
        
        # Auto-fit columns
        for sheet_name in writer.sheets:
            if sheet_name != '_FilterData':
                try:
                    writer.sheets[sheet_name].autofit()
                except:
                    pass
    
    logger.info(f"Excel file saved as '{filename}'")

# ============================================
# MAIN ANALYSIS FUNCTION
# ============================================

def analyze_stocks():
    """Main analysis function"""
    
    logger.info("Starting Stock Market Technical Analysis...")
    logger.info(f"Total symbols to process: {len(RAW_SYMBOLS)}")
    start_time = time.time()
    
    # Load static data (YOUR CUSTOM DATA)
    static_df = load_static_data()
    if static_df is None:
        logger.error("Cannot proceed without static_data.csv")
        return None
    
    # Track missing symbols for data quality report
    data_quality_report = {}
    all_symbols_set = set(RAW_SYMBOLS)
    static_symbols_set = set(static_df['Symbol'].values)
    
    # Check for symbols missing from static data
    for symbol in all_symbols_set - static_symbols_set:
        data_quality_report[symbol] = "Missing from static_data.csv"
    
    # Fetch Nifty index data
    logger.info("Fetching Nifty index data...")
    nifty_hist = fetch_nifty_index()
    if nifty_hist is None:
        logger.warning("Could not fetch Nifty index data. Beta values will be set to 1.0")
    
    # Download price data
    logger.info("Downloading price data...")
    stock_data_map = download_universe(RAW_SYMBOLS)
    logger.info(f"Downloaded price data for {len(stock_data_map)} symbols")
    
    # Process each stock
    logger.info("Processing stock data...")
    all_data = []
    failed_symbols = []
    
    for symbol in RAW_SYMBOLS:
        # Check if symbol exists in static data
        if symbol not in static_symbols_set:
            if symbol not in data_quality_report:
                data_quality_report[symbol] = "Missing from static_data.csv"
            continue
        
        hist = stock_data_map.get(symbol)
        if hist is not None and not hist.empty:
            processed = process_stock_data(symbol, hist, nifty_hist, static_df)
            if processed:
                all_data.append(processed)
            else:
                failed_symbols.append(symbol)
                data_quality_report[symbol] = "Failed during processing (insufficient data or error)"
        else:
            failed_symbols.append(symbol)
            data_quality_report[symbol] = "No price data from Yahoo Finance"
    
    if failed_symbols:
        logger.warning(f"Failed to process {len(failed_symbols)} symbols")
        
    
    if not all_data:
        logger.error("No data processed!")
        return None
    
    df = pd.DataFrame(all_data)
    logger.info(f"Processed data for {len(df)} stocks in {time.time() - start_time:.1f} seconds")
    
    # Calculate scores
    logger.info("Calculating technical scores...")
    df['Trend Score'] = df.apply(calculate_trend_score, axis=1)
    df['Momentum Score'] = df.apply(calculate_momentum_score, axis=1)
    df['Early Detection Score'] = df.apply(calculate_early_detection_score, axis=1)
    
    for mode, weights in SCORE_CONFIG.items():
        df[f'Composite_{mode}'] = df.apply(lambda x: calculate_composite_score(x, weights), axis=1)
    
    df['Composite Score'] = df['Composite_balanced']
    df['Trend Status'] = df.apply(calculate_trend_status, axis=1)
    df['Action'] = df.apply(lambda x: calculate_action(x['Trend Status'], x['Trend Score'], 
                                                       x['Momentum Score'], x['Early Detection Score']), axis=1)
    df['Tier'] = df.apply(lambda x: calculate_tier(x['Trend Status'], x['Composite Score']), axis=1)
    
    # Calculate metrics
    logger.info("Calculating comparative metrics...")
    df = calculate_comparative_metrics(df)
    df = scan_breakouts(df)
    
    # Calculate sector metrics
    logger.info("Calculating sector metrics...")
    sector_metrics = df.groupby('Sector').agg({
        'CHANGEPCT': 'mean', '3 day %': 'mean', '1 week % change': 'mean', '1m % change': 'mean',
        '3m % change': 'mean', '6m % change': 'mean', '1 year return': 'mean',
        'Momentum Score': 'mean', 'Trend Score': 'mean', 'Early Detection Score': 'mean', 'Symbol': 'count'
    }).round(2)
    
    sector_metrics.columns = ['1D_avg', '3D_avg', '1W_avg', '1M_avg', '3M_avg', '6M_avg', '1Y_avg',
                               'Momentum', 'Trend', 'Early_Detection', 'stock_count']
    sector_metrics['Momentum_Shift'] = sector_metrics['1W_avg'] - sector_metrics['1M_avg']
    
    def classify_shift(shift):
        if shift > 3: return "Strong Momentum"
        elif shift > 1: return "Emerging"
        elif shift >= -1: return "Stable"
        elif shift >= -3: return "Lagging"
        else: return "Strong Lagging"
    
    sector_metrics['Rotation_Phase'] = sector_metrics['Momentum_Shift'].apply(classify_shift)
    sector_metrics = sector_metrics.sort_values('1W_avg', ascending=False)
    
    # Industry metrics
    industry_metrics = df.groupby('Industry').agg({
        '1 week % change': 'mean', '1m % change': 'mean', '3m % change': 'mean',
        'Momentum Score': 'mean', 'Symbol': 'count'
    }).round(2)
    industry_metrics.columns = ['1W_avg', '1M_avg', '3M_avg', 'Momentum', 'stock_count']
    industry_metrics['Momentum_Shift'] = industry_metrics['1W_avg'] - industry_metrics['1M_avg']
    industry_metrics = industry_metrics.sort_values('1W_avg', ascending=False)
    
    # Cap metrics
    cap_metrics = df.groupby('Cap_Class').agg({
        'CHANGEPCT': 'mean', '1 week % change': 'mean', '1m % change': 'mean', '3m % change': 'mean', 'Symbol': 'count'
    }).round(2)
    cap_metrics.columns = ['Today_Change', '1W_avg', '1M_avg', '3M_avg', 'stock_count']
    
    # Additional analyses
    drilldown_df = create_sector_industry_drilldown(df)
    breakout_df = df[df['Breakout_Type'] != 'None'][['Symbol', 'Sector', 'Industry', 'Cap_Class', 'CMP', 'CHANGEPCT', 
                                                      'Volume Ratio', 'Breakout_Type']].sort_values('CHANGEPCT', ascending=False)
    rotation_timeline = calculate_sector_rotation_timeline(sector_metrics)
    summary = generate_summary(df, sector_metrics, breakout_df)
    
    # Create Excel output
    logger.info("Creating Excel output...")
    create_excel_output(df, sector_metrics, industry_metrics, cap_metrics, drilldown_df, breakout_df, 
                       rotation_timeline, summary, data_quality_report, filename="stock_market_analysis.xlsx")
    
    # Display summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("STOCK MARKET TECHNICAL ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Stocks Analyzed: {len(df)}")
    print(f"Failed/Missing: {len(RAW_SYMBOLS) - len(df)}")
    print(f"Analysis Time: {elapsed:.1f} seconds")
    
    print("\n" + "-"*40)
    print("MARKET CAP DISTRIBUTION")
    print("-"*40)
    for cap_class, count in df['Cap_Class'].value_counts().items():
        print(f"  {cap_class}: {count} stocks")
    
    print("\n" + "-"*40)
    print("TOP SECTORS (1W RETURN)")
    print("-"*40)
    for sector, row in sector_metrics.head(5).iterrows():
        print(f"  {sector}: {row['1W_avg']:.2f}% (1W), {row['1M_avg']:.2f}% (1M)")
    
    print("\n" + "-"*40)
    print("BREAKOUT SUMMARY")
    print("-"*40)
    for breakout_type, group in breakout_df.groupby('Breakout_Type'):
        print(f"  {breakout_type}: {len(group)} stocks")
    
    print("\n" + "-"*40)
    print("RECOMMENDATIONS")
    print("-"*40)
    for rec in summary['recommendations']:
        print(f"  • {rec}")
    
    print("\n" + "="*80)
    print(f"✅ Analysis Complete! Excel file saved as: stock_market_analysis.xlsx")
    print(f"   Data Quality Report available in Sheet 10")
    print("="*80)
    
    return df

# ============================================
# EXECUTION
# ============================================

if __name__ == "__main__":
    df = analyze_stocks()
