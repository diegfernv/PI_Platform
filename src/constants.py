from colorama import Fore

LIST_RESIDUES = [
    'A', 
    'C', 
    'D', 
    'E', 
    'F', 
    'G', 
    'H', 
    'I', 
    'N', 
    'K', 
    'L', 
    'M', 
    'P', 
    'Q', 
    'R', 
    'S', 
    'T', 
    'V', 
    'W', 
    'Y'
]

POSITION_RESIDUES = {'A' :0, 'C' : 1, 'D' : 2, 'E' : 3, 'F' : 4, 
                    'G' : 5, 'H' : 6, 'I' : 7, 'N' : 8, 'K' : 9, 
                    'L' : 10, 'M' : 11, 'P' : 12, 'Q' : 13, 'R' : 14, 
                    'S' : 15, 'T' : 16, 'V' : 17, 'W' : 18, 'Y' : 19}


class COLORS:
    DEBUG = Fore.BLUE
    INFO = Fore.WHITE
    WARNING = Fore.YELLOW
    ERROR = Fore.RED
    CRITICAL = Fore.MAGENTA
    RESET = Fore.RESET

ENCODING_TYPES = [
    "FFT",
    "Frequency",
    "KMer",
    "One-Hot",
    "Ordinal",
    "Physicochemical",
    "Embedding"
]
EMBEDDING_MODELS = {
    "Ankh2": "ElnaggarLab/ankh2-ext1",
    "Bert": "Rostlab/prot_bert",
    "ESM2": "facebook/esm2_t6_8M_UR50D",
    "ESMC": "esmc_600",
    "Mistral": "RaphaelMourad/Mistral-Prot-v1-134M",
    "Prot T5": "Rostlab/prot_t5_xl_uniref50"
}