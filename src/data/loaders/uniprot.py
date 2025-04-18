import requests, re, zlib, json, time, math, logging, sys
from functools import partial
from typing import Any, Dict, List, Optional, Callable
import pandas as pd
from requests.adapters import HTTPAdapter, Retry
from xml.etree import ElementTree
from urllib.parse import urlparse, parse_qs, urlencode
from tqdm import tqdm
from logger import get_logger

API_URL = "https://rest.uniprot.org"
POLLING_INTERVAL = 3 

class UniProtInterface():
    def __init__(self, total_retries=5):
        self.retries = Retry(total=total_retries, backoff_factor=0.25, status_forcelist=[ 500, 502, 503, 504 ])
        self.session = requests.Session()
        self.session.mount('https://', HTTPAdapter(max_retries=self.retries))

    def check_response(self, response):
        try:
            response.raise_for_status()
        except requests.HTTPError:
            print(response.json())
            raise

    def submit_id_mapping(self, from_db: str, to_db: str, ids: list):
        request = requests.post(
            f"{API_URL}/idmapping/run",
        data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
        )
        self.check_response(request)
        return request.json()["jobId"]
    
    def print_progress_batches(self, batch_index, size, total):
        n_fetched = min((batch_index + 1) * size, total)
        #print(f"Fetched: {n_fetched} / {total}")

    def combine_batches(self, all_results, batch_results, file_format):
        if file_format == "json":
            for key in ("results", "failedIds"):
                if key in batch_results and batch_results[key]:
                    all_results[key] += batch_results[key]
        elif file_format == "tsv":
            return all_results + batch_results[1:]
        else:
            return all_results + batch_results
        return all_results

    def decode_results(self, response, file_format, compressed):
        if compressed:
            decompressed = zlib.decompress(response.content, 16 + zlib.MAX_WBITS)
            if file_format == "json":
                j = json.loads(decompressed.decode("utf-8"))
                return j
            elif file_format == "tsv":
                return [line for line in decompressed.decode("utf-8").split("\n") if line]
            elif file_format == "xlsx":
                return [decompressed]
            elif file_format == "xml":
                return [decompressed.decode("utf-8")]
            else:
                return decompressed.decode("utf-8")
        elif file_format == "json":
            return response.json()
        elif file_format == "tsv":
            return [line for line in response.text.split("\n") if line]
        elif file_format == "xlsx":
            return [response.content]
        elif file_format == "xml":
            return [response.text]
        return response.text

    def get_xml_namespace(self, element):
        m = re.match(r"\{(.*)\}", element.tag)
        return m.groups()[0] if m else ""

    def merge_xml_results(self, xml_results):
        merged_root = ElementTree.fromstring(xml_results[0])
        for result in xml_results[1:]:
            root = ElementTree.fromstring(result)
            for child in root.findall("{http://uniprot.org/uniprot}entry"):
                merged_root.insert(-1, child)
        ElementTree.register_namespace("", self.get_xml_namespace(merged_root[0]))
        return ElementTree.tostring(merged_root, encoding="utf-8", xml_declaration=True)

    def get_id_mapping_results_search(self, url):
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        file_format = query["format"][0] if "format" in query else "json"
        if "size" in query:
            size = int(query["size"][0])
        else:
            size = 500
            query["size"] = size
        compressed = (
            query["compressed"][0].lower() == "true" if "compressed" in query else False
        )
        parsed = parsed._replace(query=urlencode(query, doseq=True))
        url = parsed.geturl()
        request = self.session.get(url)
        self.check_response(request)
        results = self.decode_results(request, file_format, compressed)
        total = int(request.headers["x-total-results"])
        self.print_progress_batches(0, size, total)
        for i, batch in enumerate(self.get_batch(request, file_format, compressed), 1):
            results = self.combine_batches(results, batch, file_format)
            self.print_progress_batches(i, size, total)
        if file_format == "xml":
            return self.merge_xml_results(results)
        return results

    def get_id_mapping_results_link(self, job_id):
        url = f"{API_URL}/idmapping/details/{job_id}"
        request = self.session.get(url)
        self.check_response(request)
        return request.json()["redirectURL"]

    def get_next_link(self, headers):
        re_next_link = re.compile(r'<(.+)>; rel="next"')
        if "Link" in headers:
            match = re_next_link.match(headers["Link"])
            if match:
                return match.group(1)

    def get_batch(self, batch_response, file_format, compressed):
        batch_url = self.get_next_link(batch_response.headers)
        while batch_url:
            batch_response = self.session.get(batch_url)
            batch_response.raise_for_status()
            yield self.decode_results(batch_response, file_format, compressed)
            batch_url = self.get_next_link(batch_response.headers)

    def check_id_mapping_results_ready(self, job_id):
        while True:
            request = self.session.get(f"{API_URL}/idmapping/status/{job_id}")
            self.check_response(request)
            j = request.json()
            if "jobStatus" in j:
                if j["jobStatus"] == "RUNNING":
                    #print(f"Retrying in {POLLING_INTERVAL}s")
                    time.sleep(POLLING_INTERVAL)
                else:
                    raise Exception(j["jobStatus"])
            else:
                return bool(j["results"] or j["failedIds"])

class UniProtLoader(UniProtInterface):
    '''
    This class is responsible for handling the interaction with the UniProt API.

    Attributes:
    from_db (str): The database to download the sequences from.
    to_db (str): The database to download the sequences to.
    auto_db (bool): Whether to automatically detect the database type.
    dataset (pd.DataFrame): The dataset to download the sequences from.
    column_ids (str): The column name in the dataset that contains the IDs.
    '''
    def __init__(
            self, 
            dataset: pd.DataFrame = None,
            column_ids: str = None,
            auto_db: bool = True,
            from_db: str = "UniProtKB_AC-ID", 
            to_db: str = "UniProtKB"):
        self.dataset = dataset
        self.column_ids = column_ids
        self.auto_db = auto_db
        self.from_db = from_db
        self.to_db = to_db
        self.logger = get_logger("UniProtLoader")

        self.db_config = {
            'uniprot': {
                'patterns': [r'^[A-N,R-Z][0-9][A-Z][A-Z, 0-9][A-Z, 0-9][0-9]$',
                            r'^[A-N,R-Z][0-9][A-Z][A-Z, 0-9][A-Z, 0-9][0-9][A-Z][A-Z, 0-9][A-Z, 0-9][0-9]$',
                            r'^[OPQ][0-9][A-Z0-9][A-Z0-9][A-Z0-9][0-9]$'],
                'from_db': 'UniProtKB_AC-ID',
                'to_db': 'UniProtKB'
            },
            'pdb': {
                'patterns': [r'^[0-9][A-Z0-9]{3}$'],
                'from_db': 'PDB',
                'to_db': 'UniProtKB'
            }
        }

        self.field_map = {
            'uniprot_id': ('from', self._extract_simple),
            'entry_type': ('to.entryType', self._extract_simple),
            'protein_name': ('to.proteinDescription.recommendedName.fullName.value', self._extract_simple),
            'ec_numbers': ('to.proteinDescription.recommendedName.ecNumbers', self._extract_ec_numbers),
            'organism': ('to.organism.scientificName', self._extract_simple),
            'taxon_id': ('to.organism.taxonId', self._extract_simple),
            'sequence': ('to.sequence.value', self._extract_simple),
            'length': ('to.sequence.length', self._extract_simple),
            'go_terms': ('to.uniProtKBCrossReferences', self._extract_go_terms),
            'pfam_ids': ('to.uniProtKBCrossReferences', self._extract_pfam_ids),
            'references': ('to.references', self._extract_references),
            'features': ('to.features', self._extract_features),
            'keywords': ('to.keywords', self._extract_keywords),
        }

        self.results = []

        super().__init__()

    def _identify_id_type(self, id_str: str) -> str:
        """Identifica el tipo de ID basado en patrones regex"""
        if not isinstance(id_str, str):
            return None
            
        for db_type, config in self.db_config.items():
            for pattern in config['patterns']:
                if re.fullmatch(pattern, id_str):
                    return db_type
        
        return None

    def _group_ids_by_type(self, ids: List[str]) -> Dict[str, List[str]]:
        """Agrupa IDs por su tipo detectado"""
        grouped = {db_type: [] for db_type in self.db_config}
        grouped['unknown'] = []
        
        for id_str in ids:
            if not isinstance(id_str, str):
                continue
                
            id_type = self._identify_id_type(id_str)
            if id_type in grouped:
                grouped[id_type].append(id_str)
            else:
                grouped['unknown'].append(id_str)
            
        logger_str = [f"{db_type.capitalize()}: {len(ids)}" for db_type, ids in grouped.items() if ids]
        logger_str.append(f"Unknown: {len(grouped['unknown'])}")
        logger_str = ", ".join(logger_str)
        self.logger.info(f"Automated database detection results | {logger_str}")
        return grouped

    def get_sequence():
        pass

    def download_batch(self, batch_size: int):
        ids = self.dataset[self.column_ids].dropna().unique().tolist()
        self.logger.info(f"Downloading {len(ids)} sequences")

        if self.auto_db:
            # Automatically detect and group IDs
            id_groups = self._group_ids_by_type(ids)
            
            for db_type, id_list in id_groups.items():
                if not id_list or db_type == 'unknown':
                    continue
                    
                config = self.db_config[db_type]
                self._process_id_batch(
                    ids=id_list,
                    from_db=config['from_db'],
                    to_db=config['to_db'],
                    batch_size=batch_size,
                    db_type=db_type
                )
        else:
            # Manually use the provided from_db/to_db parameters
            self._process_id_batch(
                ids=ids,
                from_db=self.from_db,
                to_db=self.to_db,
                batch_size=batch_size,
                db_type='manual'
            )

    def _process_id_batch(self, ids: List[str], from_db: str, to_db: str, batch_size: int, db_type: str):
        """Procesa un lote de IDs de un tipo específico"""
        progress_bar = tqdm(
            range(0, len(ids)), 
            desc=f"Processing {db_type} IDs", 
            total=len(ids),
            dynamic_ncols=True,
            ncols=0,
            bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {desc}"
        )
        
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start+batch_size]
            job_id = self.submit_id_mapping(from_db, to_db, batch)
            
            if self.check_id_mapping_results_ready(job_id):
                link = self.get_id_mapping_results_link(job_id)
                results = self.get_id_mapping_results_search(link)
                
                # Add information about the source to the results
                if isinstance(results, dict):
                    for result in results.get('results', []):
                        result['source_db'] = db_type
                    self.results.append(results)
                    
            progress_bar.update(len(batch))
    
    def show_results(self, raw=False):
        if self.results:
            if raw:
                for result in self.results:
                    print(result)
            else:
                print(f"{len(self.results)} results to show")
        else:
            print("No results to show")

    def parse_results(self):
        export_df = pd.DataFrame()

        for result in self.results:
            parsed_results = self.parse(result)
            export_df = pd.concat([export_df, parsed_results], ignore_index=True)

        return export_df

    def parse(self, results: Dict) -> pd.DataFrame:
        """Parse UniProt JSON results into a DataFrame"""
        parsed_data = []
        
        # Process successful results
        for result in results.get('results', []):
            parsed = self._parse_result(result)
            parsed['source_db'] = results.get('source_db', 'unknown')
            parsed_data.append(parsed)
            
        # Process failed IDs
        for failed_id in results.get('failedIds', []):
            parsed_data.append({
                'uniprot_id': failed_id,
                'source_db': results.get('source_db', 'unknown'),
                'status': 'failed'
            })
            
        return pd.DataFrame(parsed_data)
    
    def _parse_result(self, result: Dict) -> Dict:
        """Parse a single UniProt result"""
        parsed = {}
        
        for field, (path, extractor) in self.field_map.items():
            try:
                # Navigate through the path (e.g. 'to.proteinDescription...')
                data = result
                for key in path.split('.'):
                    if key.isdigit():  # For array indices
                        key = int(key)
                    data = data.get(key, {})
                
                # Extract the value using the specific function
                parsed[field] = extractor(data) if data else None
            except (KeyError, AttributeError, IndexError):
                parsed[field] = None
                
        return parsed
    
    # Specific extraction functions
    @staticmethod
    def _extract_simple(value: Any) -> Any:
        """Extracts a simple value from the data"""
        return value
    
    @staticmethod
    def _extract_ec_numbers(ec_data: List) -> List[str]:
        """Extracts EC numbers"""
        return [ec['value'] for ec in ec_data] if isinstance(ec_data, list) else []
    
    @staticmethod
    def _extract_go_terms(xrefs: List) -> List[str]:
        """Extracts GO terms"""
        return [x['id'] for x in xrefs if isinstance(x, dict) and x.get('database') == 'GO']
    
    @staticmethod
    def _extract_pfam_ids(xrefs: List) -> List[str]:
        """Extracts Pfam IDs"""
        return [x['id'] for x in xrefs if isinstance(x, dict) and x.get('database') == 'Pfam']
    
    @staticmethod
    def _extract_references(refs: List) -> List[Dict]:
        """Extracts references"""
        extracted = []
        for ref in refs if isinstance(refs, list) else []:
            citation = ref.get('citation', {})
            extracted.append({
                'title': citation.get('title'),
                'authors': citation.get('authors', []),
                'journal': citation.get('journal'),
                'pub_date': citation.get('publicationDate'),
                'pmid': next((x['id'] for x in citation.get('citationCrossReferences', []) 
                            if x.get('database') == 'PubMed'), None)
            })
        return extracted
    
    @staticmethod
    def _extract_features(features: List) -> List[Dict]:
        """Extracts protein features"""
        return [{
            'type': f.get('type'),
            'description': f.get('description', ''),
            'location': f.get('location', {})
        } for f in features if isinstance(features, list)]
    
    @staticmethod
    def _extract_keywords(keywords: List) -> List[str]:
        """Extracts keywords"""
        return [kw.get('name', '') for kw in keywords if isinstance(keywords, list)]