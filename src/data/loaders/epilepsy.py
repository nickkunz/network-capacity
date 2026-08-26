## libraries
import sys
import logging
import re
import numpy as np
import pandas as pd
import igraph as ig
from pathlib import Path
from typing import Optional, Dict, Any

## path
root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

## modules
from src.vectorizers.invariants import GraphInvariants
from src.vectorizers.signatures import ProcessSignatures
from src.data.helpers import (
    _request_with_retry,
    _create_igraph_object
)

## logging
logger = logging.getLogger(__name__)

## load eeg electrode network
def _load_network_epilepsy() -> pd.DataFrame:
    """
    Desc: 
        Loads the fixed 19-channel EEG electrode network (10-20 international system).
        Represents spatial connectivity of scalp electrodes.
    
    Args:
        None
    
    Returns:
        pd.DataFrame: A DataFrame representing the EEG electrode network.
    
    Raises:
        None
    """
    data = {
        "electrode": [
            "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
            "T3", "C3", "CZ", "C4", "T4",
            "T5", "P3", "PZ", "P4", "T6",
            "O1", "O2"
        ]
    }
    return pd.DataFrame(data)

## build eeg electrode network
def _build_network_epilepsy(data: pd.DataFrame) -> tuple[list[str], list[tuple]]:
    nodes = data["electrode"].tolist()
    
    ## define anatomically meaningful connections in 10-20 system
    edges = [
        ("FP1", "F7"), ("FP1", "F3"), ("FP2", "F4"), ("FP2", "F8"),
        ("F7", "T3"), ("F3", "C3"), ("F3", "FZ"), ("FZ", "CZ"), ("F4", "C4"), ("F4", "FZ"), ("F8", "T4"),
        ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"), ("C4", "T4"),
        ("T3", "T5"), ("C3", "P3"), ("CZ", "PZ"), ("C4", "P4"), ("T4", "T6"),
        ("T5", "P3"), ("P3", "PZ"), ("PZ", "P4"), ("P4", "T6"),
        ("T5", "O1"), ("P3", "O1"), ("PZ", "O1"), ("PZ", "O2"), ("P4", "O2"), ("T6", "O2"),
        ("O1", "O2")
    ]
    return nodes, edges

## load seizure events for a specific patient
def _load_events_epilepsy(url: str, ids: str) -> pd.DataFrame:
    
    ## load data
    url_summary = f"{url}/{ids}/{ids}-summary.txt"
    response = _request_with_retry(url = url_summary)
    
    ## parse summary file for seizure annotations
    seizures = list()
    lines = response.text.split('\n')
    current_file = None
    file_start_seconds = None
    previous_start_seconds = None
    recording_day = 0
    fallback_day = 0
    fallback_start_seconds = 0
    for i in lines:
        i = i.strip()
        
        ## extract recording file name
        if i.startswith('File Name:'):
            current_file = i.split(':', 1)[1].strip()
            file_start_seconds = None
            file_token = Path(current_file).stem.rsplit('_', 1)[1]
            file_number = int(re.match(r'\d+', file_token).group())
            fallback_day, fallback_hour = divmod(file_number - 1, 24)
            fallback_start_seconds = fallback_hour * 3600
            
        ## extract file start date/time (format: "14:43:04")
        if i.startswith('File Start Time:'):
            file_start_time = i.split(':', 1)[1].strip()
            file_start_seconds = int(pd.to_timedelta(file_start_time).total_seconds())
            if previous_start_seconds is not None and file_start_seconds < previous_start_seconds:
                recording_day += 1
            previous_start_seconds = file_start_seconds
            
        ## extract seizure onset (seconds from file start)
        if 'Seizure' in i and 'Start Time' in i:
            try:
                onset_str = i.split()[-2]  ## get seconds value
                onset_seconds = int(onset_str)
                
                if current_file:
                    if file_start_seconds is None:
                        elapsed_seconds = fallback_day * 86400 + fallback_start_seconds + onset_seconds
                    else:
                        elapsed_seconds = recording_day * 86400 + file_start_seconds + onset_seconds
                    seizures.append({
                        'file': current_file,
                        'day': elapsed_seconds // 86400
                    })
            except (ValueError, IndexError):
                continue
    
    if not seizures:
        raise RuntimeError(f"No seizures found in {ids} summary")
    
    return pd.DataFrame(seizures)[['day']].sort_values('day').reset_index(drop = True)

def _process_events_epilepsy(events: pd.DataFrame) -> pd.DataFrame:
    return events.groupby('day').size().reset_index(name = 'target')

## epilepsy seizure network
class EpilepsyProcessor:
    def __init__(self, url: str, ids: list[str]):
        self.url = url
        self.ids = ids
        self.data_network: Optional[pd.DataFrame] = None
        self.data_events: Optional[pd.DataFrame] = None
        self.graph: Optional[ig.Graph] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.signatures: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        self.data_network = _load_network_epilepsy()
        
        all_events = []
        day_offset = 0
        for i in self.ids:
            try:
                events = _load_events_epilepsy(url = self.url, ids = i)
                if not events.empty:
                    events['day'] += day_offset
                    events['patient'] = i
                    all_events.append(events)
                    day_offset = int(events['day'].max()) + 1
            except Exception as e:
                logger.warning(f"Skipping {i}: {e}")
                continue
        
        if not all_events:
            raise RuntimeError("No seizure events found across all patients")
        
        self.data_events = pd.concat(all_events, ignore_index=True)
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.data_network is None:
            self.load_data()
        nodes, edges = _build_network_epilepsy(data=self.data_network)
        self.graph = _create_igraph_object(nodes=nodes, edges=edges)
        self.invariants = GraphInvariants(graph=self.graph).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.data_events is None:
            self.load_data()
        self.events = _process_events_epilepsy(events = self.data_events)
        return self

    def process_signatures(self):
        """Computes process signatures on daily seizure counts."""
        if self.events is None:
            self.process_events()
        self.signatures = ProcessSignatures(
            data = self.events.copy(),
            sort_by = ["day"],
            target = "target"
        ).all()
        return self

    def run(self):
        """ Executes the pipeline and returns the final result. """
        self.process_network()
        self.process_signatures()
        self.process_events()
        return {
            "invariants": self.invariants,
            "signatures": self.signatures,
            "events": self.events.to_dict(orient = "records")
        }
