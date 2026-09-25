
## libraries
import sys
import numpy as np
import pandas as pd
import igraph as ig
from pathlib import Path
from typing import Optional, Dict, Any
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

## path
root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

## modules
from src.vectorizers.invariants import GraphInvariants
from src.vectorizers.signatures import ProcessSignatures
from src.data.helpers import (
    _load_network_pygt,
    _build_network_pygt,
    _create_igraph_object
)

## process metr-la dataset into daily event aggregates
def _process_events_metrla(data: np.ndarray, sample_rate_minutes: int = 5, thres_percentile: int = 1) -> pd.DataFrame:
    
    ## calculate samples per day
    samples_per_day = (24 * 60) // sample_rate_minutes

    ## derive binary congestion via per-node threshold (zeros encode missing readings)
    threshold = np.where(
        (data > 0).any(axis = 0),
        np.percentile(np.where(data > 0, data, np.nan), thres_percentile, axis = 0),
        np.nan,
    )[None, :]
    congested = (data <= threshold) & (data > 0)

    ## detect congestion onset events
    stop_events = np.zeros_like(congested, dtype=np.int32)
    stop_events[1:, :] = (congested[1:, :] & ~congested[:-1, :]).astype(np.int32)
    events_per_sample = stop_events.sum(axis = 1)

    ## trim to full days and reshape for daily aggregation
    num_days = events_per_sample.size // samples_per_day
    if num_days == 0:
        return pd.DataFrame({"day": [], "target": []})
    
    totals = num_days * samples_per_day
    daily_events = events_per_sample[:totals].reshape(-1, samples_per_day).sum(axis=1)

    ## create output dataframe
    return pd.DataFrame({
        "date": pd.to_datetime(
            pd.date_range(
                start = '2012-03-01', 
                periods = num_days, 
                freq = 'D'
            )
        ).date,
        "target": daily_events.astype(np.int64)
    })

## metr-la traffic network
class MetrLaProcessor:
    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.dataset: Optional[DynamicGraphTemporalSignal] = None
        self.speed: Optional[np.ndarray] = None
        self.graph: Optional[ig.Graph] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.signatures: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        loader = METRLADatasetLoader(raw_data_dir = self.raw_data_dir)
        self.dataset = _load_network_pygt(loader = loader)
        raw = np.load(Path(self.raw_data_dir) / "node_values.npy")
        self.speed = np.asarray(raw[:, :, 0], dtype = float)
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.dataset is None:
            self.load_data()
        nodes, edges = _build_network_pygt(dataset = self.dataset)
        self.graph = _create_igraph_object(nodes = nodes, edges = edges)
        self.invariants = GraphInvariants(graph = self.graph).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.speed is None:
            self.load_data()
        self.events = _process_events_metrla(data = self.speed)
        return self

    def process_signatures(self):
        """Computes process signatures over daily congestion events."""
        if self.events is None:
            self.process_events()
        self.signatures = ProcessSignatures(
            data = self.events.copy(),
            sort_by = ["date"],
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
