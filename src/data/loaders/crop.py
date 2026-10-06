## libraries
import io
import os
import sys
import certifi
import logging
import configparser
import pandas as pd
import numpy as np
import igraph as ig
from pathlib import Path
from typing import Optional, Any

## path
root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

## modules
from src.vectorizers.invariants import GraphInvariants
from src.vectorizers.signatures import ProcessSignatures
from src.data.helpers import (
    _create_igraph_object,
    _request_with_retry
)

## logging
logger = logging.getLogger(__name__)

## configs
config = configparser.ConfigParser()
config.read(filenames = os.path.join(root, 'conf', 'settings.ini'))

## default url for rader 2016 database
DEFAULT_URL_CROP = config['urls'].get(
    'URL_CROP',
    fallback = 'https://raw.githubusercontent.com/ibartomeus/CropPol/master/Processing_files/Datasets_Processing/RADER%202016%20DATABASE/Individual%20CSV/'
).strip('"')

## list of primary study files in rader et al. 2016 database
RADER_STUDY_FILES: list[str] = [
    "Bommarco_Rundlof_2009.csv",
    "Brittain_Klein_2008.csv",
    "Cunningham_2001.csv",
    "Cunningham_2006.csv",
    "Hipolito_2005.csv",
    "Jauker_2006.csv",
    "Lindstrom_2011.csv",
    "Lindstrom_2012.csv",
    "Reemer_Kleijn_2010.csv",
    "Reemer_Kleijn_2010_Apple.csv",
    "Reemer_Kleijn_2011.csv",
    "Reemer_Kleijn_2011_Apple.csv",
    "Rundlof_2011.csv",
    "Rundlof_2012.csv",
    "Scheper_2011.csv",
    "Scheper_2012.csv",
    "Schueep_NA.csv",
    "Szentgyorgyi_NA.csv",
    "Vergara_2004.csv",
    "Winfree_griffin_2008.csv",
    "Winfree_griffin_2010.csv",
    "Winfree_griffin_2011.csv",
    "Winfree_griffin_2012.csv",
    "mayfield_NA.csv",
    "pisanty_mandelik_2009.csv",
    "pisanty_mandelik_2010.csv",
    "pisanty_mandelik_2011.csv",
    "stanley_stout_2dataset_2009.csv",
]

## extract calendar dates from study records
def _extract_calendar_dates(data: pd.DataFrame) -> pd.Series:
    """
    Desc:
        Extract calendar dates from year, month, day columns or convert
        Excel serial dates where explicit date components are missing.

    Args:
        data: DataFrame containing study records.

    Returns:
        pd.Series: Series of extracted datetime.date objects or NaT values.

    Raises:
        None
    """
    dates_ymd = pd.to_datetime(
        arg = pd.DataFrame(
            data = {
                "year": pd.to_numeric(arg = data["Year_of_study"], errors = "coerce"),
                "month": pd.to_numeric(arg = data["month_of_study"], errors = "coerce"),
                "day": pd.to_numeric(arg = data["day_of_study"], errors = "coerce")
            }
        ),
        errors = "coerce"
    )

    if "date_round1" in data.columns:
        serial_nums = pd.to_numeric(arg = data["date_round1"], errors = "coerce")
        serial_dates = pd.to_datetime(
            arg = serial_nums,
            unit = "D",
            origin = "1899-12-30",
            errors = "coerce"
        )
        dates = dates_ymd.fillna(value = serial_dates)
    else:
        dates = dates_ymd

    return dates.dt.date

## load individual rader study
def _load_rader_study(base_url: str, fname: str) -> pd.DataFrame:
    """
    Desc:
        Load an individual study CSV from the Rader 2016 database subset.

    Args:
        base_url: Base URL directory containing the individual CSV files.
        fname: Name of the CSV file to retrieve.

    Returns:
        pd.DataFrame: Loaded DataFrame for the study.

    Raises:
        RuntimeError: If the file cannot be retrieved or read.
    """
    os.environ['SSL_CERT_FILE'] = certifi.where()
    url = f"{base_url.rstrip('/')}/{fname}"
    try:
        response = _request_with_retry(
            url = url,
            timeout = 30,
            use_cache = True,
            cache_namespace = "crop"
        )
        data = pd.read_csv(
            filepath_or_buffer = io.StringIO(initial_value = response.text),
            low_memory = False
        )
        if data.empty:
            raise RuntimeError(f"Empty data retrieved for {fname}.")
        return data
    except Exception as e:
        raise RuntimeError(f"Error loading Rader study {fname} from {url}: {e}")

## build croppol network data
def build_network_croppol(data: list[pd.DataFrame] | pd.DataFrame) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Desc:
        Construct undirected bipartite crop-pollinator interaction graph nodes
        and edges from primary census tables.

    Args:
        data: List of DataFrames or single DataFrame containing study observations.

    Returns:
        tuple[list[str], list[tuple[str, str]]]: Unique nodes and unique edges.

    Raises:
        None
    """
    studies = [data] if isinstance(data, pd.DataFrame) else data
    nodes = set()
    edges = set()

    for df in studies:
        if df.empty or "crop" not in df.columns:
            continue

        cols_list = df.columns.tolist()
        meta_end = cols_list.index("final_fruitset") + 1 if "final_fruitset" in cols_list else 29
        species_cols = [c for c in df.columns[meta_end:] if c != "calendar_date"]
        species_df = df[species_cols].apply(
            func = pd.to_numeric, 
            errors = "coerce"
        ).fillna(value = 0)

        for crop_name, sub in df.groupby(by = "crop"):
            sub_species = species_df.loc[sub.index]
            species_totals = sub_species.sum(axis = 0)
            active_pollinators = species_totals[species_totals > 0].index.tolist()

            nodes.add(str(crop_name))
            for poll in active_pollinators:
                nodes.add(str(poll))
                edges.add((str(crop_name), str(poll)))

    return sorted(list(nodes)), sorted(list(edges))

## process croppol events
def _process_events_croppol(data: list[pd.DataFrame] | pd.DataFrame) -> pd.DataFrame:
    """
    Desc:
        Extract row-level pollinator interaction counts across insect species
        columns for records with confirmed calendar dates and aggregate to
        daily counts.

    Args:
        data: List of DataFrames or single DataFrame containing study observations.

    Returns:
        pd.DataFrame: Daily event count series with date and target columns.

    Raises:
        ValueError: If no valid daily observations are found across studies.
    """
    studies = [data] if isinstance(data, pd.DataFrame) else data
    all_events = list()

    for df in studies:
        if df.empty or "crop" not in df.columns:
            continue

        dates = _extract_calendar_dates(data = df)
        valid_mask = dates.notna()
        if not valid_mask.any():
            continue

        cols_list = df.columns.tolist()
        meta_end = cols_list.index("final_fruitset") + 1 if "final_fruitset" in cols_list else 29
        species_cols = [c for c in df.columns[meta_end:] if c != "calendar_date"]
        species_df = df.loc[valid_mask, species_cols].apply(
            func = pd.to_numeric, 
            errors = "coerce"
        ).fillna(value = 0)

        sub_dates = dates.loc[valid_mask]
        row_counts = species_df.sum(axis = 1).astype(dtype = int)

        events_slice = pd.DataFrame(
            data = {
                "date": sub_dates.values,
                "target": row_counts.values
            }
        )
        all_events.append(events_slice[events_slice["target"] > 0])

    if not all_events:
        raise ValueError("No valid daily events found in CropPol Rader studies.")

    combined = (
        pd.concat(objs = all_events, ignore_index = True)
        .groupby(by = "date", as_index = False)["target"]
        .sum()
        .sort_values(by = "date")
        .reset_index(drop = True)
    )
    combined["date"] = combined["date"].astype(dtype = str)
    return combined

## crop pollinator network
class CropProcessor:
    def __init__(
        self, 
        url: Optional[str] = None,
        url_sampling: Optional[str] = None, 
        url_field: Optional[str] = None
    ) -> None:
        self.url = url or DEFAULT_URL_CROP
        self.url_sampling = url_sampling
        self.url_field = url_field
        self.data_raw: Optional[list[pd.DataFrame]] = None
        self.graph: Optional[ig.Graph] = None
        self.invariants: Optional[dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None
        self.signatures: Optional[dict[str, Any]] = None

    def load_data(self) -> 'CropProcessor':
        """
        Desc:
            Load raw study tables from the Rader 2016 database.

        Args:
            None

        Returns:
            CropProcessor: Self instance with loaded data_raw.

        Raises:
            RuntimeError: If data loading fails.
        """
        if self.data_raw is None:
            self.data_raw = list()
            for fname in RADER_STUDY_FILES:
                try:
                    df = _load_rader_study(
                        base_url = self.url,
                        fname = fname
                    )
                    self.data_raw.append(df)
                except Exception as e:
                    logger.warning(f"Could not load Rader study {fname}: {e}")
            if not self.data_raw:
                raise RuntimeError("Failed to load any Rader 2016 study datasets.")
        return self

    def process_network(self) -> 'CropProcessor':
        """
        Desc:
            Build undirected bipartite crop-pollinator interaction graph and
            compute 21 graph-theoretic invariants.

        Args:
            None

        Returns:
            CropProcessor: Self instance with graph and invariants populated.

        Raises:
            None
        """
        if self.data_raw is None:
            self.load_data()
        if self.data_raw is None:
            raise RuntimeError("Data failed to load.")
        nodes, edges = build_network_croppol(data = self.data_raw)
        self.graph = _create_igraph_object(nodes = nodes, edges = edges)
        self.invariants = GraphInvariants(graph = self.graph).all()
        return self

    def process_events(self) -> 'CropProcessor':
        """
        Desc:
            Extract and aggregate discrete daily event counts across confirmed
            calendar observation dates.

        Args:
            None

        Returns:
            CropProcessor: Self instance with events populated.

        Raises:
            None
        """
        if self.data_raw is None:
            self.load_data()
        if self.data_raw is None:
            raise RuntimeError("Data failed to load.")
        self.events = _process_events_croppol(data = self.data_raw)
        return self

    def process_signatures(self) -> 'CropProcessor':
        """
        Desc:
            Compute universal process signatures from the ordered daily event counts.

        Args:
            None

        Returns:
            CropProcessor: Self instance with signatures populated.

        Raises:
            None
        """
        if self.events is None:
            self.process_events()
        if self.events is None:
            raise RuntimeError("Events failed to process.")

        self.signatures = ProcessSignatures(
            data = self.events.copy(),
            sort_by = ["date"],
            target = "target"
        ).all()
        return self

    def run(self) -> dict[str, Any]:
        """
        Desc:
            Execute the complete data pipeline for the crop pollinator system.

        Args:
            None

        Returns:
            dict[str, Any]: Dictionary containing invariants, signatures, and events.

        Raises:
            None
        """
        self.process_network()
        self.process_events()
        self.process_signatures()
        if self.invariants is None or self.signatures is None or self.events is None:
            raise RuntimeError("Pipeline failed to produce all outputs.")
        return {
            "invariants": self.invariants,
            "signatures": self.signatures,
            "events": self.events.to_dict(orient = "records")
        }
