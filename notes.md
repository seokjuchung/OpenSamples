# 2025-12-24: Re-eval ub Open for better anomaly metric
1. Need to check whether `File` from `pynuml.io` can be used: Issue still persists. Need to replace with `File` from `file.py`
2. Remove information not needed: Optical, Pandora? Pandora could be used as classification reference
3. Plane 0 and 1 already deconvolved. What is the difference with Hits? After detsim and some processing? Only difference is gaussian blur, still, use wire data


# 2025-12-29: Preprocessing data with given labels - planning
1. Take SPINE NC Delta Selection as reference. What MC Truth is used for signal?
1. What is available in uB open samples?
    1. NC/CC
    1. Particle Category in G4 stage:
    ```
    class category(enum.Enum):
    pion = 0
    muon = 1
    kaon = 2
    proton = 3
    electron = 4
    michel = 5
    delta = 6
    other = 7
    photon = 8
    ```
    1. Should be able to label based on topology. e.g. 1g0e0mu0pi, etc ...
    1. Impossible to do interaction based, like NC Delta, NC CO, etc. Need GENIE info
1. Try simple first. Train with track-like particles only, evaluate with shower-like particles
1. ISSUE: `category` and `g4_pdg` not aligning? Ask Giuseppe
1. How to preprocess data?
    1. Find Neutrino vertex
    1. Cut [Wire Neg, Wire Pos] for Wire
    1. Cut [Time Neg, Time Pos] for Timetick
    1. Save sparse format
    1. TODO: Figure out cut values
        1. Look at distribution? Fixed size, need reference wire/time range with respect to neutrino vertex
        2. Event by event: Take min/max of wire/time tick after removing low energy depostions (`cateogory`: other) $\rightarrow$ This seems more reliable


# 2025-12-30: Preprocessing data with given labels - execution
1. Procedure (per event)
    1. Take neutrino interaction
    1. Remove `category`: other
    1. Look at min/max wire/timetick on plane2
    1. Save square region of wire data to separate array
        1. Do downsizing? Look what other studies using uB open data did
            * The sparsePIX paper used 6 times downsizing with 10/100 cutoff as given from the notebook
    