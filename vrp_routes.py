#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, requests, pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    ORTOOLS_AVAILABLE = True
except Exception:
    ORTOOLS_AVAILABLE = False

@dataclass
class Stop:
    idx:int; id:str; name:str; lat:float; lon:float; service_min:float
@dataclass
class Matrix:
    distances_km: list; durations_min: list

EARTH_RADIUS_KM = 6371.0088

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))

def build_haversine_matrix(stops: List[Stop], avg_speed_kmh: float) -> Matrix:
    n = len(stops)
    dist = [[0.0]*n for _ in range(n)]
    dur = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            d = haversine_km(stops[i].lat, stops[i].lon, stops[j].lat, stops[j].lon)
            dist[i][j] = d
            dur[i][j] = (d / max(1e-6, avg_speed_kmh)) * 60.0
    return Matrix(distances_km=dist, durations_min=dur)

def build_osrm_matrix_batched(
    stops,
    base_url: str = "http://localhost:5000",
    block: int = 100,
    sleep: float = 0.0,
    progress_cb=None,   # optional: progress_cb(done_tiles, total_tiles)
    timeout: int = 120,
) -> Matrix:
    """
    Build full NxN matrix using OSRM /table, in tiles to avoid URL-size limits.
    Distances in km, durations in minutes.

    Args:
        stops: list[Stop]
        base_url: OSRM routed URL (your local server)
        block: tile size (sources x destinations)
        sleep: seconds to sleep between tile requests
        progress_cb: optional callback to report progress
        timeout: request timeout seconds
    """
    import time

    def _osrm_get(url, timeout=120, retries=5, backoff=0.5):
        """GET with simple exponential backoff; retries 429 and 5xx."""
        for i in range(retries):
            try:
                r = requests.get(url, timeout=timeout)
                if r.status_code in (429,) or 500 <= r.status_code < 600:
                    raise requests.HTTPError(f"{r.status_code}: {r.text}", response=r)
                r.raise_for_status()
                return r.json()
            except Exception:
                if i == retries - 1:
                    raise
                time.sleep(backoff * (2 ** i))

    n = len(stops)
    dist_km = [[0.0] * n for _ in range(n)]
    dur_min = [[0.0] * n for _ in range(n)]

    # Pre-cache coordinates
    lats = [s.lat for s in stops]
    lons = [s.lon for s in stops]

    # Tiling plan
    tiles_i = list(range(0, n, block))
    tiles_j = list(range(0, n, block))
    total_tiles = len(tiles_i) * len(tiles_j)
    done = 0

    for oi in tiles_i:
        src = list(range(oi, min(oi + block, n)))
        for dj in tiles_j:
            dst = list(range(dj, min(dj + block, n)))

            # Build compact coord list for this tile (src ∪ dst)
            uniq = sorted(set(src + dst))
            remap = {old: i for i, old in enumerate(uniq)}
            coords = ";".join(f"{lons[k]},{lats[k]}" for k in uniq)

            # sources/destinations are indices into the local coords list
            src_param = ";".join(str(remap[i]) for i in src)
            dst_param = ";".join(str(remap[j]) for j in dst)

            url = (
                f"{base_url.rstrip('/')}/table/v1/driving/{coords}"
                f"?annotations=duration,distance&sources={src_param}&destinations={dst_param}"
            )

            data = _osrm_get(url, timeout=timeout, retries=5, backoff=0.5)

            durs = data.get("durations")
            dists = data.get("distances")
            if durs is None or dists is None:
                raise RuntimeError(
                    f"OSRM /table returned no durations/distances for tile (oi={oi}, dj={dj})"
                )

            # Fill global matrices; penalize unreachable pairs heavily
            for si, a in enumerate(src):
                for di, b in enumerate(dst):
                    dur_s = durs[si][di]
                    dis_m = dists[si][di]
                    INF_MIN = 1e9  # huge minutes to make unreachable edges undesirable
                    INF_KM = 1e9
                    for si, a in enumerate(src):
                        for di, b in enumerate(dst):
                            dur_s = durs[si][di]
                            dis_m = dists[si][di]
                            if a == b:
                                dur_min[a][b] = 0.0
                                dist_km[a][b] = 0.0
                                continue
                            if dur_s is None or dis_m is None:
                                # unreachable → punish heavily
                                dur_min[a][b] = INF_MIN
                                dist_km[a][b] = INF_KM
                            else:
                                mm = dur_s / 60.0
                                km = dis_m / 1000.0
                                # guard against accidental zeros off-diagonal
                                if mm <= 0.0: mm = 1e-6
                                if km <= 0.0: km = 1e-6
                                dur_min[a][b] = mm
                                dist_km[a][b] = km

            done += 1
            if progress_cb:
                try:
                    progress_cb(done, total_tiles)
                except Exception:
                    pass
            if sleep > 0:
                time.sleep(sleep)

    return Matrix(distances_km=dist_km, durations_min=dur_min)

def osrm_leg_km_min(a_lat, a_lon, b_lat, b_lon, base_url="http://localhost:5000"):
    url = f"{base_url.rstrip('/')}/route/v1/driving/{a_lon},{a_lat};{b_lon},{b_lat}?overview=false"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    route = data["routes"][0]
    km = route["distance"] / 1000.0
    minutes = route["duration"] / 60.0
    return km, minutes

def _osrm_get(url, timeout=120, retries=5, backoff=0.5):
    import time as _t
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            # Treat 429/5xx as retryable
            if r.status_code in (429,) or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"{r.status_code}: {r.text}", response=r)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == retries - 1:
                raise
            _t.sleep(backoff * (2 ** i))

def kmeans_lloyd(points, k, iters=50, seed=42):
    import random
    random.seed(seed)
    centers = [points[i] for i in random.sample(range(len(points)), k)]
    assign = [0]*len(points)
    for _ in range(iters):
        for i,p in enumerate(points):
            best = 0; bestd = 1e18
            for ci,c in enumerate(centers):
                d = (p[0]-c[0])**2+(p[1]-c[1])**2
                if d<bestd: bestd=d; best=ci
            assign[i]=best
        new=centers[:]
        for ci in range(k):
            mem=[points[i] for i,a in enumerate(assign) if a==ci]
            if mem: new[ci]=(sum(x for x,_ in mem)/len(mem), sum(y for _,y in mem)/len(mem))
        if new==centers: break
        centers=new
    return assign

def coalesce_nearby(
    df: pd.DataFrame,
    threshold_m: float = 1.0,
    service_mode: str = "sum",   # "sum" or "one"
    name_mode: str = "first",    # "first" or "concat"
    per_stop_min_default: float = 2.0,
) -> pd.DataFrame:
    """
    Merge rows whose coordinates are within threshold_m meters.
    Returns a new DataFrame with lat/lon/id/name/service_min (and any other columns kept from the representative).

    service_mode:
      - "sum": service_min is sum of all members (each original stop counts)
      - "one": service_min is taken from the first row (or per_stop_min_default if missing)

    name_mode:
      - "first": keep the first row's name/id
      - "concat": join names/ids with " + "
    """
    if df.empty:
        return df.copy()

    # Ensure columns exist
    out = df.copy()
    if "service_min" not in out.columns:
        out["service_min"] = per_stop_min_default
    if "id" not in out.columns:
        out["id"] = [f"S{i}" for i in range(len(out))]
    if "name" not in out.columns:
        out["name"] = out["id"]

    # Build spatial hash (grid) + union-find for clustering
    import math

    def deg_per_m_lon(lat_deg: float) -> float:
        # degrees longitude per meter at a given latitude
        lat_rad = math.radians(lat_deg)
        deg_per_m_lat = 1.0 / 111320.0
        return deg_per_m_lat / max(1e-9, math.cos(lat_rad))

    # Grid size in degrees
    # (we compute per-point lon cell size using its latitude)
    deg_per_m_lat = 1.0 / 111320.0
    lat_cell = deg_per_m_lat * threshold_m

    # Union-Find
    parent = list(range(len(out)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Build buckets: key = (lat_key, lon_key_approx)
    # We’ll also check neighbor cells to avoid boundary misses.
    buckets = {}
    lats = out["lat"].to_numpy()
    lons = out["lon"].to_numpy()

    for i, (la, lo) in enumerate(zip(lats, lons)):
        lon_cell = deg_per_m_lon(la) * threshold_m
        key_lat = int(round(la / max(1e-12, lat_cell)))
        key_lon = int(round(lo / max(1e-12, lon_cell)))
        buckets.setdefault((key_lat, key_lon), []).append(i)

    # Check neighbors and union points within threshold by actual haversine
    for (klat, klon), idxs in list(buckets.items()):
        # candidates from 3x3 neighborhood
        neigh = []
        for dlat in (-1, 0, 1):
            for dlon in (-1, 0, 1):
                neigh.extend(buckets.get((klat + dlat, klon + dlon), []))
        # de-dup
        neigh = list(set(neigh))
        for i in idxs:
            for j in neigh:
                if i >= j:
                    continue
                d = haversine_km(lats[i], lons[i], lats[j], lons[j]) * 1000.0  # meters
                if d <= threshold_m + 1e-9:
                    union(i, j)

    # Collect clusters
    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(len(out)):
        groups[find(i)].append(i)

    # Build merged rows
    rows = []
    for root, idxs in groups.items():
        subset = out.iloc[idxs]
        # Representative coordinates: average
        lat_mean = float(subset["lat"].mean())
        lon_mean = float(subset["lon"].mean())

        # ID/Name
        if name_mode == "concat":
            id_val = " + ".join(map(str, subset["id"].tolist()))
            name_val = " + ".join(map(str, subset["name"].tolist()))
        else:
            id_val = str(subset.iloc[0]["id"])
            name_val = str(subset.iloc[0]["name"])

        # Service
        if service_mode == "sum":
            svc = float(pd.to_numeric(subset["service_min"], errors="coerce").fillna(per_stop_min_default).sum())
        else:  # "one"
            first = subset.iloc[0]
            svc = float(pd.to_numeric(first.get("service_min", per_stop_min_default), errors="coerce")
                        .fillna(per_stop_min_default))

        # Keep other columns from the first representative (optional)
        row = dict(subset.iloc[0])
        row.update({
            "id": id_val,
            "name": name_val,
            "lat": round(lat_mean, 7),
            "lon": round(lon_mean, 7),
            "service_min": max(0.0, svc),
            "coalesced_count": len(subset),
        })
        rows.append(row)

    merged = pd.DataFrame(rows)
    # Ensure core columns order first
    core_cols = [c for c in ["id", "name", "lat", "lon", "service_min", "coalesced_count"] if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in core_cols]
    merged = merged[core_cols + other_cols]
    return merged

def nearest_neighbor_route(idx_list,matrix):
    un=set(idx_list[1:]); route=[idx_list[0]]; cur=idx_list[0]
    while un:
        nxt=min(un,key=lambda j: matrix[cur][j]); route.append(nxt); un.remove(nxt); cur=nxt
    route.append(idx_list[0]); return route

def two_opt(route,matrix):
    def cost(rt): return sum(matrix[rt[i]][rt[i+1]] for i in range(len(rt)-1))
    best=route[:]; improved=True
    while improved:
        improved=False
        for i in range(1,len(best)-2):
            for k in range(i+1,len(best)-1):
                new=best[:i]+best[i:k+1][::-1]+best[k+1:]
                if cost(new)+1e-9<cost(best):
                    best=new; improved=True
    return best

def solve_vrp_ortools(stops,vehicles,matrix,equality_basis,per_stop_min,max_route_limit,first_solution,metaheuristic,time_limit_s):
    if not ORTOOLS_AVAILABLE: raise RuntimeError("OR-Tools not installed")
    n=len(stops); depot=0
    manager=pywrapcp.RoutingIndexManager(n,vehicles,depot)
    routing=pywrapcp.RoutingModel(manager)
    cost_matrix=matrix.durations_min if equality_basis=="time" else matrix.distances_km
    service=[(s.service_min if s.service_min is not None else per_stop_min) for s in stops] if equality_basis=="time" else [0.0 for _ in stops]
    def cb(fi,ti):
        i=manager.IndexToNode(fi); j=manager.IndexToNode(ti)
        base=cost_matrix[i][j]; add=service[i] if i!=depot and equality_basis=="time" else 0.0
        return int((base+add)*1000)
    cb_index=routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(cb_index)
    routing.AddDimension(cb_index,0,int(1e12),True,"measure")
    dim=routing.GetDimensionOrDie("measure")
    dim.SetGlobalSpanCostCoefficient(100)
    if max_route_limit is not None:
        cap = int(max_route_limit * 1000)
        for v in range(vehicles):
            end_cumul = dim.CumulVar(routing.End(v))
            # Either of these is fine:
            end_cumul.SetRange(0, cap)  # preferred
            # end_cumul.SetMax(cap)
    from ortools.constraint_solver import routing_enums_pb2
    fs=getattr(routing_enums_pb2.FirstSolutionStrategy,first_solution)
    mh=getattr(routing_enums_pb2.LocalSearchMetaheuristic,metaheuristic)
    params=pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy=fs; params.local_search_metaheuristic=mh; params.time_limit.FromSeconds(time_limit_s)
    sol=routing.SolveWithParameters(params)
    if sol is None: raise RuntimeError("No solution found")
    routes=[]
    for v in range(vehicles):
        idx=routing.Start(v); seq=[]; tot=0.0
        while not routing.IsEnd(idx):
            node=manager.IndexToNode(idx)
            nxt=sol.Value(routing.NextVar(idx)); nxt_node=manager.IndexToNode(nxt) if not routing.IsEnd(nxt) else None
            seq.append(node)
            if nxt_node is not None:
                c=cost_matrix[node][nxt_node]; add=service[node] if node!=0 and equality_basis=="time" else 0.0
                tot+=c+add
            idx=nxt
        seq.append(manager.IndexToNode(idx))
        routes.append({"vehicle":v,"sequence":seq,"total_measure":tot})
    return {"routes":routes,"equality_basis":equality_basis}

def solve_vrp_cluster2opt(stops,vehicles,matrix,equality_basis,per_stop_min):
    depot=0; points=[(s.lat,s.lon) for s in stops[1:]]; assigns=kmeans_lloyd(points,vehicles)
    clusters=[[] for _ in range(vehicles)]
    for i,a in enumerate(assigns,start=1): clusters[a].append(i)
    m = [ [ (matrix.durations_min if equality_basis=="time" else matrix.distances_km)[i][j] for j in range(len(stops))] for i in range(len(stops)) ]
    service=[(s.service_min if s.service_min is not None else per_stop_min) for s in stops] if equality_basis=="time" else [0.0 for _ in stops]
    routes=[]
    for v,cl in enumerate(clusters):
        if not cl: routes.append({"vehicle":v,"sequence":[depot,depot],"total_measure":0.0}); continue
        nn=nearest_neighbor_route([depot]+cl,m); opt=two_opt(nn,m)
        if opt[-1]!=depot: opt.append(depot)
        tot=0.0
        for i in range(len(opt)-1):
            a,b=opt[i],opt[i+1]; add=service[a] if a!=depot and equality_basis=="time" else 0.0
            tot+=m[a][b]+add
        routes.append({"vehicle":v,"sequence":opt,"total_measure":tot})
    return {"routes":routes,"equality_basis":equality_basis}

def load_stops(csv_path,per_stop_min,depot_lat,depot_lon):
    df=pd.read_csv(csv_path)
    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("CSV must have lat/lon columns")
    if "id" not in df: df["id"]=[f"S{i}" for i in range(len(df))]
    if "name" not in df: df["name"]=df["id"]
    if "service_min" not in df: df["service_min"]=per_stop_min
    if depot_lat is not None and depot_lon is not None:
        depot=pd.DataFrame([{"id":"DEPOT","name":"Depot","lat":depot_lat,"lon":depot_lon,"service_min":0.0}])
        df=pd.concat([depot,df],ignore_index=True)
    stops=[]
    for i,row in df.iterrows():
        stops.append(Stop(i,str(row["id"]),str(row["name"]),float(row["lat"]),float(row["lon"]),float(row.get("service_min",per_stop_min))))
    return stops

def export_outputs(stops,matrix,routes,equality_basis,export_csv,export_xlsx,per_stop_min,use_osrm=False,osrm_base="http://localhost:5000"):
    import pandas as pd
    rec=[]
    for r in routes["routes"]:
        seq=r["sequence"]; v=r["vehicle"]; time_acc=0.0
        for order,(a,b) in enumerate(zip(seq[:-1],seq[1:]),start=1):
            if use_osrm:
                lk,lm = osrm_leg_km_min(stops[a].lat,stops[a].lon,stops[b].lat,stops[b].lon,osrm_base)
            else:
                lk=matrix.distances_km[a][b]; lm=matrix.durations_min[a][b]
            arrival=time_acc; service=(stops[a].service_min if (a!=0 and equality_basis=="time") else 0.0); depart=arrival+service
            time_acc=depart+lm
            rec.append({"route_id":v,"order":order,"from_id":stops[a].id,"from_name":stops[a].name,"from_lat":stops[a].lat,"from_lon":stops[a].lon,
                        "to_id":stops[b].id,"to_name":stops[b].name,"to_lat":stops[b].lat,"to_lon":stops[b].lon,"leg_km":round(lk,3),"leg_min":round(lm,1),
                        "arrive_min":round(arrival,1),"service_min":round(service,1),"depart_min":round(depart,1),"cum_min_to_next":round(time_acc,1)})
    df_routes=pd.DataFrame.from_records(rec)
    if not df_routes.empty:
        summary=df_routes.groupby("route_id").agg(total_km=("leg_km","sum"),drive_min=("leg_min","sum"),service_min=("service_min","sum"),legs=("order","count")).reset_index()
        summary["total_min"]=(summary["drive_min"]+summary["service_min"]).round(1); summary["stops"]=summary["legs"]
        summary=summary[["route_id","stops","total_km","drive_min","service_min","total_min"]]
        summary["total_km"]=summary["total_km"].round(3); summary["drive_min"]=summary["drive_min"].round(1); summary["service_min"]=summary["service_min"].round(1)
    else:
        summary=pd.DataFrame(columns=["route_id","stops","total_km","drive_min","service_min","total_min"])
    if export_csv: df_routes.to_csv(export_csv,index=False)
    if export_xlsx:
        with pd.ExcelWriter(export_xlsx,engine="openpyxl") as w:
            df_routes.to_excel(w,sheet_name="stops",index=False); summary.to_excel(w,sheet_name="summary",index=False)
    return df_routes,summary
