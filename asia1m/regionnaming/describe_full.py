"""Full multi-country region description with admin hierarchy roll-up -> regions_all.csv

Roll-up: if a region covers ~all of a parent (prefecture/province) emit the parent name;
if it covers all-but-one/two children emit "{parent} minus {child}"; else list children.
"""
import sys, json, csv, time, os, re
from collections import defaultdict, Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, scipy.ndimage as ndi
SCR = HERE

FULL = 0.85       # >= this coverage of a parent -> just name the parent
MISSING = 0.18    # a child below this is considered "not included"
COVERED_MIN = 0.60  # a child at/above this is "included" (for the minus pattern)
PRIMARY = 0.20    # a unit >= this share of the region is a "lead" name; smaller ones go in "(also ...)"

DIR_ADJ = {(0,-1):'northern',(0,1):'southern',(1,0):'eastern',(-1,0):'western',
           (1,-1):'northeastern',(-1,-1):'northwestern',(1,1):'southeastern',(-1,1):'southwestern',(0,0):'central'}
def direction(dx,dy,t=0.30):
    sx=1 if dx>t else(-1 if dx<-t else 0); sy=1 if dy>t else(-1 if dy<-t else 0)
    return DIR_ADJ[(sx,sy)]
def size_word(c):
    return 'half' if c>=0.33 else('part' if c>=0.15 else 'small part')

def load(pfx):
    return np.load(f"{SCR}/{pfx}.npy"), json.load(open(f"{SCR}/{pfx}_names.json",encoding="utf-8"))

def geom_stats(adm):
    maxa=int(adm.max())
    area=np.bincount(adm.ravel(),minlength=maxa+1)
    sl=ndi.find_objects(adm,max_label=maxa)
    cx=np.zeros(maxa+1);cy=np.zeros(maxa+1);hw=np.ones(maxa+1);hh=np.ones(maxa+1)
    for i,s in enumerate(sl,1):
        if s is None: continue
        ys,xs=s
        cx[i]=(xs.start+xs.stop-1)/2; cy[i]=(ys.start+ys.stop-1)/2
        hw[i]=max((xs.stop-1-xs.start)/2,1); hh[i]=max((ys.stop-1-ys.start)/2,1)
    return area,cx,cy,hw,hh,maxa

def crosstab(region_id, adm, maxa):
    rid_flat=region_id.ravel(); aid_flat=adm.ravel()
    sel=np.flatnonzero(rid_flat>0)
    r=rid_flat[sel].astype(np.int64); a=aid_flat[sel].astype(np.int64)
    rows,cols=np.divmod(sel, region_id.shape[1])
    M=maxa+1; key=r*M+a
    order=np.argsort(key); key=key[order]
    bnd=np.flatnonzero(np.diff(key)); starts=np.r_[0,bnd+1]; ends=np.r_[bnd+1,len(key)]
    cols_s=cols[order].astype(np.float64); rows_s=rows[order].astype(np.float64)
    out={}
    for st,en in zip(starts,ends):
        k=key[st]; rid=int(k//M); aid=int(k%M); cnt=en-st
        out.setdefault(rid,[]).append((aid,int(cnt),cols_s[st:en].mean(),rows_s[st:en].mean()))
    return out

def clean(x):
    return re.sub(r'\s*\[\d+\]', '', x) if x else x

def label_unit(rec):
    n=clean(rec.get('name')); loc=clean(rec.get('local'))
    if not n and loc: return loc
    if not n: return '?'
    if loc and loc!=n: return f"{n} ({loc})"
    return n

def phrase(name,cov,pcx,pcy,acx,acy,ahw,ahh):
    if cov>=0.82: return name
    if cov>=0.55: return f"most of {name}"
    d=direction((pcx-acx)/ahw,(pcy-acy)/ahh); sw=size_word(cov)
    if d=='central': return f"central {name}" if sw=='half' else f"{sw} of central {name}"
    return f"{d} {sw} of {name}"

def build_hierarchy(names, area):
    """Returns dicts for parent nodes. Leaf node = ('L', gid); parent node = key tuple."""
    node_total=defaultdict(int)         # key -> total area
    children=defaultdict(set)           # key -> set of child nodes
    node_disp={}                        # key -> display string
    leaf_adm1={}; leaf_adm2={}          # gid -> parent key or None
    for gid in range(1, len(names)):
        rec=names[gid]
        if rec is None: continue
        country=rec.get('country'); prov=rec.get('prov'); dist=rec.get('dist')
        a1=(country,prov) if prov else None
        a2=(country,prov,dist) if dist else None
        leaf_adm1[gid]=a1; leaf_adm2[gid]=a2
        if a1:
            node_total[a1]+=area[gid]; node_disp[a1]=clean(prov)
            children[a1].add(a2 if a2 else ('L',gid))
        if a2:
            node_total[a2]+=area[gid]; node_disp[a2]=label_unit({'name':dist,'local':rec.get('dist_local')})
            children[a2].add(('L',gid))
    return dict(node_total=node_total, children=children, node_disp=node_disp,
                leaf_adm1=leaf_adm1, leaf_adm2=leaf_adm2)

def describe_region(parts, names, area, cx, cy, hw, hh, H):
    covered_leaf={}; cent={}
    for aid,c,pcx,pcy in parts:
        if aid==0: continue
        covered_leaf[aid]=c; cent[aid]=(pcx,pcy)
    region_land=sum(covered_leaf.values())
    if not region_land:
        return "", '?', []
    # node coverage
    covamt=defaultdict(int)
    for gid,c in covered_leaf.items():
        a1=H['leaf_adm1'][gid]; a2=H['leaf_adm2'][gid]
        if a1: covamt[a1]+=c
        if a2: covamt[a2]+=c
    nt=H['node_total']
    def is_leaf(n): return isinstance(n,tuple) and len(n)==2 and n[0]=='L'
    def n_area(n): return area[n[1]] if is_leaf(n) else nt[n]
    def n_amt(n):  return covered_leaf.get(n[1],0) if is_leaf(n) else covamt.get(n,0)
    def n_cov(n):  return n_amt(n)/max(n_area(n),1)
    def n_disp(n): return label_unit(names[n[1]]) if is_leaf(n) else H['node_disp'][n]

    def items_for(node):
        """Return a flat list of (text, bare_name, amt). Roll-ups/minus collapse to one item;
        minor whole-parents collapse to their name; only major partial parents expand to children."""
        if is_leaf(node):
            gid=node[1]; rec=names[gid]; pcx,pcy=cent[gid]
            lab=label_unit(rec)
            return [(phrase(lab, n_cov(node), pcx,pcy, cx[gid],cy[gid],hw[gid],hh[gid]), lab, n_amt(node))]
        cov=n_cov(node); amt=n_amt(node); disp=n_disp(node)
        if cov>=FULL:
            return [(disp, disp, amt)]
        ch_all=H['children'][node]
        touched=[c for c in ch_all if n_amt(c)>0]
        missing=[c for c in ch_all if n_cov(c)<MISSING]
        coveredf=[c for c in touched if n_cov(c)>=COVERED_MIN]
        if (len(ch_all)>=3 and 1<=len(missing)<=2 and len(coveredf)>=2
                and set(touched)==set(coveredf) and len(coveredf)==len(ch_all)-len(missing)):
            return [(f"{disp} minus {' and '.join(n_disp(m) for m in missing)}", disp, amt)]
        if amt/region_land < PRIMARY:            # minor parent -> collapse to its own name
            return [(("most of "+disp) if cov>=0.55 else ("part of "+disp), disp, amt)]
        out=[]                                   # major partial parent -> expand children
        for c in touched:
            if n_amt(c)/region_land<0.04 and n_cov(c)<0.12: continue
            out+=items_for(c)
        return out

    # top nodes: adm1 parents touched, plus parentless leaves
    top=set()
    for gid in covered_leaf:
        a1=H['leaf_adm1'][gid]
        top.add(a1 if a1 else ('L',gid))
    items=[]
    for n in top: items+=items_for(n)
    items.sort(key=lambda it:-it[2])
    prim=[]; sec=[]
    for i,(text,bare,amt) in enumerate(items):
        if i==0 or amt/region_land>=PRIMARY: prim.append(text)
        else: sec.append(bare)
    desc="; ".join(prim[:8])
    more = len(sec) > 12
    sec = sec[:12]
    if sec:
        desc=(desc+" " if desc else "")+"(also "+", ".join(sec)+("…" if more else "")+")"
    # country + provinces
    cnt_country=Counter(); prov_amt=defaultdict(int)
    for gid,c in covered_leaf.items():
        rec=names[gid]; cnt_country[rec.get('country')]+=c
        if rec.get('prov'): prov_amt[rec['prov']]+=c
    country=cnt_country.most_common(1)[0][0]
    provs=[p for p,_ in sorted(prov_amt.items(), key=lambda x:-x[1])]
    return desc, country, provs

def describe_township(parts, names, area):
    groups={}
    landtot=sum(c for a,c,_,_ in parts if a!=0)
    for aid,c,_,_ in parts:
        if aid==0: continue
        rec=names[aid]; cov=c/max(area[aid],1)
        if c/max(landtot,1)<0.03 and cov<0.15: continue
        city=rec.get('city') or ''; county=rec.get('county') or ''
        key=county if (not city or city==county) else f"{city}{county}"
        g=groups.setdefault(key,[0,rec.get('prov'),[]])
        g[0]+=c; g[2].append((rec.get('town'),cov,c))
    if not groups: return None,[]
    parts_str=[]; provs=[]
    for key,(tc,prov,towns) in sorted(groups.items(),key=lambda x:-x[1][0])[:6]:
        towns.sort(key=lambda t:-t[2])
        tl=[tn if cov>=0.5 else f"{tn}(部分)" for tn,cov,c in towns[:7]]
        parts_str.append(f"{key}: {'、'.join(tl)}")
        if prov and prov not in provs: provs.append(prov)
    return "; ".join(parts_str), provs

def main():
    t0=time.time()
    region_id=np.load(f"{SCR}/region_id.npy")
    regs={int(r['rid']):r for r in csv.DictReader(open(f"{SCR}/regions_prelim.csv"))}
    adm,names=load("adm_all")
    area,cx,cy,hw,hh,maxa=geom_stats(adm)
    H=build_hierarchy(names, area)
    print(f"[{time.time()-t0:.0f}s] adm stats + hierarchy")
    ct=crosstab(region_id,adm,maxa)
    print(f"[{time.time()-t0:.0f}s] county crosstab")
    twn,tnames=load("adm_twn")
    tarea=np.bincount(twn.ravel(),minlength=int(twn.max())+1)
    tct=crosstab(region_id,twn,int(twn.max()))
    print(f"[{time.time()-t0:.0f}s] township crosstab")

    rows=[]
    for rid,r in regs.items():
        parts=ct.get(rid,[])
        desc,country,provs=describe_region(parts,names,area,cx,cy,hw,hh,H)
        tot=sum(c for _,c,_,_ in parts); land=sum(c for a,c,_,_ in parts if a!=0)
        maxcov=max([c/max(area[a],1) for a,c,_,_ in parts if a!=0], default=0)
        if country=='China' and maxcov<0.30 and rid in tct:
            td,tprovs=describe_township(tct[rid],tnames,tarea)
            if td: desc=td; provs=tprovs or provs
        flags=[]
        if int(r['area_px'])<5: flags.append('tiny<5px')
        if land/max(tot,1)<0.5: flags.append('mostly-offshore/uncovered')
        if not desc: flags.append('no-admin')
        rows.append(dict(name='', rid=rid, color=r['color'], area_px=r['area_px'],
                         lon=r['lon'], lat=r['lat'], country=country,
                         province=", ".join(provs), admin_divisions=desc, flags=";".join(flags)))
    rows.sort(key=lambda x:(x['country'], -float(x['lat'])))
    with open(f"{SCR}/regions_all.csv","w",newline='',encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=['name','rid','color','area_px','lon','lat','country','province','admin_divisions','flags'])
        w.writeheader(); w.writerows(rows)
    print(f"[{time.time()-t0:.0f}s] wrote {len(rows)} regions")
    cc=Counter(r['country'] for r in rows)
    for k,v in cc.most_common(): print(f"  {k}: {v}")
    print("no-admin:",sum(1 for r in rows if 'no-admin' in r['flags']),
          " offshore:",sum(1 for r in rows if 'offshore' in r['flags']))

if __name__=="__main__": main()
