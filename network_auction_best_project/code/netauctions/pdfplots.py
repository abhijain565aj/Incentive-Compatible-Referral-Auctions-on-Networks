from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

COLORS = [(0.12,0.47,0.71),(0.84,0.15,0.16),(0.17,0.63,0.17),(0.58,0.40,0.74),(1.0,0.5,0.05),(0.09,0.75,0.81)]

def esc(s: str) -> str:
    return s.replace('\\','\\\\').replace('(','\\(').replace(')','\\)')

def text_cmd(x: float, y: float, txt: str, size: int = 8) -> str:
    return 'BT /F1 %d Tf 1 0 0 1 %g %g Tm (%s) Tj ET' % (size, x, y, esc(txt))

def pdf_plot(path: Path, series: Dict[str, List[Tuple[float,float]]], xlabel: str, ylabel: str, title: str) -> None:
    W,H=720,440; ml,mr,mt,mb=75,170,55,70
    xs=[x for pts in series.values() for x,_ in pts]; ys=[y for pts in series.values() for _,y in pts]
    if not xs or not ys: return
    xmin,xmax=min(xs),max(xs); ymin,ymax=min(0,min(ys)),max(ys)
    if xmax==xmin: xmax+=1
    if ymax==ymin: ymax+=1
    pad=0.07*(ymax-ymin); ymin-=pad; ymax+=pad
    def sx(x): return ml+(x-xmin)*(W-ml-mr)/(xmax-xmin)
    def sy(y): return mb+(y-ymin)*(H-mt-mb)/(ymax-ymin)
    c=[]
    c.append('1 1 1 rg 0 0 %d %d re f' % (W,H))
    c.append(text_cmd(W/2-210,H-28,title[:80],14))
    # grid and axes
    for t in range(6):
        yv=ymin+t*(ymax-ymin)/5; yy=sy(yv)
        c.append('0.9 0.9 0.9 RG 0.5 w %g %g m %g %g l S' % (ml,yy,W-mr,yy))
        c.append(text_cmd(18, yy-3, '%.1f' % yv, 8))
        xv=xmin+t*(xmax-xmin)/5; xx=sx(xv)
        c.append(text_cmd(xx-8, 45, '%.0f' % xv, 8))
    c.append('0 0 0 RG 1.2 w %g %g m %g %g l S %g %g m %g %g l S' % (ml,mb,W-mr,mb,ml,mb,ml,H-mt))
    c.append(text_cmd((ml+W-mr)/2-55,20,xlabel,10))
    c.append(text_cmd(8,H/2,ylabel[:34],10))
    for idx,(name,pts) in enumerate(series.items()):
        color=COLORS[idx%len(COLORS)]; pts=sorted(pts)
        c.append('%g %g %g RG 1.8 w' % color)
        if pts:
            c.append('%g %g m' % (sx(pts[0][0]), sy(pts[0][1])))
            for x,y in pts[1:]: c.append('%g %g l' % (sx(x),sy(y)))
            c.append('S')
            for x,y in pts:
                c.append('%g %g %g rg %g %g 6 6 re f' % (*color, sx(x)-3, sy(y)-3))
        ly=H-mt-20-idx*17
        c.append('%g %g %g RG 1.8 w %g %g m %g %g l S' % (*color, W-mr+18, ly, W-mr+45, ly))
        c.append(text_cmd(W-mr+50, ly-3, name[:26], 8))
    stream='\n'.join(c).encode('latin-1','replace')
    objs=[]
    objs.append(b'<< /Type /Catalog /Pages 2 0 R >>')
    objs.append(b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>')
    objs.append(('<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>' % (W,H)).encode())
    objs.append(('<< /Length %d >>\nstream\n' % len(stream)).encode()+stream+b'\nendstream')
    objs.append(b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>')
    out=b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n'; offsets=[]
    for i,obj in enumerate(objs,1):
        offsets.append(len(out)); out += ('%d 0 obj\n' % i).encode()+obj+b'\nendobj\n'
    xref=len(out); out += ('xref\n0 %d\n0000000000 65535 f \n' % (len(objs)+1)).encode()
    for off in offsets: out += ('%010d 00000 n \n' % off).encode()
    out += ('trailer << /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n' % (len(objs)+1,xref)).encode()
    path.write_bytes(out)

def make_all(summary_csv: Path, out_dir: Path) -> None:
    rows=list(csv.DictReader(summary_csv.open()))
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in ['revenue_mean','net_welfare_mean']:
        sub=[r for r in rows if r['scenario']=='single_item_cost' and r['topology']=='random']
        series={m:[(float(r['n']),float(r[metric])) for r in sub if r['mechanism']==m] for m in sorted(set(r['mechanism'] for r in sub))}
        pdf_plot(out_dir/f'single_item_cost_{metric}.pdf', series, 'number of buyers', metric.replace('_',' '), f'Single item with participation costs: {metric}')
    sub=[r for r in rows if r['scenario']=='sybil']
    series={m:[(float(r['param']),float(r['revenue_mean'])) for r in sub if r['mechanism']==m] for m in sorted(set(r['mechanism'] for r in sub))}
    pdf_plot(out_dir/'sybil_revenue.pdf', series, 'inserted sybil identities', 'seller revenue mean', 'Sybil stress test on a path')
    sub=[r for r in rows if r['scenario']=='multi_item']
    for metric in ['revenue_mean','welfare_mean']:
        series={m:[(float(r['n']),float(r[metric])) for r in sub if r['mechanism']==m] for m in sorted(set(r['mechanism'] for r in sub))}
        pdf_plot(out_dir/f'multi_item_{metric}.pdf', series, 'number of buyers', metric.replace('_',' '), f'Multiple/different items: {metric}')

if __name__=='__main__':
    make_all(Path('../results/summary.csv'), Path('../report/figures'))
