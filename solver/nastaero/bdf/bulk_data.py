"""Bulk data card dispatcher."""
from __future__ import annotations
from typing import List
from .model import BDFModel
from .cards.grid import GRID
from .cards.coord import CORD2R
from .cards.elements import CBAR, CROD, CQUAD4, CTRIA3
from .cards.properties import PBAR, PROD, PSHELL, PSOLID
from .cards.materials import MAT1
from .cards.loads import FORCE, MOMENT, GRAV, LoadCombination
from .cards.constraints import SPC, SPC1
from .cards.mass import CONM2
from .cards.eigrl import EIGRL
from .cards.rbe import RBE2, RBE3
from .cards.param import PARAM
from .cards.sets import SET1
from .cards.aero import (AERO, AEROS, CAERO1, PAERO1, SPLINE1, SPLINE2,
                         AESTAT, AESURF, AELIST, TRIM, FLFACT, MKAERO1)
from ..config import logger

def parse_bulk_card(fields: List[str], model: BDFModel) -> None:
    if not fields: return
    card_name = fields[0].strip().upper().rstrip("*")
    try:
        if card_name == "GRID":
            g = GRID.from_fields(fields); model.nodes[g.nid] = g
        elif card_name == "CORD2R":
            c = CORD2R.from_fields(fields); model.coords[c.cid] = c
        elif card_name == "CBAR":
            e = CBAR.from_fields(fields); model.elements[e.eid] = e
        elif card_name == "CROD":
            e = CROD.from_fields(fields); model.elements[e.eid] = e
        elif card_name == "CQUAD4":
            e = CQUAD4.from_fields(fields); model.elements[e.eid] = e
        elif card_name == "CTRIA3":
            e = CTRIA3.from_fields(fields); model.elements[e.eid] = e
        elif card_name == "PBAR":
            p = PBAR.from_fields(fields); model.properties[p.pid] = p
        elif card_name == "PROD":
            p = PROD.from_fields(fields); model.properties[p.pid] = p
        elif card_name == "PSHELL":
            p = PSHELL.from_fields(fields); model.properties[p.pid] = p
        elif card_name == "PSOLID":
            p = PSOLID.from_fields(fields); model.properties[p.pid] = p
        elif card_name == "MAT1":
            m = MAT1.from_fields(fields); model.materials[m.mid] = m
        elif card_name == "FORCE":
            l = FORCE.from_fields(fields); model.loads.setdefault(l.sid, []).append(l)
        elif card_name == "MOMENT":
            l = MOMENT.from_fields(fields); model.loads.setdefault(l.sid, []).append(l)
        elif card_name == "GRAV":
            l = GRAV.from_fields(fields); model.loads.setdefault(l.sid, []).append(l)
        elif card_name == "LOAD":
            lc = LoadCombination.from_fields(fields); model.load_combinations[lc.sid] = lc
        elif card_name == "SPC":
            s = SPC.from_fields(fields); model.spcs.setdefault(s.sid, []).append(s)
        elif card_name == "SPC1":
            s = SPC1.from_fields(fields); model.spcs.setdefault(s.sid, []).append(s)
        elif card_name == "CONM2":
            m = CONM2.from_fields(fields); model.masses[m.eid] = m
        elif card_name == "EIGRL":
            e = EIGRL.from_fields(fields); model.eigrls[e.sid] = e
        elif card_name == "RBE2":
            r = RBE2.from_fields(fields); model.rigids[r.eid] = r
        elif card_name == "RBE3":
            r = RBE3.from_fields(fields); model.rigids[r.eid] = r
        elif card_name == "SET1":
            s = SET1.from_fields(fields); model.sets[s.sid] = s
        elif card_name == "PARAM":
            name, value = PARAM.from_fields(fields); model.params[name] = value
        # Aerodynamic cards
        elif card_name == "AERO":
            model.aero = AERO.from_fields(fields)
        elif card_name == "AEROS":
            model.aeros = AEROS.from_fields(fields)
        elif card_name == "CAERO1":
            c = CAERO1.from_fields(fields); model.caero_panels[c.eid] = c
        elif card_name == "PAERO1":
            p = PAERO1.from_fields(fields); model.properties[p.pid] = p
        elif card_name == "SPLINE1":
            s = SPLINE1.from_fields(fields); model.splines[s.eid] = s
        elif card_name == "SPLINE2":
            s = SPLINE2.from_fields(fields); model.splines[s.eid] = s
        elif card_name == "AESTAT":
            a = AESTAT.from_fields(fields); model.aestats[a.id] = a
        elif card_name == "AESURF":
            a = AESURF.from_fields(fields); model.aesurfs[a.id] = a
        elif card_name == "AELIST":
            a = AELIST.from_fields(fields); model.aelists[a.sid] = a
        elif card_name == "TRIM":
            t = TRIM.from_fields(fields); model.trims[t.tid] = t
        elif card_name == "FLFACT":
            f = FLFACT.from_fields(fields); model.flfacts[f.sid] = f
        elif card_name == "MKAERO1":
            m = MKAERO1.from_fields(fields); model.mkaeros.append(m)
        else:
            logger.debug("Unsupported card: %s", card_name)
    except (IndexError, ValueError) as exc:
        logger.warning("Error parsing %s: %s", card_name, exc)
