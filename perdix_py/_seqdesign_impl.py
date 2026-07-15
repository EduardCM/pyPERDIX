from __future__ import annotations

from dataclasses import dataclass

from .data_prob import ProbType
from .data_geom import GeomType
from .data_mesh import MeshType
from .data_dna import DNAType, TopType, StrandType
from .mani import Mani_Go_Start_Base
from . import para


@dataclass
class RegionType:
    types: int = 0
    length: int = 0
    sta_base: int = -1
    cen_base: int = -1
    end_base: int = -1
    sta_pos: int = 0
    cen_pos: int = 0
    end_pos: int = 0


@dataclass
class ScafRegionType:
    length: int = 0
    strt_pos: int = 0
    end_pos: int = 0
    cntr_pos: int = 0
    strt_base: int = -1
    cntr_base: int = -1


def seqdesign_design(prob: ProbType, geom: GeomType, mesh: MeshType, dna: DNAType) -> None:
    supported_cut_methods = {"max", "seed", "opt"}
    if para.para_cut_stap_method not in supported_cut_methods:
        supported = ", ".join(sorted(supported_cut_methods))
        raise ValueError(
            f"Unsupported para_cut_stap_method: {para.para_cut_stap_method!r}. "
            f"Supported values: {supported}."
        )

    _build_dna_top(dna)
    _build_strand(dna)
    _make_noncir_stap_nick(mesh, dna)
    if para.para_cut_stap_method == "max":
        _break_staples_length(prob, mesh, dna)
    elif para.para_cut_stap_method in ("seed", "opt"):
        _break_staples_seeds(prob, mesh, dna)
    _make_nick_scaf(geom, mesh, dna)
    _make_short_scaf(mesh, dna)
    _rebuild_strand(dna)
    _order_staple(dna)
    if para.para_max_cut_scaf == 0:
        _print_14nt_region_simple(geom, mesh, dna)
    _assign_sequence(prob, dna)
    _round_top_positions(dna, digits=4)


def _round_top_positions(dna: DNAType, digits: int = 4) -> None:
    for t in dna.top:
        t.pos = tuple(round(float(v), digits) for v in t.pos)


def _print_14nt_region_simple(geom: GeomType, mesh: MeshType, dna: DNAType) -> None:
    n_14nt = 0
    n_4nt = 0

    for i, strand in enumerate(dna.strand):
        if strand.type1 == "scaf":
            strand.type2 = "vertex"
            continue

        strand.type2 = "edge  "
        region = _build_region_staple_1(dna, i)
        n_region = len(region)
        if n_region == 0:
            continue

        n_sec_14nt = 0
        n_sec_4nt = 0
        b_14nt = False
        b_4nt = False

        for reg in region:
            if reg.types == 1:
                strand.type2 = "vertex"

            if (reg.types == 1 and reg.length + 1 >= 14) or (reg.types == 2 and reg.length + 2 >= 14):
                if not b_14nt:
                    n_14nt += 1
                    b_14nt = True
                n_sec_14nt += 1

            if (reg.types == 1 and reg.length + 1 <= 4) or (reg.types == 2 and reg.length + 2 <= 4):
                if not b_4nt:
                    n_4nt += 1
                    b_4nt = True
                n_sec_4nt += 1

        strand.n_14nt = n_sec_14nt
        strand.n_4nt = n_sec_4nt

    dna.n_14nt = n_14nt
    dna.n_4nt = n_4nt


def _assign_sequence(prob: ProbType, dna: DNAType) -> None:
    mode = (para.para_scaf_seq or "").strip().lower()
    if mode == "":
        mode = "user"

    if mode == "m13":
        _set_scaffold_sequence_from_file(prob, dna, use_m13=True)
    elif mode == "user":
        _set_scaffold_sequence_from_file(prob, dna, use_m13=False)
    elif mode == "rand":
        _set_random_sequence(prob, dna)
    else:
        raise ValueError("Scaffold sequences are not assigned.")


def _set_scaffold_sequence_from_file(prob: ProbType, dna: DNAType, use_m13: bool) -> None:
    from .resource_utils import read_packaged_text

    seq = prob.scaf_seq
    wrap_len = None
    if not seq:
        if use_m13:
            if dna.n_base_scaf <= 7249:
                resource_name = "m13mp18_perdix.txt"
            else:
                resource_name = "lamda_perdix.txt"
            # Fortran wraps by M13 length even when using Lamda.
            wrap_len = 7249
            seq = read_packaged_text(resource_name)
            seq = "".join(ch for ch in seq if ch.isalpha()).upper()
            prob.scaf_seq = seq
        else:
            # user mode expects seq.txt-provided sequence
            raise ValueError("Scaffold sequence is empty; check seq.txt for user mode.")

    _apply_scaffold_sequence(prob, dna, seq, wrap_len=wrap_len)


def _set_random_sequence(prob: ProbType, dna: DNAType) -> None:
    import random

    bases = ["A", "T", "G", "C"]
    seq = "".join(random.choice(bases) for _ in range(dna.n_base_scaf))
    _apply_scaffold_sequence(prob, dna, seq)


def _apply_scaffold_sequence(prob: ProbType, dna: DNAType, seq: str, wrap_len: int | None = None) -> None:
    seq = seq.strip().upper()
    if len(seq) == 0:
        raise ValueError("Scaffold sequence is empty")
    # Fortran behavior wraps by M13 length when using the built-in scaffold.
    if wrap_len is None:
        wrap_len = len(seq)

    for i, strand in enumerate(dna.strand):
        if strand.type1 != "scaf":
            continue

        base = Mani_Go_Start_Base(dna, i)
        if base == -1:
            continue
        across = dna.top[base].across

        for offset in range(strand.n_base):
            if base == -1:
                break
            count = offset + para.para_set_start_scaf
            if count >= wrap_len:
                count = count % wrap_len
                if count == 0:
                    count = wrap_len

            dna.top[base].seq = seq[count - 1]
            if across != -1:
                dna.top[across].seq = _get_comp_sequence(dna.top[base].seq)

            base = dna.top[base].up
            if base != -1:
                across = dna.top[base].across


def _get_comp_sequence(base: str) -> str:
    b = base.upper()
    if b == "A":
        return "T"
    if b == "T":
        return "A"
    if b == "G":
        return "C"
    if b == "C":
        return "G"
    return "N"


def _build_dna_top(dna: DNAType) -> None:
    dna.n_top = dna.n_base_scaf + dna.n_base_stap
    dna.top = [TopType() for _ in range(dna.n_top)]

    for i in range(dna.n_base_scaf):
        dna.top[i].id = dna.base_scaf[i].id
        dna.top[i].node = dna.base_scaf[i].node
        dna.top[i].up = dna.base_scaf[i].up
        dna.top[i].dn = dna.base_scaf[i].dn
        dna.top[i].xover = dna.base_scaf[i].xover
        dna.top[i].across = -1 if dna.base_scaf[i].across == -1 else dna.base_scaf[i].across + dna.n_base_scaf
        dna.top[i].strand = -1
        dna.top[i].address = -1
        dna.top[i].b_14nt = False
        dna.top[i].seq = "N"
        dna.top[i].pos = dna.base_scaf[i].pos

    n_jump = dna.n_base_scaf
    for i in range(dna.n_base_stap):
        idx = i + n_jump
        dna.top[idx].id = dna.base_stap[i].id + n_jump
        dna.top[idx].node = dna.base_stap[i].node
        dna.top[idx].up = -1 if dna.base_stap[i].up == -1 else dna.base_stap[i].up + n_jump
        dna.top[idx].dn = -1 if dna.base_stap[i].dn == -1 else dna.base_stap[i].dn + n_jump
        dna.top[idx].xover = -1 if dna.base_stap[i].xover == -1 else dna.base_stap[i].xover + n_jump
        dna.top[idx].across = dna.base_stap[i].across
        dna.top[idx].strand = -1
        dna.top[idx].address = -1
        dna.top[idx].b_14nt = False
        dna.top[idx].seq = "T" if dna.base_stap[i].across == -1 else "N"
        dna.top[idx].pos = dna.base_stap[i].pos


def _build_strand(dna: DNAType) -> None:
    n_top = dna.n_top
    visited = [False] * n_top
    strands: list[tuple[list[int], bool]] = []

    for i in range(n_top):
        if visited[i]:
            continue

        cur = dna.top[i]
        start_id = cur.id

        while True:
            if cur.dn == -1:
                circular = False
                break
            if cur.dn == start_id:
                circular = True
                cur = dna.top[start_id]
                break
            cur = dna.top[cur.dn]
            if visited[cur.id]:
                raise ValueError("Reached a visited base while building strands.")

        base_ids: list[int] = []
        while True:
            base_ids.append(cur.id)
            dna.top[cur.id].strand = len(strands) + 1
            dna.top[cur.id].address = len(base_ids)
            dna.top[cur.id].b_14nt = False
            visited[cur.id] = True
            if (not circular and cur.up == -1) or (circular and cur.up == start_id):
                break
            cur = dna.top[cur.up]
            if visited[cur.id]:
                raise ValueError("Reached a visited base while building strands.")

        strands.append((base_ids, circular))

    dna.n_strand = len(strands)
    dna.n_scaf = 1
    dna.n_stap = dna.n_strand - 1
    dna.strand = []

    for idx, (base_ids, circular) in enumerate(strands):
        s = StrandType()
        s.n_base = len(base_ids)
        s.b_circular = circular
        s.base = list(base_ids)
        s.type1 = "scaf" if idx == 0 else "stap"
        dna.strand.append(s)


def _make_noncir_stap_nick(mesh: MeshType, dna: DNAType) -> None:
    for strand in dna.strand:
        if strand.type1 == "scaf":
            continue
        if not strand.b_circular:
            continue

        base = strand.base[0]
        for _ in range(strand.n_base):
            dn_base = dna.top[base].dn
            if dn_base != -1 and dna.top[base].xover != -1 and dna.top[dn_base].xover != -1:
                break
            base = dna.top[base].dn

        init_base = base
        count = 1
        max_count = 0
        start_base = base
        max_start_base = base
        for _ in range(strand.n_base):
            across = dna.top[base].across
            if across == -1:
                if max_count < count:
                    max_count = count
                    max_start_base = start_base
                count = 0
                start_base = base
            else:
                if dna.top[base].xover != -1 or dna.top[across].xover != -1:
                    if max_count < count:
                        max_count = count
                        max_start_base = start_base
                    count = 0
                    start_base = base
            count += 1
            base = dna.top[base].up

        max_count = max_count - 1
        if max_count > para.para_gap_xover_nick1 - 2:
            base = dna.top[max_start_base].id
            for _ in range(max_count // 2 + 1):
                base = dna.top[base].up
        else:
            base = init_base

        dn_base = dna.top[base].dn
        if dn_base == -1:
            continue
        dna.top[base].dn = -1
        dna.top[dn_base].up = -1

        if dna.top[base].xover == dna.top[dn_base].id and dna.top[dn_base].xover == dna.top[base].id:
            dna.n_xover_stap -= 1
            dna.n_sxover_stap += 1
            dna.top[base].xover = -1
            dna.top[dn_base].xover = -1


def _break_staples_seeds(prob: ProbType, mesh: MeshType, dna: DNAType) -> None:
    for i, strand in enumerate(dna.strand):
        if _skip_staple_cut_candidate(prob, dna, strand, unpaired_edge_mode=3):
            continue

        region = _build_region_staple(dna, i)
        n_region = len(region)
        if n_region == 0:
            continue
        region1 = [None] + region

        bgn_pos = 0
        pre_reg = 0
        b_cut = False
        b_14nt = False
        j = 0

        while True:
            j += 1
            if j == n_region + 1:
                break

            if j != n_region and region1[j].length < 5:
                continue

            length = region1[j].cen_pos - bgn_pos
            if j == n_region:
                length = strand.n_base - bgn_pos + 1

            if not b_cut and length >= para.para_min_cut_stap:
                b_cut = True

            if j != n_region and length <= para.para_max_cut_stap:
                if (
                    (not b_14nt and region1[j].length + 1 >= 14 and region1[j].types == 1)
                    or (not b_14nt and region1[j].length + 2 >= 14 and region1[j].types == 2)
                ):
                    b_14nt = True
                    continue

            if b_cut and b_14nt and length <= para.para_max_cut_stap:
                cen_base = region1[j].cen_base
                cen_pos = region1[j].cen_pos

                if strand.n_base - cen_pos < para.para_min_cut_stap:
                    continue

                _cut_staple_after_base(dna, cen_base)

                bgn_pos = cen_pos
                b_cut = False
                b_14nt = False
            elif b_cut and length > para.para_max_cut_stap:
                selection = _select_staple_cut_region(
                    prob,
                    region1,
                    start_idx=j,
                    stop_idx=pre_reg,
                    bgn_pos=bgn_pos,
                    strand_n_base=strand.n_base,
                    type_sensitive=False,
                    allow_extended=True,
                )
                if selection is None:
                    continue
                jj, pre_base, pre_pos, b_ext = selection

                if b_ext:
                    if j == n_region:
                        continue

                if pre_pos - bgn_pos < para.para_min_cut_stap:
                    continue

                up = dna.top[region1[jj].end_base].up
                if up == -1:
                    xover = -1
                    upxover = -1
                else:
                    xover = dna.top[up].xover
                    up_up = dna.top[up].up
                    upxover = dna.top[up_up].xover if up_up != -1 else -1

                if _can_cut_staple_short_xover(region1[jj], bgn_pos, strand.n_base, xover, upxover):
                    _cut_staple_short_xover(dna, up, xover)

                    pre_reg = jj
                    bgn_pos = region1[jj].end_pos + 1
                    b_cut = False
                    b_14nt = False
                    j = jj
                else:
                    _cut_staple_after_base(dna, pre_base)

                    pre_reg = jj
                    bgn_pos = pre_pos
                    b_cut = False
                    b_14nt = False
                    j = jj

            if strand.n_base - bgn_pos < para.para_max_cut_stap:
                break

            if j == n_region:
                final_length = strand.n_base - bgn_pos
                if final_length > para.para_max_cut_stap:
                    jj = j
                    cng_para = para.para_gap_xover_nick
                    while True:
                        if region1[jj].length >= cng_para * 2 + 2:
                            pre_base = region1[jj].cen_base
                            pre_pos = region1[jj].cen_pos
                            if _is_valid_staple_cut_position(pre_pos, bgn_pos, strand.n_base):
                                break
                        jj -= 1
                        if jj == pre_reg or jj == 0:
                            jj = j
                            cng_para -= 1
                            if cng_para == 0:
                                return

                    if dna.top[pre_base].up == -1:
                        continue
                    if (
                        final_length - (strand.n_base - pre_pos) < para.para_min_cut_stap
                        or strand.n_base - pre_pos < para.para_min_cut_stap
                    ):
                        continue

                    _cut_staple_after_base(dna, pre_base)


def _break_staples_length(prob: ProbType, mesh: MeshType, dna: DNAType) -> None:
    for i, strand in enumerate(dna.strand):
        if _skip_staple_cut_candidate(prob, dna, strand, unpaired_edge_mode=0):
            continue

        region = _build_region_staple(dna, i)
        n_region = len(region)
        if n_region == 0:
            continue
        region1 = [None] + region  # 1-based access

        bgn_pos = 0
        pre_region = 0
        b_cut = False

        for j in range(1, n_region + 1):
            cen_base = region1[j].cen_base
            cen_pos = region1[j].cen_pos
            length = cen_pos - bgn_pos

            if not b_cut and length >= para.para_min_cut_stap:
                b_cut = True

            if b_cut and length >= para.para_max_cut_stap:
                selection = _select_staple_cut_region(
                    prob,
                    region1,
                    start_idx=j - 1,
                    stop_idx=pre_region,
                    bgn_pos=bgn_pos,
                    strand_n_base=strand.n_base,
                    type_sensitive=True,
                    allow_extended=True,
                )
                if selection is None:
                    continue
                jj, pre_base, pre_pos, b_ext = selection

                if b_ext:
                    if j == n_region:
                        continue
                else:
                    if not _cut_staple_after_base(dna, pre_base):
                        continue


                    pre_region = jj
                    bgn_pos = pre_pos
                    b_cut = False

            if strand.n_base - bgn_pos <= para.para_max_cut_stap:
                break

            if j == n_region:
                final_length = strand.n_base - bgn_pos
                if final_length >= para.para_max_cut_stap:
                    jj = j
                    cng_para = para.para_gap_xover_nick
                    b_ext = False

                    while True:
                        if jj == 0:
                            return
                        if region1[jj].length >= para.para_gap_xover_nick * 2 + 2:
                            pre_base = region1[jj].cen_base
                            pre_pos = region1[jj].cen_pos
                            if _is_valid_staple_cut_position(pre_pos, bgn_pos, strand.n_base):
                                break
                        jj -= 1
                        if jj == pre_region or jj == 0:
                            jj = j - 1
                            cng_para -= 1
                            if cng_para == 0:
                                b_ext = True
                                pre_base = region1[jj].cen_base
                                pre_pos = region1[jj].cen_pos
                                break
                            _update_max_stap_gap_change(prob, cng_para)

                    if dna.top[pre_base].up == -1:
                        continue
                    if (
                        final_length - (strand.n_base - pre_pos) < para.para_min_cut_stap
                        or strand.n_base - pre_pos < para.para_min_cut_stap
                    ):
                        continue

                    _cut_staple_after_base(dna, pre_base)


def _skip_staple_cut_candidate(
    prob: ProbType,
    dna: DNAType,
    strand: StrandType,
    unpaired_edge_mode: int,
) -> bool:
    if strand.type1 == "scaf":
        return True
    if strand.n_base <= para.para_max_cut_stap:
        return True
    if not _strand_has_unpaired_partner(dna, strand):
        return False
    if unpaired_edge_mode in (0, 1) and prob.sel_edge_sec == 1 and strand.n_base < 80:
        return True
    if unpaired_edge_mode in (0, 3) and prob.sel_edge_sec == 3 and strand.n_base < 70:
        return True
    return False


def _strand_has_unpaired_partner(dna: DNAType, strand: StrandType) -> bool:
    return any(dna.top[base].across == -1 for base in strand.base)


def _is_valid_staple_cut_position(pre_pos: int, bgn_pos: int, strand_n_base: int) -> bool:
    return (
        pre_pos - bgn_pos >= para.para_min_cut_stap
        and pre_pos - bgn_pos <= para.para_max_cut_stap
        and strand_n_base - pre_pos >= para.para_min_cut_stap
    )


def _update_max_stap_gap_change(prob: ProbType, cng_para: int) -> None:
    delta = para.para_gap_xover_nick - cng_para
    if delta == 1 and prob.n_cng_max_stap == 0:
        prob.n_cng_max_stap = 1
    if delta == 2 and prob.n_cng_max_stap == 1:
        prob.n_cng_max_stap = 2
    if delta == 3 and prob.n_cng_max_stap == 2:
        prob.n_cng_max_stap = 3


def _select_staple_cut_region(
    prob: ProbType,
    regions: list[RegionType | None],
    start_idx: int,
    stop_idx: int,
    bgn_pos: int,
    strand_n_base: int,
    type_sensitive: bool,
    allow_extended: bool,
) -> tuple[int, int, int, bool] | None:
    jj = start_idx
    cng_para = para.para_gap_xover_nick

    while True:
        if jj == 0:
            return None
        region = regions[jj]
        if region is not None and _region_accepts_staple_cut(region, cng_para, type_sensitive):
            pre_base = region.cen_base
            pre_pos = region.cen_pos
            if _is_valid_staple_cut_position(pre_pos, bgn_pos, strand_n_base):
                return jj, pre_base, pre_pos, False

        jj -= 1
        if jj == stop_idx or jj == 0:
            jj = start_idx
            cng_para -= 1
            if cng_para == 0:
                if not allow_extended:
                    return None
                region = regions[jj]
                if region is None:
                    return None
                return jj, region.cen_base, region.cen_pos, True
            _update_max_stap_gap_change(prob, cng_para)


def _region_accepts_staple_cut(region: RegionType, cng_para: int, type_sensitive: bool) -> bool:
    if not type_sensitive:
        return region.length >= cng_para * 2 + 2
    return (
        (region.types == 1 and region.length >= cng_para * 2 + 3)
        or (region.types == 2 and region.length >= cng_para * 2 + 2)
    )


def _can_cut_staple_short_xover(
    region: RegionType,
    bgn_pos: int,
    strand_n_base: int,
    xover: int,
    upxover: int,
) -> bool:
    return (
        region.end_pos + 1 - bgn_pos >= para.para_min_cut_stap
        and region.end_pos + 1 - bgn_pos <= para.para_max_cut_stap
        and strand_n_base - region.end_pos + 1 >= para.para_min_cut_stap
        and region.length >= 12
        and xover != -1
        and upxover != -1
        and para.para_set_stap_sxover == "on"
    )


def _cut_staple_after_base(dna: DNAType, base: int) -> bool:
    dna.n_stap += 1
    return _break_top_link_after_base(dna, base)


def _cut_staple_short_xover(dna: DNAType, up: int, xover: int) -> None:
    dna.n_stap += 1
    if up != -1:
        if dna.top[up].up == xover:
            dna.top[up].up = -1
        if dna.top[up].dn == xover:
            dna.top[up].dn = -1

    if xover != -1:
        if dna.top[xover].up == up:
            dna.top[xover].up = -1
        if dna.top[xover].dn == up:
            dna.top[xover].dn = -1

    if up != -1:
        dna.top[up].xover = -1
    if xover != -1:
        dna.top[xover].xover = -1

    dna.n_xover_stap -= 1
    dna.n_sxover_stap += 1


def _build_region_staple(dna: DNAType, strand_index: int) -> list[RegionType]:
    strand = dna.strand[strand_index]
    base = Mani_Go_Start_Base(dna, strand_index)
    if base == -1:
        return []

    region: list[RegionType] = []
    b_region = False
    b_vertex = False

    for j in range(1, strand.n_base + 1):
        across = dna.top[base].across

        if across == -1:
            b_region = True
            b_vertex = True
            base = dna.top[base].up
            continue
        if (
            dna.top[base].dn == -1
            or dna.top[base].up == -1
            or dna.top[base].xover != -1
            or dna.top[across].xover != -1
        ):
            b_region = True
            base = dna.top[base].up
            continue

        if b_region:
            b_region = False
            reg = RegionType()
            reg.sta_base = base
            reg.length = 1
            reg.sta_pos = j
            reg.end_pos = j
            if b_vertex:
                reg.types = 1
                if region:
                    region[-1].types = 1
                b_vertex = False
            else:
                reg.types = 2
            region.append(reg)
        else:
            region[-1].length += 1
            region[-1].end_pos = j

        base = dna.top[base].up

    for reg in region:
        len_cen = (reg.length + 1) // 2 - 1
        if reg.types == 1:
            dn = dna.top[reg.sta_base].dn
            if dn != -1 and dna.top[dn].across != -1:
                len_cen -= 1
            else:
                len_cen += 1

        cen_base = reg.sta_base
        for _ in range(max(0, len_cen)):
            cen_base = dna.top[cen_base].up
        reg.cen_base = cen_base
        reg.cen_pos = reg.sta_pos + len_cen

        end_base = reg.sta_base
        for _ in range(reg.length - 1):
            end_base = dna.top[end_base].up
        reg.end_base = end_base
        reg.end_pos = reg.sta_pos + (reg.length - 1)

    return region


def _build_region_staple_1(dna: DNAType, strand_index: int) -> list[RegionType]:
    strand = dna.strand[strand_index]
    base = Mani_Go_Start_Base(dna, strand_index)
    if base == -1:
        return []

    region: list[RegionType] = []
    b_region = False
    b_vertex = False

    for j in range(1, strand.n_base + 1):
        across = dna.top[base].across

        if across == -1:
            b_region = True
            b_vertex = True
            base = dna.top[base].up
            continue
        if (
            dna.top[base].dn == -1
            or dna.top[base].up == -1
            or dna.top[base].xover != -1
            or dna.top[across].xover != -1
        ):
            b_region = True
            base = dna.top[base].up
            continue

        if b_region:
            b_region = False
            reg = RegionType()
            reg.sta_base = base
            reg.length = 1
            reg.sta_pos = j
            reg.end_pos = j
            if b_vertex:
                reg.types = 1
                if region:
                    region[-1].types = 1
                b_vertex = False
            else:
                reg.types = 2
            region.append(reg)
        else:
            region[-1].length += 1
            region[-1].end_pos = j

        base = dna.top[base].up

    for reg in region:
        len_cen = (reg.length + 1) // 2 - 1
        if reg.types == 1:
            dn = dna.top[reg.sta_base].dn
            if dn != -1 and dna.top[dn].across != -1:
                len_cen -= 1
            else:
                len_cen += 1

        cen_base = reg.sta_base
        for _ in range(max(0, len_cen)):
            cen_base = dna.top[cen_base].up
        reg.cen_base = cen_base
        reg.cen_pos = reg.sta_pos + len_cen

        end_base = reg.sta_base
        for _ in range(reg.length - 1):
            end_base = dna.top[end_base].up
        reg.end_base = end_base
        reg.end_pos = reg.sta_pos + (reg.length - 1)

        if (reg.types == 1 and reg.length + 1 >= 14) or (reg.types == 2 and reg.length + 2 >= 14):
            sta_base = reg.sta_base
            dn = dna.top[sta_base].dn
            if dn != -1:
                dna.top[dn].b_14nt = True
                dna.top[dn].status = "S"
            dna.top[sta_base].b_14nt = True
            dna.top[sta_base].status = "S"
            for _ in range(reg.length - 1):
                sta_base = dna.top[sta_base].up
                dna.top[sta_base].b_14nt = True
                dna.top[sta_base].status = "S"
            up = dna.top[sta_base].up
            if up != -1 and dna.top[up].across != -1:
                dna.top[up].b_14nt = True
                dna.top[up].status = "S"

        if (reg.types == 1 and reg.length + 1 <= 4) or (reg.types == 2 and reg.length + 2 <= 4):
            sta_base = reg.sta_base
            dn = dna.top[sta_base].dn
            if dn != -1:
                dna.top[dn].status = "F"
            dna.top[sta_base].status = "F"
            for _ in range(reg.length - 1):
                sta_base = dna.top[sta_base].up
                dna.top[sta_base].status = "F"
            up = dna.top[sta_base].up
            if up != -1 and dna.top[up].across != -1:
                dna.top[up].status = "F"

    return region


def _make_nick_scaf(geom: GeomType, mesh: MeshType, dna: DNAType) -> None:
    if dna.n_strand == 0:
        return

    croL_n_xover = _count_scaffold_crossovers_by_edge(geom, mesh, dna)

    nick_marked = False
    for strand in dna.strand:
        if strand.type1 == "stap":
            continue

        b_inside = False
        while True:
            base = _find_scaffold_nick_start_base(mesh, dna, strand)
            if base == -1:
                break

            max_strt_base, max_count = _scan_scaffold_nick_candidate_run(
                geom,
                mesh,
                dna,
                strand,
                base,
                croL_n_xover,
                b_inside,
            )

            if max_count == 0:
                if b_inside:
                    break
                b_inside = True
                continue

            base = dna.top[max_strt_base].id
            for _ in range(max_count // 2 + 1):
                if base == -1:
                    break
                base = dna.top[base].up
            if base == -1:
                break
            dn_base = dna.top[base].dn
            if dn_base == -1:
                break
            nick_marked = _apply_scaffold_nick(dna, base, dn_base, mark_status=not nick_marked) or nick_marked
            break


def _count_scaffold_crossovers_by_edge(geom: GeomType, mesh: MeshType, dna: DNAType) -> list[int]:
    croL_n_xover = [0 for _ in range(geom.n_croL)]
    if not dna.strand:
        return croL_n_xover
    for base in dna.strand[0].base:
        node = dna.top[base].node
        if node == -1:
            continue
        edge = mesh.node[node].croL
        if dna.top[base].xover != -1:
            croL_n_xover[edge] += 1
    return croL_n_xover


def _find_scaffold_nick_start_base(mesh: MeshType, dna: DNAType, strand: StrandType) -> int:
    base = strand.base[0]
    for _ in range(strand.n_base):
        if base == -1:
            break
        node = dna.top[base].node
        if node != -1 and mesh.node[node].dn == -1:
            break
        base = dna.top[base].dn
    return base


def _advance_to_scaffold_nick_row_base(
    geom: GeomType,
    mesh: MeshType,
    dna: DNAType,
    base: int,
) -> int:
    while True:
        if base == -1:
            return -1
        if dna.top[base].node != -1:
            node = dna.top[base].node
            sec = mesh.node[node].sec
            if geom.sec.posR[sec] == 1:
                return base
        base = dna.top[base].up


def _scan_scaffold_nick_candidate_run(
    geom: GeomType,
    mesh: MeshType,
    dna: DNAType,
    strand: StrandType,
    base: int,
    croL_n_xover: list[int],
    b_inside: bool,
) -> tuple[int, int]:
    max_strt_base = base
    max_count = 0
    count = 0
    strt_base = base

    for _ in range(strand.n_base):
        base = _advance_to_scaffold_nick_row_base(geom, mesh, dna, base)
        if base == -1:
            break

        node = dna.top[base].node
        edge = mesh.node[node].croL

        if croL_n_xover[edge] == 0:
            if _is_scaffold_nick_boundary(geom, mesh, dna, base, node, b_inside):
                if max_count < count:
                    max_count = count
                    max_strt_base = strt_base
                count = 0
                strt_base = base
            else:
                count += 1

        base = dna.top[base].up

    return max_strt_base, max_count


def _apply_scaffold_nick(dna: DNAType, base: int, dn_base: int, mark_status: bool) -> bool:
    dna.top[base].dn = -1
    dna.top[dn_base].up = -1
    if mark_status:
        dna.top[base].status = "nick_scaf"
        return True
    return False


def _is_scaffold_nick_boundary(
    geom: GeomType,
    mesh: MeshType,
    dna: DNAType,
    base: int,
    node: int,
    b_inside: bool,
) -> bool:
    across = dna.top[base].across
    across_xover = dna.top[across].xover if across != -1 else -1
    across_up = dna.top[across].up if across != -1 else -1
    across_dn = dna.top[across].dn if across != -1 else -1
    return (
        dna.top[base].xover != -1
        or across == -1
        or across_xover != -1
        or across_up == -1
        or across_dn == -1
        or mesh.node[node].up == -1
        or mesh.node[node].dn == -1
        or mesh.node[node].mitered != -1
        or (
            not b_inside
            and geom.iniL[mesh.node[node].iniL].neiF[0] != -1
            and geom.iniL[mesh.node[node].iniL].neiF[1] != -1
        )
        or (
            not b_inside
            and geom.iniL[mesh.node[node].iniL].neiF[1] == -1
            and geom.croL[mesh.node[node].croL].sec == 0
        )
        or (
            not b_inside
            and geom.iniL[mesh.node[node].iniL].neiF[0] == -1
            and geom.croL[mesh.node[node].croL].sec == 1
        )
    )


def _make_short_scaf(mesh: MeshType, dna: DNAType) -> None:
    if para.para_max_cut_scaf == 0:
        return

    for i, strand in enumerate(dna.strand):
        if strand.type1 == "stap":
            continue

        base = Mani_Go_Start_Base(dna, i)
        if base == -1:
            continue

        region = _collect_short_scaffold_regions(dna, base, strand.n_base)
        _assign_short_scaffold_region_centers(dna, region)
        _cut_short_scaffold_regions(dna, region)


def _collect_short_scaffold_regions(dna: DNAType, base: int, n_base: int) -> list[ScafRegionType]:
    region: list[ScafRegionType] = []
    b_region = False
    for j in range(n_base):
        across = dna.top[base].across

        if across == -1:
            b_region = True
            base = dna.top[base].up
            continue
        if (
            dna.top[base].dn == -1
            or dna.top[base].up == -1
            or dna.top[base].xover != -1
            or dna.top[across].xover != -1
        ):
            b_region = True
            base = dna.top[base].up
            continue

        if b_region:
            b_region = False
            reg = ScafRegionType()
            reg.strt_base = base
            reg.length = 1
            reg.strt_pos = j + 1
            reg.end_pos = j + 1
            region.append(reg)
        else:
            region[-1].length += 1
            region[-1].end_pos = j + 1

        base = dna.top[base].up
    return region


def _assign_short_scaffold_region_centers(dna: DNAType, region: list[ScafRegionType]) -> None:
    for reg in region:
        cntr_base = reg.strt_base
        for _ in range((reg.length + 1) // 2 - 1):
            cntr_base = dna.top[cntr_base].up
        reg.cntr_base = cntr_base
        reg.cntr_pos = reg.strt_pos + ((reg.length + 1) // 2 - 1)


def _cut_short_scaffold_regions(dna: DNAType, region: list[ScafRegionType]) -> None:
    begin_pos = 0
    b_cut = False
    for j, reg in enumerate(region):
        cntr_base = reg.cntr_base
        cntr_pos = reg.cntr_pos
        length = cntr_pos - begin_pos
        if not b_cut and length > para.para_max_cut_scaf:
            b_cut = True
        if b_cut:
            pre_base = cntr_base
            pre_pos = cntr_pos
            if j == len(region) - 1:
                continue

            dna.n_scaf += 1
            if not _break_top_link_after_base(dna, pre_base):
                continue

            begin_pos = pre_pos
            b_cut = False


def _break_top_link_after_base(dna: DNAType, base: int) -> bool:
    up_base = dna.top[base].up
    if up_base == -1:
        return False
    dna.top[base].up = -1
    dna.top[up_base].dn = -1
    return True


def _rebuild_strand(dna: DNAType) -> None:
    n_top = dna.n_top
    visited = [False] * n_top
    strands: list[tuple[list[int], bool]] = []

    for i in range(n_top):
        if visited[i]:
            continue

        cur = dna.top[i]
        start_id = cur.id

        while True:
            if cur.dn == -1:
                circular = False
                break
            if cur.dn == start_id:
                circular = True
                cur = dna.top[start_id]
                break
            cur = dna.top[cur.dn]
            if visited[cur.id]:
                raise ValueError("Reached a visited base while rebuilding strands.")

        base_ids: list[int] = []
        while True:
            base_ids.append(cur.id)
            dna.top[cur.id].strand = len(strands) + 1
            dna.top[cur.id].address = len(base_ids)
            dna.top[cur.id].b_14nt = False
            visited[cur.id] = True
            if (not circular and cur.up == -1) or (circular and cur.up == start_id):
                break
            cur = dna.top[cur.up]
            if visited[cur.id]:
                raise ValueError("Reached a visited base while rebuilding strands.")

        strands.append((base_ids, circular))

    dna.n_strand = dna.n_scaf + dna.n_stap
    dna.strand = []

    for idx, (base_ids, circular) in enumerate(strands):
        s = StrandType()
        s.n_base = len(base_ids)
        s.b_circular = circular
        s.base = list(base_ids)
        s.type1 = "scaf" if idx < dna.n_scaf else "stap"
        dna.strand.append(s)

    dna.len_min_stap = 10000
    dna.len_max_stap = -10000
    for s in dna.strand:
        if s.type1 == "stap":
            if s.n_base < dna.len_min_stap:
                dna.len_min_stap = s.n_base
            if s.n_base > dna.len_max_stap:
                dna.len_max_stap = s.n_base


def _order_staple(dna: DNAType) -> None:
    if dna.n_stap <= 0:
        dna.order_stap = []
        return

    order = []
    for idx, strand in enumerate(dna.strand):
        if strand.type1 != "stap":
            continue
        order.append([idx, strand.n_base])

    order.sort(key=lambda item: item[1])
    dna.order_stap = order
