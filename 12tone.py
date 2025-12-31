import numpy as np

from urllib.parse import urlparse, unquote

import heapq

PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]


def find_triads_in_matrix(matrix, wrap: bool = False, allow_inversions: bool = True):
    """Search a pitch-class matrix (0-11) for adjacent major and minor triads.

    Returns tuples:
      (label, location_type, index_1based, start_pos_0based, pitches_list, root_pc)
    """
    MAJOR_INTERVALS = (4, 7)
    MINOR_INTERVALS = (3, 7)

    results = []

    def check_triad(trio, location_type, index, start_pos):
        def _check_as_root(root, t2, t3):
            interval1 = (t2 - root) % 12
            interval2 = (t3 - root) % 12
            s = {interval1, interval2}
            if s == set(MAJOR_INTERVALS):
                return "Major"
            if s == set(MINOR_INTERVALS):
                return "Minor"
            return None

        trio_list = [int(x) for x in trio]

        if not allow_inversions:
            p1, p2, p3 = trio_list
            triad_type = _check_as_root(p1, p2, p3)
            if triad_type:
                root_pc = int(p1 % 12)
                results.append(
                    (
                        f"{triad_type} Triad (Root: {root_pc})",
                        location_type,
                        index,
                        start_pos,
                        trio_list,
                        root_pc,
                    )
                )
            return

        inversions = {0: "root position", 1: "1st inversion", 2: "2nd inversion"}
        for rot in range(3):
            root = trio_list[rot % 3]
            t2 = trio_list[(rot + 1) % 3]
            t3 = trio_list[(rot + 2) % 3]
            triad_type = _check_as_root(root, t2, t3)
            if triad_type:
                root_pc = int(root % 12)
                inv_label = inversions[rot]
                results.append(
                    (
                        f"{triad_type} Triad (Root: {root_pc}, {inv_label})",
                        location_type,
                        index,
                        start_pos,
                        trio_list,
                        root_pc,
                    )
                )
                return

    # --- Rows ---
    for i, row in enumerate(matrix):
        n = len(row)
        starts = range(n) if wrap else range(max(0, n - 2))
        for j in starts:
            trio = (
                np.array([row[j % n], row[(j + 1) % n], row[(j + 2) % n]])
                if wrap
                else row[j : j + 3]
            )
            if len(trio) == 3:
                check_triad(trio, "Row", i + 1, j)

    # --- Columns ---
    transposed_matrix = matrix.T
    for i, col in enumerate(transposed_matrix):
        n = len(col)
        starts = range(n) if wrap else range(max(0, n - 2))
        for j in starts:
            trio = (
                np.array([col[j % n], col[(j + 1) % n], col[(j + 2) % n]])
                if wrap
                else col[j : j + 3]
            )
            if len(trio) == 3:
                check_triad(trio, "Column", i + 1, j)

    return results


def print_results(results, show_note_names: bool = False):
    """Formats and prints the final output."""
    if not results:
        print("\n❌ No adjacent major or minor triads were found.")
        return

    print("\n✅ Found Triads:")
    print("---------------------------------------------------------")

    for entry in results:
        if len(entry) == 6:
            triad_type, location, index, start_pos, pitches, root_pc = entry
        else:
            triad_type, location, index, start_pos, pitches = entry
            root_pc = pitches[0]

        readable_pitches = [PITCH_NAMES[int(p) % 12] for p in pitches]
        readable_type = triad_type.replace(
            f"Root: {root_pc}", f"Root: {PITCH_NAMES[int(root_pc) % 12]}"
        )

        pos_note = PITCH_NAMES[int(root_pc) % 12]
        start_display = pos_note if show_note_names else str(start_pos)

        print(
            f"[{location} {index}, Start: {start_display}, Root: {pos_note}] {readable_type} -> Pitches: {readable_pitches}"
        )

    print("---------------------------------------------------------")


def find_consonant_dyads(matrix, wrap: bool = False):
    """Search rows/columns for adjacent perfect fifth (IC5) and octave/unison (IC0) dyads."""
    TARGET_INTERVAL_CLASSES = {0: "Octave/Unison", 5: "Perfect Fifth"}
    results = []

    def check_pair(pair, location_type, index, start_pos):
        p1, p2 = int(pair[0]), int(pair[1])
        interval = (p2 - p1) % 12
        ic = min(interval, 12 - interval)
        if ic in TARGET_INTERVAL_CLASSES:
            label = TARGET_INTERVAL_CLASSES[ic]
            results.append((label, location_type, index, start_pos, [p1, p2], int(ic)))

    # Rows
    for i, row in enumerate(matrix):
        n = len(row)
        starts = range(n) if wrap else range(max(0, n - 1))
        for j in starts:
            pair = (
                np.array([row[j % n], row[(j + 1) % n]]) if wrap else row[j : j + 2]
            )
            if len(pair) == 2:
                check_pair(pair, "Row", i + 1, j)

    # Columns
    transposed_matrix = matrix.T
    for i, col in enumerate(transposed_matrix):
        n = len(col)
        starts = range(n) if wrap else range(max(0, n - 1))
        for j in starts:
            pair = (
                np.array([col[j % n], col[(j + 1) % n]]) if wrap else col[j : j + 2]
            )
            if len(pair) == 2:
                check_pair(pair, "Column", i + 1, j)

    return results


def _reference_row_for_matrix(matrix):
    """Return a representative 12-tone row for analysis.

    For a 12x12 twelve-tone matrix, the first row is a prime form.
    For other inputs, this falls back to the first row as provided.
    """
    if getattr(matrix, "shape", None) is None:
        return []
    if matrix.shape[0] < 1:
        return []
    row = [int(x) % 12 for x in matrix[0]]
    return row


def analyze_row_symmetry(row):
    """Analyze simple, high-signal symmetry patterns in a 12-note row.

    Currently focuses on hexachord mirror relationships:
      - exact retrograde mirror: last 6 == reverse(first 6)
      - transposed mirror: last 6 == Tn(reverse(first 6))
      - inverted mirror: last 6 == In(reverse(first 6))  (In: x -> -x + n)
    """
    if len(row) != 12:
        return {
            "hex_mirror_exact": False,
            "hex_mirror_transposed": False,
            "hex_mirror_transposed_n": None,
            "hex_mirror_inverted": False,
            "hex_mirror_inverted_n": None,
        }

    first = [int(x) % 12 for x in row[:6]]
    last = [int(x) % 12 for x in row[6:]]
    rev_first = list(reversed(first))

    hex_mirror_exact = (last == rev_first)

    # Transposed mirror: last[i] == rev_first[i] + n (mod 12)
    transposed_n = None
    for n in range(12):
        if all(((rev_first[i] + n) % 12) == last[i] for i in range(6)):
            transposed_n = n
            break

    # Inverted mirror: last[i] == -rev_first[i] + n (mod 12)
    inverted_n = None
    for n in range(12):
        if all(((-rev_first[i] + n) % 12) == last[i] for i in range(6)):
            inverted_n = n
            break

    return {
        "hex_mirror_exact": bool(hex_mirror_exact),
        "hex_mirror_transposed": transposed_n is not None,
        "hex_mirror_transposed_n": transposed_n,
        "hex_mirror_inverted": inverted_n is not None,
        "hex_mirror_inverted_n": inverted_n,
    }


def analyze_hexachordal_combinatoriality(row):
    """Analyze basic hexachordal combinatoriality for the first hexachord.

    For H = set(row[:6]), count how many transpositions/inversions produce a disjoint
    hexachord (i.e., H ∩ transform(H) == ∅). Since |H|=6, disjoint implies the union is
    the full aggregate.
    """
    if len(row) != 12:
        return {
            "comb_t_count": 0,
            "comb_t_first_n": None,
            "comb_i_count": 0,
            "comb_i_first_n": None,
        }

    h = {int(x) % 12 for x in row[:6]}
    if len(h) != 6:
        # Not a proper hexachord (row not a permutation or has duplicates)
        return {
            "comb_t_count": 0,
            "comb_t_first_n": None,
            "comb_i_count": 0,
            "comb_i_first_n": None,
        }

    t_ns = []
    i_ns = []

    for n in range(12):
        tset = {(x + n) % 12 for x in h}
        if h.isdisjoint(tset):
            t_ns.append(n)

        iset = {(-x + n) % 12 for x in h}
        if h.isdisjoint(iset):
            i_ns.append(n)

    return {
        "comb_t_count": int(len(t_ns)),
        "comb_t_first_n": (t_ns[0] if t_ns else None),
        "comb_i_count": int(len(i_ns)),
        "comb_i_first_n": (i_ns[0] if i_ns else None),
    }


def analyze_matrix_symmetry_and_set_structure(matrix):
    """Compute additive symmetry/set-structure metrics across all rows and columns.

    This treats repeated occurrences as additive by counting how many rows/columns
    exhibit each symmetry, and summing combinatoriality counts across rows/columns.
    """
    if getattr(matrix, "shape", None) is None or len(matrix.shape) != 2:
        return {
            "hex_mirror_exact_rows": 0,
            "hex_mirror_transposed_rows": 0,
            "hex_mirror_inverted_rows": 0,
            "hex_mirror_exact_cols": 0,
            "hex_mirror_transposed_cols": 0,
            "hex_mirror_inverted_cols": 0,
            "hex_mirror_transposed_first_n": None,
            "hex_mirror_inverted_first_n": None,
            "comb_t_total_rows": 0,
            "comb_i_total_rows": 0,
            "comb_t_total_cols": 0,
            "comb_i_total_cols": 0,
            "comb_t_first_n": None,
            "comb_i_first_n": None,
        }

    rows, cols = matrix.shape

    out = {
        "hex_mirror_exact_rows": 0,
        "hex_mirror_transposed_rows": 0,
        "hex_mirror_inverted_rows": 0,
        "hex_mirror_exact_cols": 0,
        "hex_mirror_transposed_cols": 0,
        "hex_mirror_inverted_cols": 0,
        "hex_mirror_transposed_first_n": None,
        "hex_mirror_inverted_first_n": None,
        "comb_t_total_rows": 0,
        "comb_i_total_rows": 0,
        "comb_t_total_cols": 0,
        "comb_i_total_cols": 0,
        "comb_t_first_n": None,
        "comb_i_first_n": None,
    }

    # Rows
    for r in range(rows):
        row = [int(x) % 12 for x in matrix[r, :]]
        if len(row) == 12:
            sym = analyze_row_symmetry(row)
            if sym.get("hex_mirror_exact"):
                out["hex_mirror_exact_rows"] += 1
            if sym.get("hex_mirror_transposed"):
                out["hex_mirror_transposed_rows"] += 1
                if out["hex_mirror_transposed_first_n"] is None:
                    out["hex_mirror_transposed_first_n"] = sym.get(
                        "hex_mirror_transposed_n"
                    )
            if sym.get("hex_mirror_inverted"):
                out["hex_mirror_inverted_rows"] += 1
                if out["hex_mirror_inverted_first_n"] is None:
                    out["hex_mirror_inverted_first_n"] = sym.get(
                        "hex_mirror_inverted_n"
                    )

            comb = analyze_hexachordal_combinatoriality(row)
            out["comb_t_total_rows"] += int(comb.get("comb_t_count", 0))
            out["comb_i_total_rows"] += int(comb.get("comb_i_count", 0))
            if out["comb_t_first_n"] is None and comb.get("comb_t_first_n") is not None:
                out["comb_t_first_n"] = comb.get("comb_t_first_n")
            if out["comb_i_first_n"] is None and comb.get("comb_i_first_n") is not None:
                out["comb_i_first_n"] = comb.get("comb_i_first_n")

    # Columns
    for c in range(cols):
        col = [int(x) % 12 for x in matrix[:, c]]
        if len(col) == 12:
            sym = analyze_row_symmetry(col)
            if sym.get("hex_mirror_exact"):
                out["hex_mirror_exact_cols"] += 1
            if sym.get("hex_mirror_transposed"):
                out["hex_mirror_transposed_cols"] += 1
                if out["hex_mirror_transposed_first_n"] is None:
                    out["hex_mirror_transposed_first_n"] = sym.get(
                        "hex_mirror_transposed_n"
                    )
            if sym.get("hex_mirror_inverted"):
                out["hex_mirror_inverted_cols"] += 1
                if out["hex_mirror_inverted_first_n"] is None:
                    out["hex_mirror_inverted_first_n"] = sym.get(
                        "hex_mirror_inverted_n"
                    )

            comb = analyze_hexachordal_combinatoriality(col)
            out["comb_t_total_cols"] += int(comb.get("comb_t_count", 0))
            out["comb_i_total_cols"] += int(comb.get("comb_i_count", 0))
            if out["comb_t_first_n"] is None and comb.get("comb_t_first_n") is not None:
                out["comb_t_first_n"] = comb.get("comb_t_first_n")
            if out["comb_i_first_n"] is None and comb.get("comb_i_first_n") is not None:
                out["comb_i_first_n"] = comb.get("comb_i_first_n")

    return out


def analyze_diatonic_affinity(matrix):
    """Diatonic affinity (2A) across rows and columns.

    For each row/column of length 12, we find the major scale transposition that yields
    the longest *consecutive run* of notes contained in that scale, then sum those best
    run-lengths across rows and across columns.

    This makes the metric non-trivial for 12-tone permutations (plain set-overlap would
    always be 7).
    """

    MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}

    def longest_run(seq_bools):
        best = 0
        cur = 0
        for b in seq_bools:
            if b:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        return best

    def best_major_run_for_sequence(seq):
        # seq is a list of 12 pitch-classes
        best_len = 0
        best_t = None
        for t in range(12):
            scale = {(pc + t) % 12 for pc in MAJOR_SCALE}
            in_scale = [((int(x) % 12) in scale) for x in seq]
            run_len = longest_run(in_scale)
            if run_len > best_len:
                best_len = run_len
                best_t = t
        return best_len, best_t

    if getattr(matrix, "shape", None) is None or len(matrix.shape) != 2:
        return {
            "diatonic_max_run_rows_sum": 0,
            "diatonic_max_run_cols_sum": 0,
            "diatonic_max_run_rows_max": 0,
            "diatonic_max_run_cols_max": 0,
            "diatonic_best_t_first": None,
            "diatonic_best_row_index_1based": None,
            "diatonic_best_row_run": 0,
            "diatonic_best_row_t": None,
            "diatonic_best_col_index_1based": None,
            "diatonic_best_col_run": 0,
            "diatonic_best_col_t": None,
        }

    rows, cols = matrix.shape

    rows_sum = 0
    cols_sum = 0
    rows_max = 0
    cols_max = 0
    best_t_first = None
    best_row_index_1based = None
    best_row_t = None
    best_col_index_1based = None
    best_col_t = None

    # Rows
    for r in range(rows):
        seq = [int(x) % 12 for x in matrix[r, :]]
        if len(seq) != 12:
            continue
        run_len, t = best_major_run_for_sequence(seq)
        rows_sum += int(run_len)
        if run_len > rows_max:
            rows_max = int(run_len)
            best_row_index_1based = int(r + 1)
            best_row_t = int(t) if t is not None else None
        if best_t_first is None and t is not None:
            best_t_first = int(t)

    # Columns
    for c in range(cols):
        seq = [int(x) % 12 for x in matrix[:, c]]
        if len(seq) != 12:
            continue
        run_len, t = best_major_run_for_sequence(seq)
        cols_sum += int(run_len)
        if run_len > cols_max:
            cols_max = int(run_len)
            best_col_index_1based = int(c + 1)
            best_col_t = int(t) if t is not None else None
        if best_t_first is None and t is not None:
            best_t_first = int(t)

    return {
        "diatonic_max_run_rows_sum": int(rows_sum),
        "diatonic_max_run_cols_sum": int(cols_sum),
        "diatonic_max_run_rows_max": int(rows_max),
        "diatonic_max_run_cols_max": int(cols_max),
        "diatonic_best_t_first": best_t_first,
        "diatonic_best_row_index_1based": best_row_index_1based,
        "diatonic_best_row_run": int(rows_max),
        "diatonic_best_row_t": best_row_t,
        "diatonic_best_col_index_1based": best_col_index_1based,
        "diatonic_best_col_run": int(cols_max),
        "diatonic_best_col_t": best_col_t,
    }


def describe_consonance(
    matrix,
    consonance: dict,
    *,
    show_note_names: bool = False,
    max_examples_per_type: int = 3,
):
    """Generate a text description of the consonance characteristics.

    Intentionally focuses on a handful of high-signal musical qualities and points
    to specific rows/columns when the underlying analysis includes locations.
    """

    def _pc_name(pc: int) -> str:
        return PITCH_NAMES[int(pc) % 12]

    def _fmt_pc(pc: int) -> str:
        return _pc_name(pc) if show_note_names else str(int(pc) % 12)

    lines = []

    score = int(consonance.get("score", 0))
    majors = int(consonance.get("major_triads", 0))
    minors = int(consonance.get("minor_triads", 0))
    fifths = int(consonance.get("perfect_fifth_dyads", 0))
    octaves = int(consonance.get("octave_dyads", 0))
    balanced_expected = bool(consonance.get("expected_balanced_triads"))

    lines.append(f"Overall: score={score}; triads Major={majors}, Minor={minors}; dyads 5ths={fifths}, oct/unison={octaves}.")

    if majors != minors:
        if majors > minors:
            lines.append("Color: slightly brighter/major-leaning (more major than minor triads detected).")
        else:
            lines.append("Color: slightly darker/minor-leaning (more minor than major triads detected).")
        if balanced_expected:
            lines.append("Note: for 12×12 matrices you often expect major/minor to balance; this one is unbalanced.")
    else:
        lines.append("Color: major/minor balance (triad counts match).")

    triads = consonance.get("triads") or []
    dyads = consonance.get("dyads") or []

    def _triad_examples(prefix: str):
        out = []
        for entry in triads:
            if not entry:
                continue
            label = entry[0]
            if not (isinstance(label, str) and label.startswith(prefix)):
                continue
            # triad tuple shape is either 6-tuple (includes root_pc) or 5-tuple.
            if len(entry) >= 6:
                _label, loc, idx_1b, start, pitches, root_pc = entry[:6]
            else:
                _label, loc, idx_1b, start, pitches = entry[:5]
                root_pc = pitches[0] if pitches else None

            if root_pc is None:
                continue

            pitch_str = ", ".join(_fmt_pc(p) for p in pitches)
            root_str = _pc_name(root_pc) if show_note_names else str(int(root_pc) % 12)
            out.append(
                f"- {loc} {int(idx_1b)} (start {int(start)}): {label} ⇒ [{pitch_str}] (root {root_str})"
            )
            if len(out) >= int(max_examples_per_type):
                break
        return out

    major_ex = _triad_examples("Major")
    minor_ex = _triad_examples("Minor")
    if major_ex or minor_ex:
        lines.append("Triad hotspots (examples):")
        lines.extend(major_ex)
        lines.extend(minor_ex)

    def _dyad_examples(target_label: str):
        out = []
        for entry in dyads:
            if not entry:
                continue
            label = entry[0]
            if label != target_label:
                continue
            # (label, location_type, index, start_pos, [p1,p2], ic)
            if len(entry) < 6:
                continue
            _label, loc, idx_1b, start, pitches, ic = entry[:6]
            if not pitches or len(pitches) != 2:
                continue
            p1, p2 = pitches
            out.append(
                f"- {loc} {int(idx_1b)} (start {int(start)}): {label} (ic{int(ic)}) between {_fmt_pc(p1)} and {_fmt_pc(p2)}"
            )
            if len(out) >= int(max_examples_per_type):
                break
        return out

    fifth_ex = _dyad_examples("Perfect Fifth")
    oct_ex = _dyad_examples("Octave/Unison")
    if fifth_ex or oct_ex:
        lines.append("Dyad anchors (examples):")
        lines.extend(fifth_ex)
        lines.extend(oct_ex)

    # Matrix-wide structure hints (these don’t currently track exact row/col indices, just counts).
    hm_r = int(consonance.get("hex_mirror_exact_rows", 0))
    hm_c = int(consonance.get("hex_mirror_exact_cols", 0))
    ht_r = int(consonance.get("hex_mirror_transposed_rows", 0))
    ht_c = int(consonance.get("hex_mirror_transposed_cols", 0))
    hi_r = int(consonance.get("hex_mirror_inverted_rows", 0))
    hi_c = int(consonance.get("hex_mirror_inverted_cols", 0))
    if (hm_r + hm_c + ht_r + ht_c + hi_r + hi_c) > 0:
        parts = []
        if (hm_r + hm_c) > 0:
            parts.append(f"exact hexachord mirrors in {hm_r} row(s) and {hm_c} column(s)")
        if (ht_r + ht_c) > 0:
            n = consonance.get("hex_mirror_transposed_first_n")
            parts.append(f"transposed mirrors in {ht_r} row(s) and {ht_c} column(s)" + (f" (first Tn={n})" if n is not None else ""))
        if (hi_r + hi_c) > 0:
            n = consonance.get("hex_mirror_inverted_first_n")
            parts.append(f"inverted mirrors in {hi_r} row(s) and {hi_c} column(s)" + (f" (first In={n})" if n is not None else ""))
        lines.append("Set-structure: " + "; ".join(parts) + ".")

    comb_t = int(consonance.get("comb_t_total_rows", 0)) + int(consonance.get("comb_t_total_cols", 0))
    comb_i = int(consonance.get("comb_i_total_rows", 0)) + int(consonance.get("comb_i_total_cols", 0))
    if comb_t or comb_i:
        extra = []
        if comb_t:
            extra.append(f"T-combinatoriality count≈{comb_t}")
        if comb_i:
            extra.append(f"I-combinatoriality count≈{comb_i}")
        lines.append("Combinatoriality: " + "; ".join(extra) + ".")

    # Diatonic affinity: now includes best row/col for the longest major-scale run.
    br = consonance.get("diatonic_best_row_index_1based")
    bc = consonance.get("diatonic_best_col_index_1based")
    br_run = int(consonance.get("diatonic_best_row_run", 0))
    bc_run = int(consonance.get("diatonic_best_col_run", 0))
    br_t = consonance.get("diatonic_best_row_t")
    bc_t = consonance.get("diatonic_best_col_t")
    if (br is not None and br_run > 0) or (bc is not None and bc_run > 0):
        def _scale_name(t):
            if t is None:
                return ""
            return _pc_name(int(t)) if show_note_names else str(int(t) % 12)

        pieces = []
        if br is not None and br_run > 0:
            pieces.append(f"Row {int(br)} has the strongest major-scale run (len {br_run}" + (f", T={_scale_name(br_t)}" if br_t is not None else "") + ")")
        if bc is not None and bc_run > 0:
            pieces.append(f"Column {int(bc)} has the strongest major-scale run (len {bc_run}" + (f", T={_scale_name(bc_t)}" if bc_t is not None else "") + ")")
        lines.append("Diatonic affinity: " + "; ".join(pieces) + ".")

    return "\n".join(lines)


DEFAULT_WEIGHTS = {
    # Balanced by default (your research indicates major/minor counts match in 12x12 matrices)
    "major_triad": 3,
    "minor_triad": 3,
    "perfect_fifth_dyad": 1,
    "octave_dyad": 2,

    # Row symmetry / set-structure (initial defaults; tune as desired)
    # NOTE: These are now applied per matching row/column (additive).
    "hex_mirror_exact": 10,
    "hex_mirror_transposed": 8,
    "hex_mirror_inverted": 8,
    # Combinatoriality counts can be >1 per row/col; these apply per count.
    "comb_t": 1,
    "comb_i": 1,

    # Diatonic affinity (per best run-length per row/col)
    "diatonic_max_run": 1,
}


def _redact_db_url(db_url: str) -> str:
    try:
        parsed = urlparse(db_url)
        if not parsed.scheme:
            return "<redacted>"
        netloc = parsed.netloc
        if "@" in netloc and ":" in netloc.split("@", 1)[0]:
            userinfo, hostinfo = netloc.split("@", 1)
            user = userinfo.split(":", 1)[0]
            netloc = f"{user}:***@{hostinfo}"
        return parsed._replace(netloc=netloc).geturl()
    except Exception:
        return "<redacted>"


def _parse_mysql_url(db_url: str):
    parsed = urlparse(db_url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"mysql", "mysql+pymysql", "mariadb", "mariadb+pymysql"}:
        raise ValueError(
            "--export-sql-db must be a MySQL/MariaDB URL like mysql://user:pass@host:3306/db"
        )

    if not parsed.hostname:
        raise ValueError("--export-sql-db is missing a hostname")
    database = (parsed.path or "").lstrip("/")
    if not database:
        raise ValueError("--export-sql-db is missing a database name (e.g. /12Tone)")

    return {
        "host": parsed.hostname,
        "port": int(parsed.port or 3306),
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "database": unquote(database),
    }


def _mysql_connect(db_url: str):
    try:
        import pymysql
    except Exception as ex:
        raise RuntimeError(
            "Missing dependency for --export-sql-db. Install with: pip install pymysql"
        ) from ex

    cfg = _parse_mysql_url(db_url)
    return pymysql.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        autocommit=True,
        charset="utf8mb4",
    )


def _mysql_ensure_table(cursor, table_name: str = "Tonality_Test"):
    if not table_name.replace("_", "").isalnum():
        raise ValueError("SQL table name must be alphanumeric/underscore")
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
          id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
          created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
          run_index INT NULL,
          seed BIGINT NULL,
          matrix_rows INT NOT NULL,
          matrix_cols INT NOT NULL,
          wrap TINYINT(1) NOT NULL,
          allow_inversions TINYINT(1) NOT NULL,
          major_triads INT NOT NULL,
          minor_triads INT NOT NULL,
          perfect_fifth_dyads INT NOT NULL,
          octave_dyads INT NOT NULL,

          hex_mirror_exact TINYINT(1) NOT NULL DEFAULT 0,
          hex_mirror_transposed TINYINT(1) NOT NULL DEFAULT 0,
          hex_mirror_transposed_n INT NULL,
          hex_mirror_inverted TINYINT(1) NOT NULL DEFAULT 0,
          hex_mirror_inverted_n INT NULL,

          comb_t_count INT NOT NULL DEFAULT 0,
          comb_t_first_n INT NULL,
          comb_i_count INT NOT NULL DEFAULT 0,
          comb_i_first_n INT NULL,

          score INT NOT NULL
        ) ENGINE=InnoDB
        """
    )

    # Auto-migrate existing tables (MySQL doesn't reliably support ADD COLUMN IF NOT EXISTS)
    desired_columns = {
        "hex_mirror_exact": "TINYINT(1) NOT NULL DEFAULT 0",
        "hex_mirror_transposed": "TINYINT(1) NOT NULL DEFAULT 0",
        "hex_mirror_transposed_n": "INT NULL",
        "hex_mirror_inverted": "TINYINT(1) NOT NULL DEFAULT 0",
        "hex_mirror_inverted_n": "INT NULL",
        "comb_t_count": "INT NOT NULL DEFAULT 0",
        "comb_t_first_n": "INT NULL",
        "comb_i_count": "INT NOT NULL DEFAULT 0",
        "comb_i_first_n": "INT NULL",

        # Additive (matrix-wide) versions
        "hex_mirror_exact_rows": "INT NOT NULL DEFAULT 0",
        "hex_mirror_transposed_rows": "INT NOT NULL DEFAULT 0",
        "hex_mirror_inverted_rows": "INT NOT NULL DEFAULT 0",
        "hex_mirror_exact_cols": "INT NOT NULL DEFAULT 0",
        "hex_mirror_transposed_cols": "INT NOT NULL DEFAULT 0",
        "hex_mirror_inverted_cols": "INT NOT NULL DEFAULT 0",
        "hex_mirror_transposed_first_n": "INT NULL",
        "hex_mirror_inverted_first_n": "INT NULL",
        "comb_t_total_rows": "INT NOT NULL DEFAULT 0",
        "comb_i_total_rows": "INT NOT NULL DEFAULT 0",
        "comb_t_total_cols": "INT NOT NULL DEFAULT 0",
        "comb_i_total_cols": "INT NOT NULL DEFAULT 0",

        # Diatonic affinity (2A)
        "diatonic_max_run_rows_sum": "INT NOT NULL DEFAULT 0",
        "diatonic_max_run_cols_sum": "INT NOT NULL DEFAULT 0",
        "diatonic_max_run_rows_max": "INT NOT NULL DEFAULT 0",
        "diatonic_max_run_cols_max": "INT NOT NULL DEFAULT 0",
        "diatonic_best_t_first": "INT NULL",
    }

    for col_name, col_def in desired_columns.items():
        cursor.execute(f"SHOW COLUMNS FROM `{table_name}` LIKE %s", (col_name,))
        exists = cursor.fetchone() is not None
        if not exists:
            cursor.execute(
                f"ALTER TABLE `{table_name}` ADD COLUMN `{col_name}` {col_def}"
            )


def _mysql_insert_run(
    cursor,
    consonance: dict,
    matrix,
    *,
    run_index: int | None,
    seed: int | None,
    wrap: bool,
    allow_inversions: bool,
    table_name: str = "Tonality_Test",
):
    rows, cols = matrix.shape
    cursor.execute(
        f"""
        INSERT INTO `{table_name}`
          (run_index, seed, matrix_rows, matrix_cols, wrap, allow_inversions,
           major_triads, minor_triads, perfect_fifth_dyads, octave_dyads,
           hex_mirror_exact, hex_mirror_transposed, hex_mirror_transposed_n,
           hex_mirror_inverted, hex_mirror_inverted_n,
           comb_t_count, comb_t_first_n, comb_i_count, comb_i_first_n,
                     hex_mirror_exact_rows, hex_mirror_transposed_rows, hex_mirror_inverted_rows,
                     hex_mirror_exact_cols, hex_mirror_transposed_cols, hex_mirror_inverted_cols,
                     hex_mirror_transposed_first_n, hex_mirror_inverted_first_n,
                     comb_t_total_rows, comb_i_total_rows, comb_t_total_cols, comb_i_total_cols,
                      diatonic_max_run_rows_sum, diatonic_max_run_cols_sum,
                      diatonic_max_run_rows_max, diatonic_max_run_cols_max,
                      diatonic_best_t_first,
           score)
        VALUES
          (%s, %s, %s, %s, %s, %s,
           %s, %s, %s, %s,
           %s, %s, %s,
           %s, %s,
           %s, %s, %s, %s,
                     %s, %s, %s,
                     %s, %s, %s,
                     %s, %s,
                     %s, %s, %s, %s,
                      %s, %s,
                      %s, %s,
                      %s,
           %s)
        """,
        (
            run_index,
            seed,
            int(rows),
            int(cols),
            1 if wrap else 0,
            1 if allow_inversions else 0,
            int(consonance.get("major_triads", 0)),
            int(consonance.get("minor_triads", 0)),
            int(consonance.get("perfect_fifth_dyads", 0)),
            int(consonance.get("octave_dyads", 0)),

            1 if consonance.get("hex_mirror_exact") else 0,
            1 if consonance.get("hex_mirror_transposed") else 0,
            consonance.get("hex_mirror_transposed_n"),
            1 if consonance.get("hex_mirror_inverted") else 0,
            consonance.get("hex_mirror_inverted_n"),

            int(consonance.get("comb_t_count", 0)),
            consonance.get("comb_t_first_n"),
            int(consonance.get("comb_i_count", 0)),
            consonance.get("comb_i_first_n"),

            int(consonance.get("hex_mirror_exact_rows", 0)),
            int(consonance.get("hex_mirror_transposed_rows", 0)),
            int(consonance.get("hex_mirror_inverted_rows", 0)),
            int(consonance.get("hex_mirror_exact_cols", 0)),
            int(consonance.get("hex_mirror_transposed_cols", 0)),
            int(consonance.get("hex_mirror_inverted_cols", 0)),
            consonance.get("hex_mirror_transposed_first_n"),
            consonance.get("hex_mirror_inverted_first_n"),
            int(consonance.get("comb_t_total_rows", 0)),
            int(consonance.get("comb_i_total_rows", 0)),
            int(consonance.get("comb_t_total_cols", 0)),
            int(consonance.get("comb_i_total_cols", 0)),

            int(consonance.get("diatonic_max_run_rows_sum", 0)),
            int(consonance.get("diatonic_max_run_cols_sum", 0)),
            int(consonance.get("diatonic_max_run_rows_max", 0)),
            int(consonance.get("diatonic_max_run_cols_max", 0)),
            consonance.get("diatonic_best_t_first"),

            int(consonance.get("score", 0)),
        ),
    )


def _seed_to_row_permutation(seed_int: int):
    import random as _random

    rnd = _random.Random(int(seed_int))
    seed_nums = list(range(12))
    rnd.shuffle(seed_nums)
    return seed_nums


def _evaluate_random_seed(seed_int: int, *, wrap: bool, allow_inversions: bool):
    seed_nums = _seed_to_row_permutation(seed_int)
    mat = generate_dodecaphonic_matrix(seed_nums)

    found_triads = find_triads_in_matrix(mat, wrap=wrap, allow_inversions=allow_inversions)
    found_dyads = find_consonant_dyads(mat, wrap=wrap)
    consonance = consonance_factor(
        mat,
        wrap=wrap,
        allow_inversions=allow_inversions,
        triad_results=found_triads,
        dyad_results=found_dyads,
    )

    # Avoid storing large details (triads/dyads lists) in the best-only heap.
    consonance = {k: v for k, v in consonance.items() if k not in {"triads", "dyads"}}
    return mat, consonance


def _format_duration_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds != seconds or seconds < 0:  # NaN or negative
        seconds = 0.0
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def export_tonal_cube_slice_heatmap(
    matrix: np.ndarray,
    out_path: str,
    *,
    z_depth: int = 0,
    title: str | None = None,
    label_mode: str = "numbers",
) -> None:
    """Export a 12x12 tonal-cube slice heatmap as a PNG.

    This is essentially the 12-tone matrix with an optional Z-depth offset:
    (matrix + z_depth) % 12.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.shape != (12, 12):
        raise ValueError("--export-heatmap requires a 12x12 matrix")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        raise RuntimeError(
            "Missing dependency for --export-heatmap. Install with: pip install matplotlib seaborn"
        )

    z_depth = int(z_depth)
    data = (matrix.astype(int) + z_depth) % 12

    plt.figure(figsize=(10, 8))

    if label_mode == "pitches":
        pitch_names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
        annot = np.vectorize(lambda x: pitch_names[int(x) % 12])(data)
        sns.heatmap(
            data,
            annot=annot,
            fmt="s",
            cmap="hsv",
            cbar=True,
            linewidths=1,
            linecolor="black",
            square=True,
        )
    else:
        sns.heatmap(
            data,
            annot=True,
            fmt="d",
            cmap="hsv",
            cbar=True,
            linewidths=1,
            linecolor="black",
            square=True,
        )

    if title is None:
        title = ""
    plt.title(str(title))
    plt.xlabel("Prime Axis (X)")
    plt.ylabel("Inversion Axis (Y)")

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _best_only_worker(payload):
    """Worker for multiprocessing best-only search.

    Returns a list of (score, seed_int, consonance_summary).
    """
    n, top_k, wrap, allow_inversions, throttle_every, throttle_ms, heartbeat_seconds, worker_id, progress_enabled, updates_queue = payload
    import secrets as _secrets
    import time as _time
    import sys as _sys

    heap = []
    throttle_every = int(throttle_every or 0)
    throttle_ms = float(throttle_ms or 0.0)
    sleep_s = max(0.0, throttle_ms / 1000.0)

    heartbeat_seconds = float(heartbeat_seconds or 0.0)
    worker_id = int(worker_id or 0)
    progress_enabled = bool(progress_enabled)
    t0 = _time.perf_counter()
    last_beat = t0
    best_score = None

    updates_queue = updates_queue

    for i in range(int(n)):
        seed_int = _secrets.randbits(63)
        _, c = _evaluate_random_seed(seed_int, wrap=wrap, allow_inversions=allow_inversions)
        score = int(c.get("score", 0))
        item = (score, int(seed_int), c)

        if best_score is None or score > best_score:
            best_score = score

        heap_changed = False
        if len(heap) < int(top_k):
            heapq.heappush(heap, item)
            heap_changed = True
        else:
            if item[0] > heap[0][0] or (item[0] == heap[0][0] and item[1] > heap[0][1]):
                heapq.heapreplace(heap, item)
                heap_changed = True

        # Stream candidate updates to the parent so it can maintain a global top-K and optionally
        # snapshot the current winners to DB during long runs.
        if heap_changed and updates_queue is not None:
            try:
                updates_queue.put((int(score), int(seed_int)))
            except Exception:
                # Best-effort only; never fail the worker for telemetry.
                pass

        if sleep_s > 0.0 and throttle_every > 0 and ((i + 1) % throttle_every == 0):
            _time.sleep(sleep_s)

        if progress_enabled and heartbeat_seconds > 0:
            now = _time.perf_counter()
            if (now - last_beat) >= heartbeat_seconds:
                last_beat = now
                done = i + 1
                elapsed = now - t0
                rate = (done / elapsed) if elapsed > 0 else 0.0
                remaining = int(n) - done
                eta = (remaining / rate) if rate > 0 else 0.0
                pct = (100.0 * done / int(n)) if int(n) > 0 else 100.0
                print(
                    f"worker {worker_id}: {done}/{int(n)} ({pct:5.1f}%) rate={rate:,.1f}/s ETA={_format_duration_seconds(eta)} best={best_score}",
                    file=_sys.stderr,
                )

    return heap


def consonance_factor(matrix, wrap: bool = False, allow_inversions: bool = True, weights=None, triad_results=None, dyad_results=None):
    """Compute a Consonance Factor score for a matrix.

    Repeated occurrences are additive.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    triads = triad_results if triad_results is not None else find_triads_in_matrix(
        matrix, wrap=wrap, allow_inversions=allow_inversions
    )
    dyads = dyad_results if dyad_results is not None else find_consonant_dyads(matrix, wrap=wrap)

    majors = sum(1 for t, *_ in triads if isinstance(t, str) and t.startswith("Major"))
    minors = sum(1 for t, *_ in triads if isinstance(t, str) and t.startswith("Minor"))
    perfect_fifths = sum(1 for label, *_ in dyads if label == "Perfect Fifth")
    octaves = sum(1 for label, *_ in dyads if label == "Octave/Unison")

    ref_row = _reference_row_for_matrix(matrix)
    symmetry = analyze_row_symmetry(ref_row)
    comb = analyze_hexachordal_combinatoriality(ref_row)
    additive = analyze_matrix_symmetry_and_set_structure(matrix)
    diatonic = analyze_diatonic_affinity(matrix)

    score = (
        majors * w["major_triad"]
        + minors * w["minor_triad"]
        + perfect_fifths * w["perfect_fifth_dyad"]
        + octaves * w["octave_dyad"]
        # Additive across rows/cols
        + (additive.get("hex_mirror_exact_rows", 0) + additive.get("hex_mirror_exact_cols", 0))
        * w["hex_mirror_exact"]
        + (additive.get("hex_mirror_transposed_rows", 0) + additive.get("hex_mirror_transposed_cols", 0))
        * w["hex_mirror_transposed"]
        + (additive.get("hex_mirror_inverted_rows", 0) + additive.get("hex_mirror_inverted_cols", 0))
        * w["hex_mirror_inverted"]
        + (additive.get("comb_t_total_rows", 0) + additive.get("comb_t_total_cols", 0)) * w["comb_t"]
        + (additive.get("comb_i_total_rows", 0) + additive.get("comb_i_total_cols", 0)) * w["comb_i"]
        + (diatonic.get("diatonic_max_run_rows_sum", 0) + diatonic.get("diatonic_max_run_cols_sum", 0))
        * w["diatonic_max_run"]
    )

    expected_balanced_triads = bool(matrix.shape == (12, 12))

    return {
        "score": int(score),
        "major_triads": int(majors),
        "minor_triads": int(minors),
        "perfect_fifth_dyads": int(perfect_fifths),
        "octave_dyads": int(octaves),
        # Single-row fields (first row) kept for backwards compatibility/debugging
        **symmetry,
        **comb,

        # Additive, matrix-wide fields
        **additive,

        # Diatonic affinity
        **diatonic,
        "triads": triads,
        "dyads": dyads,
        "expected_balanced_triads": expected_balanced_triads,
    }


# Backward-compatible alias for the user-facing label "Connsonance Factor".
connsonance_factor = consonance_factor


# --- EXAMPLE USAGE ---

# Pitch Class Mapping: C=0, C#=1, D=2, Eb=3, E=4, F=5, F#=6, G=7, G#=8, A=9, Bb=10, B=11

# Example Matrix:
# Row 1: Major Triad (3, 7, 10 -> Eb, G, Bb)
# Row 2: Minor Triad (8, 11, 2 -> G#, B, D)
# Column 1: Minor Triad (3, 8, 5 -> Eb, G#, F) (Wait, this is not a triad 3+5=8, 8+5=1, 3 to 8 is 5, 3 to 5 is 2. Let's check...)
# Column 3: Major Triad (10, 2, 5 -> Bb, D, F) (10 to 2 is 4, 10 to 5 is 7) -> Correct!
test_matrix = np.array([
    [3, 7, 10, 1, 6],
    [8, 11, 2, 4, 9],
    [5, 0, 4, 9, 11],
    [1, 6, 9, 10, 2]
])

def parse_matrix_from_string(s):
    """
    Parse a user-provided matrix string where rows are separated by ';'
    and elements in a row are separated by spaces or commas.

    Examples:
      "3 7 10 1 6;8 11 2 4 9"  -> two rows
      "3,7,10;8,11,2"          -> two rows with commas
    """
    rows = [r.strip() for r in s.split(';') if r.strip()]
    matrix = []
    for row in rows:
        # allow comma or space separators
        parts = [p for p in (x.strip() for x in row.replace(',', ' ').split()) if p]
        try:
            nums = [int(x) for x in parts]
        except ValueError:
            raise ValueError(f"Invalid integer in row: '{row}'")
        matrix.append(nums)
    return np.array(matrix, dtype=int)


def parse_matrix_from_file(path):
    """Read a file as a matrix.

    Supports whitespace/comma-separated text files and CSV files.
    If the path ends with '.csv' it will be parsed using the csv module.
    """
    import os

    _, ext = os.path.splitext(path)
    ext = (ext or '').lower()
    if ext == '.csv':
        return parse_matrix_from_csv(path)

    with open(path, 'r', encoding='utf-8') as fh:
        content = fh.read().strip()
    return parse_matrix_from_string(content.replace('\n', ';'))


def parse_matrix_from_csv(path):
    """Parse a CSV file into a numeric matrix.

    The function reads rows from a CSV file and converts each value to an int.
    It ignores empty cells. If a row contains no numeric cells it will be skipped.
    All rows must have the same number of columns (after ignoring empty cells) or
    a ValueError will be raised.
    """
    import csv
    # Read raw rows first
    raw_rows = []
    with open(path, 'r', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        for r in reader:
            # Keep raw tokens (they might contain floats like '3.0' or extra whitespace)
            raw_rows.append([(p or '').strip() for p in r])

    def _strip_bom(s):
        # strip BOM if present (common for files exported on some platforms)
        if s.startswith('\ufeff'):
            return s.lstrip('\ufeff')
        return s

    def _is_int_token(tok):
        if tok == '':
            return False
        # integer-like or float that represents integer
        try:
            if '.' in tok:
                f = float(tok)
                return f.is_integer()
            int(tok)
            return True
        except Exception:
            return False

    def _to_int(tok):
        # convert integer-like token (e.g. '3', '3.0') to int
        if '.' in tok:
            f = float(tok)
            if not f.is_integer():
                raise ValueError(f"Non-integer float in CSV token: {tok}")
            return int(f)
        return int(tok)

    # Normalize tokens and drop empty tokens at row ends
    normalized_rows = []
    for r in raw_rows:
        # strip BOM from first token if present
        if r:
            r[0] = _strip_bom(r[0])

        # strip whitespace and keep tokens (including empty strings in-between)
        tokens = [t.strip() for t in r]

        # Remove trailing empty tokens (common when exporting from Sheets with extra columns)
        while tokens and tokens[-1] == '':
            tokens.pop()

        # Skip fully-empty rows
        if not tokens:
            continue

        normalized_rows.append(tokens)

    if not normalized_rows:
        raise ValueError("CSV file contains no numeric rows")

    # Heuristic: detect header row (Google Sheets often exports with a header of non-numeric labels)
    first_row = normalized_rows[0]
    non_numeric_count = sum(1 for t in first_row if not _is_int_token(t))
    if non_numeric_count >= (len(first_row) + 1) // 2:
        # If at least half of first row tokens aren't numeric, assume it's a header and drop it
        normalized_rows = normalized_rows[1:]
        if not normalized_rows:
            raise ValueError("CSV contained only a header row and no numeric data")

    # Convert tokens to integers where possible; raise if non-integer token remains
    converted = []
    for r in normalized_rows:
        row_vals = []
        for t in r:
            if t == '':
                # skip empty in middle
                continue
            if not _is_int_token(t):
                raise ValueError(f"Non-integer value in CSV row after cleaning: {r}")
            row_vals.append(_to_int(t))
        if not row_vals:
            # skip rows that became empty after removing blanks
            continue
        converted.append(row_vals)

    if not converted:
        raise ValueError("CSV file contains no valid numeric rows after cleaning")

    # Ensure a consistent column count across rows. Google Sheets exports sometimes produce
    # rows with variable lengths due to trailing blank columns. We trim or salvage where reasonable.
    counts = [len(r) for r in converted]
    if len(set(counts)) != 1:
        # Attempt to salvage: trim rows to the most common length when it seems reasonable.
        from collections import Counter
        counter = Counter(counts)
        most_common_len, freq = counter.most_common(1)[0]

        # If the most common length occurs for a majority of the rows we will trim others to that length.
        if freq >= (len(converted) // 2 + 1) and most_common_len >= 3:
            # Trim longer rows and drop rows that are shorter than the most_common_len (they likely are incomplete)
            salvaged = []
            for r in converted:
                if len(r) >= most_common_len:
                    salvaged.append(r[:most_common_len])
                else:
                    # If a row is shorter than the common length, skip it as likely a partial/empty export row
                    continue
            if not salvaged:
                raise ValueError(f"Unable to salvage CSV rows to consistent length; row lengths: {counts}")
            converted = salvaged
        else:
            raise ValueError(f"Inconsistent row lengths in CSV after cleaning: {counts}")

    # (already validated above) return the cleaned numeric array

    return np.array(converted, dtype=int)


def generate_dodecaphonic_matrix(seed_row):
    """Generate a 12x12 dodecaphonic (12-tone) matrix given an initial 12-element seed row.

    The algorithm constructs the matrix so that the top row is the prime form
    (the provided seed). The first column is the inversion of the seed around
    the first element; each matrix element is produced as (I_r + interval_c) % 12
    where interval_c = (seed[c] - seed[0]) % 12 and I_r = (2*seed[0] - seed[r]) % 12.
    """
    if len(seed_row) != 12:
        raise ValueError('Seed row must contain exactly 12 elements')
    if set(seed_row) != set(range(12)):
        raise ValueError('Seed row must be a permutation of 0..11 (each pitch-class appears once)')

    p0 = seed_row[0]
    # intervals from first element for each column
    intervals = [((v - p0) % 12) for v in seed_row]

    # compute first column (inversion around p0)
    inv_col = [((2 * p0 - v) % 12) for v in seed_row]

    # build matrix: M[r][c] = (inv_col[r] + intervals[c]) % 12
    matrix = np.zeros((12, 12), dtype=int)
    for r in range(12):
        for c in range(12):
            matrix[r, c] = (inv_col[r] + intervals[c]) % 12

    return matrix


def prompt_user_for_matrix():
    print("Enter matrix rows one per line using spaces or commas between pitch-class numbers (0-11).")
    print("When finished enter an empty line. Example row: 3 7 10 1 6")

    rows = []
    while True:
        try:
            line = input(f"Row {len(rows)+1}: ").strip()
        except EOFError:
            # treat EOF like a blank line
            line = ''
        if line == '':
            break
        rows.append(line)

    if not rows:
        raise ValueError("No rows were entered")

    return parse_matrix_from_string(';'.join(rows))


def prompt_user_for_seed():
    print("Enter initial 12-element seed row as comma-separated pitch-classes (0..11). Example: 0,6,9,4,5,11,7,1,8,10,3,2")
    while True:
        try:
            line = input("Seed row: ").strip()
        except EOFError:
            line = ''
        if not line:
            raise ValueError('No seed provided')

        tokens = [t.strip() for t in line.replace(',', ' ').split() if t.strip()]
        try:
            nums = [int(x) for x in tokens]
        except Exception:
            print('Invalid input: seed must contain integer values 0..11 separated by commas or spaces. Try again.')
            continue

        if len(nums) != 12:
            print('Seed must contain exactly 12 values. Try again.')
            continue
        if set(nums) != set(range(12)):
            print('Seed must be a permutation of 0..11 (each pitch-class once). Try again.')
            continue

        return generate_dodecaphonic_matrix(nums)


def validate_matrix(matrix):
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Matrix must be a numpy.ndarray")
    if matrix.ndim != 2:
        raise ValueError("Matrix must be 2-dimensional (rows, columns)")
    if matrix.shape[1] < 3 and matrix.shape[0] < 3:
        raise ValueError("Matrix must have at least 3 columns OR at least 3 rows to find triads (you need adjacent triples).")
    if not np.all((matrix >= 0) & (matrix <= 11)):
        raise ValueError("All pitch-class values must be integers between 0 and 11 (inclusive).")

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Search a pitch-class matrix for adjacent major and minor triads (0=C, 1=C#, ..., 11=B)'
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--matrix', '-m', help='Matrix as a string; rows separated by ";" and values by spaces or commas. Example: "3 7 10; 8 11 2"')
    group.add_argument('--file', '-f', help='Path to a file containing the matrix. Each row on its own line; values separated by spaces or commas.')

    # Seed-based generation: accept an initial row (12 pitch-classes) and build full 12x12 matrix
    group.add_argument('--seed', '-s', help='Initial 12-element row as comma-separated values (e.g. "0,6,9,...")')
    group.add_argument('--random', '-r', nargs='?', const='RANDOM', help='Create a 12-element seed row from a random permutation. Optionally pass an integer seed to make it deterministic: --random 42')
    group.add_argument('--seed-file', help='Path to a file containing the initial row (one row, comma- or space-separated)')
    group.add_argument('--interactive', '-i', action='store_true', help='Prompt to enter matrix rows interactively')

    parser.add_argument('--example', action='store_true', help='Use the built-in example matrix (for quick testing)')
    parser.add_argument('--wrap', '-w', action='store_true', help='Enable wrap-around scanning of rows and columns (treat rows/cols as cyclic)')
    parser.add_argument('--note-names', '-n', action='store_true', help='Show matrix and triad output using note names (C,C#,D,...) instead of numbers')
    parser.add_argument('--transpositions', '-t', action='store_true', help='Show an explicit transposition row (T:) above the 12x12 headers')
    parser.add_argument('--no-inversions', action='store_true', help='Disable detection of triads in inversions (inversions enabled by default)')
    parser.add_argument('--retrogrades', action='store_true', help='Enable printing of retrograde (R) and retrograde-inversion (RI) labels in the 12x12 header/rows')
    parser.add_argument('--export-file', '-e', help='Write the matrix to a file (plain text, no headers). Honors --note-names to write note-names instead of numbers')
    parser.add_argument('--export-csv', help='Write the matrix to a CSV file (no headers). Each row is a matrix row; last two rows: Major triads,Minor triads')
    parser.add_argument('--export-heatmap', help='Write a 12x12 tonal-cube slice heatmap PNG for this matrix (single-matrix runs only)')
    parser.add_argument('--heatmap-z', type=int, default=0, help='For --export-heatmap: Z-depth offset added to the 12x12 matrix values mod 12 (default: 0)')
    parser.add_argument('--heatmap-labels', choices=['numbers', 'pitches'], default='numbers', help='For --export-heatmap: annotate cells with numbers (0-11) or pitch names (C,C#,D,Eb,...)')
    parser.add_argument('--export-sql-db', help='Write run summaries into a MySQL table (default: Tonality_Test). URL like mysql://user:pass@host:3306/12Tone')
    parser.add_argument('--export-sql-table', default='Tonality_Test', help='SQL table name for --export-sql-db (default: Tonality_Test)')
    parser.add_argument('--quiet', action='store_true', help='Suppress matrix/triad/summary printing (errors still print)')
    parser.add_argument('--progress', action='store_true', help='Print progress updates to stderr (useful with --random-repeats)')

    parser.add_argument('--describe-consonance', action='store_true', help='Include a musical, row/column-referenced text description in the consonance summary')

    parser.add_argument('--search-best', type=int, help='Randomly sample N candidates and keep only the top-K by consonance score (best-only mode)')
    parser.add_argument('--top-k', type=int, default=100, help='When using --search-best, keep the top K results (default: 100)')
    parser.add_argument('--jobs', type=int, default=1, help='When using --search-best, run with this many worker processes (default: 1). Use 0 to auto-pick ~90%% of CPU cores.')
    parser.add_argument('--throttle-ms', type=float, default=0.0, help='When using --search-best, sleep this many milliseconds periodically in each worker to reduce sustained CPU/heat (default: 0)')
    parser.add_argument('--throttle-every', type=int, default=500, help='When using --search-best with --throttle-ms, sleep every N candidates per worker (default: 500)')
    parser.add_argument('--heartbeat-seconds', type=float, default=5.0, help='When using --search-best with --progress, print a per-worker heartbeat at this interval (seconds). Use 0 to disable (default: 5.0)')
    parser.add_argument('--best-only-table', default='Best_Only', help='SQL table name to write best-only winners into (default: Best_Only)')
    parser.add_argument('--best-only-reset', action='store_true', help='When using --search-best with --export-sql-db, truncate the best-only table before inserting winners')
    parser.add_argument('--best-only-live-db-seconds', type=float, default=0.0, help='When using --search-best with --export-sql-db, periodically snapshot the evolving global top-K into the best-only table every N seconds. Use 0 to disable (default: 0)')
    parser.add_argument('--random-repeats', '-R', type=int, default=1, help='When using --random without a numeric seed, repeat generation this many times (creates indexed export files)')

    args = parser.parse_args()

    # Disallow heatmap export for batch modes.
    if args.export_heatmap:
        if args.search_best is not None:
            print("Error: --export-heatmap is not allowed with --search-best")
            return 1
        if args.random == 'RANDOM' and (args.random_repeats and args.random_repeats > 1):
            print("Error: --export-heatmap is not allowed with batch runs (--random-repeats > 1)")
            return 1

    # --- Best-only search mode (top-K random sampling) ---
    if args.search_best is not None:
        if int(args.search_best) <= 0:
            print("Error: --search-best must be > 0")
            return 1
        if int(args.top_k) <= 0:
            print("Error: --top-k must be > 0")
            return 1

        wrap = bool(args.wrap)
        allow_inversions = bool(not args.no_inversions)

        n = int(args.search_best)
        top_k = int(args.top_k)
        jobs_arg = int(args.jobs or 1)
        if jobs_arg == 0:
            import os as _os

            cores = int(_os.cpu_count() or 1)
            jobs = max(1, int(cores * 0.9))
            if cores > 1:
                jobs = min(jobs, cores - 1)
        else:
            jobs = max(1, jobs_arg)

        # Run workers (each returns a local top-K heap)
        throttle_every = int(args.throttle_every or 0)
        throttle_ms = float(args.throttle_ms or 0.0)

        heartbeat_seconds = float(args.heartbeat_seconds or 0.0)
        live_db_seconds = float(args.best_only_live_db_seconds or 0.0)
        if live_db_seconds < 0:
            print("Error: --best-only-live-db-seconds must be >= 0")
            return 1
        if live_db_seconds > 0 and not args.export_sql_db:
            print("Error: --best-only-live-db-seconds requires --export-sql-db")
            return 1

        # Activity monitor: estimate runtime with a quick calibration pass on this machine.
        if args.progress:
            try:
                import os as _os
                import platform as _platform
                import secrets as _secrets
                import time as _time

                cores = int(_os.cpu_count() or 1)
                calib_n = min(100, max(10, n // 100000))
                t_cal0 = _time.perf_counter()
                for _ in range(int(calib_n)):
                    seed_int = _secrets.randbits(63)
                    _evaluate_random_seed(seed_int, wrap=wrap, allow_inversions=allow_inversions)
                t_cal1 = _time.perf_counter()

                per_candidate = (t_cal1 - t_cal0) / float(calib_n)
                est_compute = (per_candidate * float(n)) / float(jobs)
                est_throttle = 0.0
                if throttle_ms > 0.0 and throttle_every > 0:
                    # each worker sleeps roughly once per throttle_every candidates
                    sleeps_per_worker = (float(n) / float(jobs)) / float(throttle_every)
                    est_throttle = sleeps_per_worker * (throttle_ms / 1000.0)
                est_total = est_compute + est_throttle
                finish_epoch = _time.time() + est_total
                finish_local = _time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime(finish_epoch))

                print(
                    "monitor: "
                    f"platform={_platform.platform()} py={_platform.python_version()} "
                    f"cores={cores} jobs={jobs} n={n} top_k={top_k} "
                    f"throttle={throttle_ms}ms/{throttle_every} heartbeat={heartbeat_seconds}s "
                    f"calib={calib_n} per_candidate={per_candidate:.4f}s "
                    f"est_total={_format_duration_seconds(est_total)} finish~{finish_local}",
                    file=sys.stderr,
                )
            except Exception as ex:
                print(f"monitor: calibration failed: {ex}", file=sys.stderr)

        worker_payloads = []
        base = n // jobs
        extra = n % jobs
        updates_queue = None
        updates_manager = None
        if live_db_seconds > 0:
            import multiprocessing as _mp

            ctx = _mp.get_context('spawn')
            # Under the 'spawn' start method, plain Queue objects cannot be pickled into
            # worker payloads. Use a Manager().Queue() proxy instead.
            updates_manager = ctx.Manager()
            updates_queue = updates_manager.Queue()

        for i in range(jobs):
            chunk = base + (1 if i < extra else 0)
            worker_payloads.append(
                (
                    chunk,
                    top_k,
                    wrap,
                    allow_inversions,
                    throttle_every,
                    throttle_ms,
                    heartbeat_seconds,
                    i + 1,
                    bool(args.progress),
                    updates_queue,
                )
            )

        def _merge_into_global_heap(global_heap, score_int: int, seed_int: int, top_k_int: int):
            item = (int(score_int), int(seed_int), None)
            if len(global_heap) < top_k_int:
                heapq.heappush(global_heap, item)
                return True
            if item[0] > global_heap[0][0] or (item[0] == global_heap[0][0] and item[1] > global_heap[0][1]):
                heapq.heapreplace(global_heap, item)
                return True
            return False

        def _snapshot_best_only_table(db_cursor, table_name: str, winners_seeds_scores):
            # Rewrite the table with the current winners (ranked). Requires `--best-only-reset`.
            db_cursor.execute(f"TRUNCATE TABLE `{table_name}`")
            for rank, (score_int, seed_int) in enumerate(winners_seeds_scores, start=1):
                mat, c = _evaluate_random_seed(seed_int, wrap=wrap, allow_inversions=allow_inversions)
                _mysql_insert_run(
                    db_cursor,
                    c,
                    mat,
                    run_index=int(rank),
                    seed=int(seed_int),
                    wrap=wrap,
                    allow_inversions=allow_inversions,
                    table_name=table_name,
                )

        heaps = []
        global_heap_live = []
        last_snapshot = None
        db_conn_live = None
        db_cursor_live = None

        if live_db_seconds > 0:
            if not args.best_only_reset:
                print("Error: --best-only-live-db-seconds requires --best-only-reset (so snapshots can safely rewrite the table)")
                return 1
            try:
                db_conn_live = _mysql_connect(args.export_sql_db)
                db_cursor_live = db_conn_live.cursor()
                _mysql_ensure_table(db_cursor_live, table_name=args.best_only_table)
                # Clear immediately so the table reflects the current run even before the first snapshot.
                db_cursor_live.execute(f"TRUNCATE TABLE `{args.best_only_table}`")
            except Exception as ex:
                print(f"Error connecting to DB for live best-only snapshots {_redact_db_url(args.export_sql_db)}: {ex}")
                return 1

        try:
            if jobs == 1:
                # Single process: just run the worker directly.
                heaps = [_best_only_worker(worker_payloads[0])]
            else:
                import multiprocessing as _mp
                import time as _time

                ctx = _mp.get_context('spawn')
                with ctx.Pool(processes=jobs) as pool:
                    async_results = [pool.apply_async(_best_only_worker, (payload,)) for payload in worker_payloads]
                    completed = 0
                    last_snapshot = _time.time()

                    while completed < len(async_results):
                        # Drain candidate updates from workers
                        if updates_queue is not None:
                            drained_any = False
                            while True:
                                try:
                                    s, seed_int = updates_queue.get_nowait()
                                except Exception:
                                    break
                                drained_any = True
                                _merge_into_global_heap(global_heap_live, int(s), int(seed_int), top_k)

                            if drained_any and live_db_seconds > 0 and db_cursor_live is not None:
                                now = _time.time()
                                if (now - last_snapshot) >= live_db_seconds:
                                    last_snapshot = now
                                    winners_live = sorted(global_heap_live, key=lambda x: (x[0], x[1]), reverse=True)
                                    winners_seeds_scores = [(int(s), int(seed)) for (s, seed, _none) in winners_live]
                                    try:
                                        _snapshot_best_only_table(db_cursor_live, args.best_only_table, winners_seeds_scores)
                                        if args.progress:
                                            max_s = winners_seeds_scores[0][0] if winners_seeds_scores else 0
                                            min_s = winners_seeds_scores[-1][0] if winners_seeds_scores else 0
                                            print(
                                                f"live-db: snapshot top {len(winners_seeds_scores)} max={max_s} min={min_s}",
                                                file=sys.stderr,
                                            )
                                    except Exception as ex:
                                        print(f"live-db: snapshot failed: {ex}", file=sys.stderr)

                        # Check for completed workers
                        new_completed = 0
                        for r in async_results:
                            if r.ready():
                                new_completed += 1
                        if new_completed != completed:
                            completed = new_completed
                            if args.progress:
                                print(f"progress: workers complete {completed}/{jobs}", file=sys.stderr)

                        _time.sleep(0.05)

                    # Collect all heaps
                    for r in async_results:
                        heaps.append(r.get())
        except KeyboardInterrupt:
            # If interrupted, best-effort snapshot current winners if live updates are enabled.
            if live_db_seconds > 0 and db_cursor_live is not None:
                try:
                    import time as _time

                    # Drain any remaining queued updates
                    if updates_queue is not None:
                        while True:
                            try:
                                s, seed_int = updates_queue.get_nowait()
                            except Exception:
                                break
                            _merge_into_global_heap(global_heap_live, int(s), int(seed_int), top_k)

                    winners_live = sorted(global_heap_live, key=lambda x: (x[0], x[1]), reverse=True)
                    winners_seeds_scores = [(int(s), int(seed)) for (s, seed, _none) in winners_live]
                    _snapshot_best_only_table(db_cursor_live, args.best_only_table, winners_seeds_scores)
                    print(
                        f"live-db: wrote snapshot on interrupt (top {len(winners_seeds_scores)})",
                        file=sys.stderr,
                    )
                except Exception as ex:
                    print(f"live-db: failed to snapshot on interrupt: {ex}", file=sys.stderr)
            return 130
        finally:
            if updates_manager is not None:
                try:
                    updates_manager.shutdown()
                except Exception:
                    pass
            if db_conn_live is not None:
                try:
                    db_conn_live.close()
                except Exception:
                    pass

        # Merge local heaps into a global top-K (authoritative)
        global_heap = []
        for h in heaps:
            for item in h:
                if len(global_heap) < top_k:
                    heapq.heappush(global_heap, item)
                else:
                    if item[0] > global_heap[0][0] or (item[0] == global_heap[0][0] and item[1] > global_heap[0][1]):
                        heapq.heapreplace(global_heap, item)

        winners = sorted(global_heap, key=lambda x: (x[0], x[1]), reverse=True)

        if args.progress or args.export_sql_db:
            max_score = winners[0][0] if winners else 0
            min_score = winners[-1][0] if winners else 0
            print(f"best-only: final top {len(winners)} max={max_score} min={min_score}", file=sys.stderr)

        # Optionally export winners to a dedicated best-only table
        if args.export_sql_db:
            try:
                db_conn = _mysql_connect(args.export_sql_db)
                db_cursor = db_conn.cursor()
                _mysql_ensure_table(db_cursor, table_name=args.best_only_table)
                if args.best_only_reset:
                    db_cursor.execute(f"TRUNCATE TABLE `{args.best_only_table}`")

                for rank, (score, seed_int, summary) in enumerate(winners, start=1):
                    # Recreate matrix for this seed for shape; summary already contains fields.
                    mat, c = _evaluate_random_seed(seed_int, wrap=wrap, allow_inversions=allow_inversions)
                    # Use run_index as rank in the best-only table.
                    _mysql_insert_run(
                        db_cursor,
                        c,
                        mat,
                        run_index=int(rank),
                        seed=int(seed_int),
                        wrap=wrap,
                        allow_inversions=allow_inversions,
                        table_name=args.best_only_table,
                    )

                db_conn.close()
            except Exception as ex:
                print(f"Error exporting best-only winners to DB {_redact_db_url(args.export_sql_db)}: {ex}")
                return 1

        if not args.quiet:
            print(f"Best-only search complete: sampled {n} candidates; top {len(winners)}")
            for rank, (score, seed_int, _summary) in enumerate(winners[: min(10, len(winners))], start=1):
                print(f"  #{rank}: score={score} seed={seed_int}")

        return 0

    # Built-in example matrix (kept for convenience)
    example_matrix = np.array([
        [3, 7, 10, 1, 6],
        [8, 11, 2, 4, 9],
        [5, 0, 4, 9, 11],
        [1, 6, 9, 10, 2]
    ])

    matrix = None
    run_seed_int = None
    seed_title = None
    try:
        if args.example:
            matrix = example_matrix
        elif args.matrix:
            matrix = parse_matrix_from_string(args.matrix)
        elif args.file:
            matrix = parse_matrix_from_file(args.file)
        elif args.seed or args.seed_file:
            # Read the seed row from inline or file and generate the full 12x12 matrix
            seed = None
            if args.seed:
                seed = args.seed
            else:
                # read seed file
                with open(args.seed_file, 'r', encoding='utf-8') as fh:
                    seed = fh.read().strip()

            # Accept comma or space separated inline seed row
            seed_tokens = [t.strip() for t in seed.replace(',', ' ').split() if t.strip()]
            try:
                seed_nums = [int(x) for x in seed_tokens]
            except Exception:
                raise ValueError('Seed row must contain integer pitch-class values (0-11)')

            if len(seed_nums) != 12:
                raise ValueError('Seed row must contain exactly 12 pitch-class integers (a permutation of 0..11)')
            if set(seed_nums) != set(range(12)):
                raise ValueError('Seed row must be a permutation of 0..11 (each pitch-class appears once)')

            matrix = generate_dodecaphonic_matrix(seed_nums)
            seed_title = "seed-row"
        elif args.random is not None:
            # auto-generate a 12-element seed row either deterministically (if an integer
            # seed was provided) or using system randomness when no numeric seed is given.
            import random as _random
            import secrets as _secrets

            random_arg = args.random
            if random_arg == 'RANDOM':
                # choose a high-entropy integer seed for non-deterministic generation.
                # Use 63-bit so it safely fits in signed BIGINT when exporting to SQL.
                seed_int = _secrets.randbits(63)
            else:
                try:
                    seed_int = int(random_arg)
                except Exception:
                    raise ValueError('When providing --random with a value it must be an integer seed')

            run_seed_int = int(seed_int)
            seed_title = str(run_seed_int)

            rnd = _random.Random(seed_int)
            seed_nums = list(range(12))
            rnd.shuffle(seed_nums)
            matrix = generate_dodecaphonic_matrix(seed_nums)
        elif args.interactive or (not any([args.matrix, args.file, args.example])):
            # If nothing provided: prompt interactively. Ask whether they want to provide
            # a 12-element seed row or a full matrix entry.
            try:
                pick = None
                try:
                    pick = input('Would you like to enter an initial 12-element seed row to build the full 12x12 matrix? (y/N): ').strip().lower()
                except EOFError:
                    pick = 'n'

                if pick and pick[0] == 'y':
                    matrix = prompt_user_for_seed()
                else:
                    matrix = prompt_user_for_matrix()
            except Exception as e:
                # Fallback to example if user's interactive entry fails
                print(f"Input error: {e}. Falling back to example matrix.")
                matrix = example_matrix

        validate_matrix(matrix)
    except Exception as e:
        print(f"Error preparing matrix: {e}")
        return 1

    if not args.quiet:
        print("🔎 Searching Matrix:")
    # If this is a 12x12 dodecaphonic matrix, show conventional headers:
    # rows labeled P0..P11 (primes) and columns labeled I0..I11 (inversions).
    def _matrix_lines_with_headers(mat, with_names: bool, show_transpositions: bool = False, show_retrogrades: bool = False):
        pitch_names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
        rows, cols = mat.shape
        lines = []
        if rows == 12 and cols == 12:
            # header line — inversion labels (columns) use transposition numbers from the top row
            header = [f"I{int(mat[0, c])}" for c in range(cols)]
            # RI header uses bottom-row values (retrograde inversion transposition numbers)
            ri_header = [f"RI{int(mat[-1, c])}" for c in range(cols)]

            if with_names:
                # Optionally print transpositions (numeric offsets of the top-row) before headers
                if show_transpositions:
                    trans_nums = [str(int(mat[0, c])) for c in range(cols)]
                    lines.append('T:   ' + ' '.join(f"{n:>5}" for n in trans_nums))

                # print column inversion header and RI header
                lines.append('     ' + ' '.join(f"{h:>6}" for h in header))
                # RI header moved to footer below the rows (see end of block)

                # column pitch names (use top row values)
                col_note_names = [pitch_names[int(mat[0, c]) % 12] for c in range(cols)]
                lines.append('     ' + ' '.join(f"{n:>6}" for n in col_note_names))

                # row lines with P-labels on the left and R-label at the end of each row (retrograde)
                for r in range(rows):
                    p_num = int(mat[r, 0])
                    r_num = int(mat[r, -1])
                    # include retrograde pairing in the row label when requested
                    if show_retrogrades:
                        row_label = f"P{p_num}/R{r_num}:"
                    else:
                        row_label = f"P{p_num}:"
                    row_entries = [pitch_names[int(mat[r, c]) % 12] for c in range(cols)]
                    lines.append(f"{row_label:>8} " + ' '.join(f"{e:>6}" for e in row_entries))
                # After printing all rows, optionally print the RI footer (retrograde inversions) under each column
                if show_retrogrades:
                    ri_footer = [f"RI{int(mat[-1, c])}" for c in range(cols)]
                    lines.append('     ' + ' '.join(f"{h:>6}" for h in ri_footer))
            else:
                if show_transpositions:
                    trans_nums = [str(int(mat[0, c])) for c in range(cols)]
                    lines.append('T:   ' + ' '.join(f"{n:>5}" for n in trans_nums))
                lines.append('     ' + ' '.join(f"{h:>6}" for h in header))
                # RI header moved to footer below the rows (see end of block)
                for r in range(rows):
                    p_num = int(mat[r, 0])
                    r_num = int(mat[r, -1])
                    # numeric rows include retrograde pairing in the row label when requested
                    if show_retrogrades:
                        row_label = f"P{p_num}/R{r_num}:"
                    else:
                        row_label = f"P{p_num}:"
                    row_entries = [str(int(mat[r, c])) for c in range(cols)]
                    lines.append(f"{row_label:>8} " + ' '.join(f"{e:>6}" for e in row_entries))
                # After printing all numeric rows, optionally show the RI footer
                if show_retrogrades:
                    ri_footer = [f"RI{int(mat[-1, c])}" for c in range(cols)]
                    lines.append('     ' + ' '.join(f"{h:>6}" for h in ri_footer))
        else:
            # fallback: simple printing (name or number)
            if with_names:
                name_matrix = np.vectorize(lambda x: pitch_names[int(x) % 12])(mat)
                for row in name_matrix:
                    lines.append(' '.join(str(x) for x in row))
            else:
                lines.append(str(mat))
        return lines

    def _matrix_plain_lines(mat, with_names: bool = False):
        """Return matrix rows as plain text lines (no headers).

        - when with_names is True return note names (C,C#,D,...)
        - otherwise return numeric rows (space separated)
        """
        pitch_names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
        rows, cols = mat.shape
        lines = []
        if with_names:
            for r in range(rows):
                row_entries = [pitch_names[int(mat[r, c]) % 12] for c in range(cols)]
                lines.append(' '.join(str(x) for x in row_entries))
        else:
            for r in range(rows):
                row_entries = [str(int(mat[r, c])) for c in range(cols)]
                lines.append(' '.join(row_entries))
        return lines

    # If the user requested multiple non-deterministic random matrices, handle that loop first.
    if args.random == 'RANDOM' and (args.random_repeats and args.random_repeats > 1):
        import random as _random
        import secrets as _secrets
        import csv as _csv

        db_conn = None
        db_cursor = None
        if args.export_sql_db:
            try:
                db_conn = _mysql_connect(args.export_sql_db)
                db_cursor = db_conn.cursor()
                _mysql_ensure_table(db_cursor, table_name=args.export_sql_table)
            except Exception as ex:
                print(f"Error connecting to DB {_redact_db_url(args.export_sql_db)}: {ex}")
                return 1

        repeats = int(args.random_repeats)
        for idx in range(1, repeats + 1):
            # Use 63-bit so it safely fits in signed BIGINT when exporting to SQL.
            seed_int = _secrets.randbits(63)
            rnd = _random.Random(seed_int)
            seed_nums = list(range(12))
            rnd.shuffle(seed_nums)
            mat = generate_dodecaphonic_matrix(seed_nums)

            if args.progress:
                print(f"progress: {idx}/{repeats} (seed {seed_int})", file=sys.stderr)

            found = find_triads_in_matrix(mat, wrap=args.wrap, allow_inversions=(not args.no_inversions))
            found_dyads = find_consonant_dyads(mat, wrap=args.wrap)
            consonance = consonance_factor(
                mat,
                wrap=args.wrap,
                allow_inversions=(not args.no_inversions),
                triad_results=found,
                dyad_results=found_dyads,
            )
            major_count = consonance["major_triads"]
            minor_count = consonance["minor_triads"]

            # prepare export filenames (if requested) with index suffix
            if args.export_file:
                base = args.export_file
                import os as _os
                name, ext = _os.path.splitext(base)
                out_path = f"{name}_{idx}{ext}"
                try:
                    export_lines = _matrix_plain_lines(mat, with_names=args.note_names)
                    export_lines.append(f"Major triads: {major_count}")
                    export_lines.append(f"Minor triads: {minor_count}")
                    with open(out_path, 'w', encoding='utf-8') as ef:
                        ef.write('\n'.join(export_lines) + '\n')
                except Exception as ex:
                    print(f"Error writing export file '{out_path}': {ex}")
                    return 1

            if args.export_csv:
                base = args.export_csv
                import os as _os
                name, ext = _os.path.splitext(base)
                out_path = f"{name}_{idx}{ext}"
                try:
                    with open(out_path, 'w', newline='', encoding='utf-8') as cf:
                        writer = _csv.writer(cf)
                        for line in _matrix_plain_lines(mat, with_names=False):
                            writer.writerow([x for x in line.split()])
                        # metadata rows with seed and counts as integers
                        writer.writerow(['Seed', str(seed_int)])
                        writer.writerow(['Number of Major Triads', str(major_count)])
                        writer.writerow(['Number of Minor Triads', str(minor_count)])
                except Exception as ex:
                    print(f"Error writing CSV export file '{out_path}': {ex}")
                    return 1

            if db_cursor is not None:
                try:
                    _mysql_insert_run(
                        db_cursor,
                        consonance,
                        mat,
                        run_index=idx,
                        seed=int(seed_int),
                        wrap=bool(args.wrap),
                        allow_inversions=bool(not args.no_inversions),
                        table_name=args.export_sql_table,
                    )
                except Exception as ex:
                    print(f"Error inserting DB row (run {idx}): {ex}")
                    return 1

            # If no export requested, print the matrix and results for this iteration
            if (not args.quiet) and (not args.export_file and not args.export_csv and not args.export_sql_db):
                print(f"\n--- Random run {idx} (seed: {seed_int}) ---")
                for l in _matrix_lines_with_headers(mat, args.note_names, show_transpositions=args.transpositions, show_retrogrades=(args.retrogrades)):
                    print(l)
                found and print_results(found, show_note_names=args.note_names)

                print("\nConsonance Factor Summary:")
                print(f"  Score: {consonance['score']}")
                print(f"  Major triads: {consonance['major_triads']}")
                print(f"  Minor triads: {consonance['minor_triads']}")
                print(f"  Perfect fifth dyads (ic5): {consonance['perfect_fifth_dyads']}")
                print(f"  Octave/unison dyads (ic0): {consonance['octave_dyads']}")

                if args.describe_consonance:
                    print("\nConsonance Description:")
                    print(describe_consonance(mat, consonance, show_note_names=args.note_names))

        if db_conn is not None:
            try:
                db_conn.close()
            except Exception:
                pass

        # finished repeated runs
        return 0

    # Single-run path (existing behavior)
    if not args.quiet:
        for l in _matrix_lines_with_headers(matrix, args.note_names, show_transpositions=args.transpositions, show_retrogrades=(args.retrogrades)):
            print(l)

    # Run the search
    found_triads = find_triads_in_matrix(matrix, wrap=args.wrap, allow_inversions=(not args.no_inversions))
    found_dyads = find_consonant_dyads(matrix, wrap=args.wrap)

    consonance = consonance_factor(
        matrix,
        wrap=args.wrap,
        allow_inversions=(not args.no_inversions),
        triad_results=found_triads,
        dyad_results=found_dyads,
    )

    import csv as _csv

    if args.export_sql_db:
        try:
            db_conn = _mysql_connect(args.export_sql_db)
            db_cursor = db_conn.cursor()
            _mysql_ensure_table(db_cursor, table_name=args.export_sql_table)
            seed_value = None
            # Prefer the computed/assigned seed when available.
            if run_seed_int is not None:
                seed_value = int(run_seed_int)
            elif args.random is not None and args.random != 'RANDOM':
                try:
                    seed_value = int(args.random)
                except Exception:
                    seed_value = None
            _mysql_insert_run(
                db_cursor,
                consonance,
                matrix,
                run_index=None,
                seed=seed_value,
                wrap=bool(args.wrap),
                allow_inversions=bool(not args.no_inversions),
                table_name=args.export_sql_table,
            )
            db_conn.close()
        except Exception as ex:
            print(f"Error exporting to DB {_redact_db_url(args.export_sql_db)}: {ex}")
            return 1
    # If exporting requested, write the plain matrix rows and append counts of major/minor triads
    if args.export_file:
        try:
            export_lines = _matrix_plain_lines(matrix, with_names=args.note_names)
            # calculate counts
            major_count = sum(1 for t, *_ in found_triads if isinstance(t, str) and t.startswith('Major'))
            minor_count = sum(1 for t, *_ in found_triads if isinstance(t, str) and t.startswith('Minor'))
            export_lines.append(f"Major triads: {major_count}")
            export_lines.append(f"Minor triads: {minor_count}")
            with open(args.export_file, 'w', encoding='utf-8') as ef:
                ef.write('\n'.join(export_lines) + '\n')
        except Exception as ex:
            print(f"Error writing export file '{args.export_file}': {ex}")
            return 1

    if args.export_csv:
        try:
            major_count = sum(1 for t, *_ in found_triads if isinstance(t, str) and t.startswith('Major'))
            minor_count = sum(1 for t, *_ in found_triads if isinstance(t, str) and t.startswith('Minor'))
            # Determine seed for single-run: extract from seed_nums if available, else N/A
            seed_value = 'N/A'
            if run_seed_int is not None:
                seed_value = int(run_seed_int)
            elif args.random is not None and args.random != 'RANDOM':
                try:
                    seed_value = int(args.random)
                except Exception:
                    seed_value = 'N/A'
            
            with open(args.export_csv, 'w', newline='', encoding='utf-8') as cf:
                writer = _csv.writer(cf)
                for line in _matrix_plain_lines(matrix, with_names=False):
                    writer.writerow([x for x in line.split()])
                # metadata rows with seed and counts as integers
                writer.writerow(['Seed', str(seed_value)])
                writer.writerow(['Number of Major Triads', str(major_count)])
                writer.writerow(['Number of Minor Triads', str(minor_count)])
        except Exception as ex:
            print(f"Error writing CSV export file '{args.export_csv}': {ex}")
            return 1

    if args.export_heatmap:
        if seed_title is None:
            if run_seed_int is not None:
                seed_title = str(run_seed_int)
            elif args.random is not None and args.random != 'RANDOM':
                seed_title = str(args.random)
            else:
                seed_title = "unknown"
        try:
            export_tonal_cube_slice_heatmap(
                matrix,
                args.export_heatmap,
                z_depth=int(args.heatmap_z or 0),
                title=seed_title,
                label_mode=str(args.heatmap_labels or "numbers"),
            )
        except Exception as ex:
            print(f"Error writing heatmap '{args.export_heatmap}': {ex}")
            return 1

    # Output the answers
    if not args.quiet:
        print_results(found_triads, show_note_names=args.note_names)

        print("\nConsonance Factor Summary:")
        print(f"  Score: {consonance['score']}")
        print(f"  Major triads: {consonance['major_triads']}")
        print(f"  Minor triads: {consonance['minor_triads']}")
        print(f"  Perfect fifth dyads (ic5): {consonance['perfect_fifth_dyads']}")
        print(f"  Octave/unison dyads (ic0): {consonance['octave_dyads']}")

        if args.describe_consonance:
            print("\nConsonance Description:")
            print(describe_consonance(matrix, consonance, show_note_names=args.note_names))

        if consonance.get('expected_balanced_triads') and consonance['major_triads'] != consonance['minor_triads']:
            print("  ⚠️ Major/minor triad counts are unbalanced; research expectation is equal counts for 12x12 matrices.")


if __name__ == '__main__':
    raise SystemExit(main())
