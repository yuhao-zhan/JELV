import argparse
import os


def apply_edits_to_sentence(sentence_tokens, annotations):
    """
    Applies edits from annotations in reverse order to avoid index shifts.
    Each annotation is a tuple: (token_range, edit_type, correction).
    """
    tokens = list(sentence_tokens)  # make a mutable copy
    # Process from last to first
    for token_range, edit_type, correction in reversed(annotations):
        start, end = map(int, token_range.split())
        correction_tokens = correction.split() if correction and correction != '-NONE-' else []

        if start == end:
            # Insertion
            tokens[start:start] = correction_tokens
        elif correction_tokens:
            # Replacement
            tokens[start:end] = correction_tokens
        else:
            # Deletion
            del tokens[start:end]

    return " ".join(tokens)


def parse_m2_blocks(m2_path):
    """
    Parses an M2 file into a list of blocks.
    Each block is a dict with:
      - sentence_tokens: list of source tokens
      - annotations: list of (annotator_id, token_range, edit_type, correction)
    """
    blocks = []
    with open(m2_path, 'r', encoding='utf-8') as f:
        lines = [l.rstrip() for l in f]

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('S '):
            src = line[2:]
            sentence_tokens = src.split()
            i += 1
            anns = []
            # Collect following A- lines
            while i < len(lines) and lines[i].startswith('A '):
                parts = lines[i].split('|||')
                # parts[0]: 'A <start> <end>'
                _, token_range = parts[0].split(' ', 1)
                edit_type = parts[1]
                correction = parts[2]
                annotator_id = int(parts[-1])
                anns.append((annotator_id, token_range, edit_type, correction))
                i += 1
            blocks.append({'sentence_tokens': sentence_tokens, 'annotations': anns})
        else:
            i += 1

    return blocks


def main():
    input_m2 = "/data/zyh/Reference_expansion/hypo/m2/ABCN-dev/ABCN-dev-predicted-aligned.m2"
    min_annotators = 5
    blocks = parse_m2_blocks(input_m2)
    # Determine maximum annotator_id seen
    max_id = -1
    for blk in blocks:
        for ann in blk['annotations']:
            if ann[0] > max_id:
                max_id = ann[0]
    total_ann = min(max_id + 1, min_annotators)

    # Prepare output files
    out_dir = "/data/zyh/Reference_expansion/hypo/txt/ABCN"
    src_f = open(os.path.join(out_dir, 'src.txt'), 'w', encoding='utf-8')
    ann_files = []
    for aid in range(total_ann):
        fname = os.path.join(out_dir, f'annotator_{aid}.txt')
        ann_files.append(open(fname, 'w', encoding='utf-8'))

    # Track last written sentences for each annotator
    last_written = [None] * total_ann

    # Process each block
    for blk in blocks:
        tokens = blk['sentence_tokens']
        src_line = " ".join(tokens)
        src_f.write(src_line + '\n')

        # Group annotations by annotator_id
        by_id = {}
        for ann in blk['annotations']:
            aid, token_range, edit_type, correction = ann
            by_id.setdefault(aid, []).append((token_range, edit_type, correction))

        # Write per-annotator outputs
        for aid in range(total_ann):
            if aid in by_id:
                corrected = apply_edits_to_sentence(tokens, by_id[aid])
                ann_files[aid].write(corrected + '\n')
                last_written[aid] = corrected
            else:
                # No annotation: reuse last or fallback to source
                fallback = last_written[aid] if last_written[aid] is not None else src_line
                ann_files[aid].write(fallback + '\n')
                last_written[aid] = fallback

    # Close files
    src_f.close()
    for f in ann_files:
        f.close()
