"""Check MedExQA answer keys against the upstream expert explanations.

Upstream: bluesky333/MedExQA, revision 3a9f80d6de2c354956d7e86d323214b92d547d6e,
test/<specialty>_test.tsv (no header; columns: question, A, B, C, D,
explanation1, explanation2, answer). Download the five test TSVs into one folder, then:

    python data/check_medexqa_keys.py <tsv_dir> [--apply]

Items are matched to data/medexqa/test.jsonl by question text. For every item the
script reports the option that the row's own two explanations support, either
automatically (the option text appears verbatim in the explanations and no other
option does, or its word overlap leads by >= 0.25) or through a reviewed phrase in
REVIEWED that must occur in the explanations. In speech_pathologist_test.tsv the
supported option usually equals the key stored on the NEXT row (one-row offset).
CORRECTIONS lists every change; each one is asserted to be explanation-supported.
--apply writes the corrected keys to data/medexqa/test.jsonl and test_hard.jsonl.
"""
import csv, json, os, re, sys
from collections import Counter

SUBSETS = ['biomedical_engineer', 'clinical_laboratory_scientist', 'clinical_psychologist',
           'occupational_therapist', 'speech_pathologist']
STOP = {'the', 'a', 'an', 'of', 'and', 'or', 'in', 'to', 'is', 'for', 'with', 'by', 'on', 'are', 'as', 'that', 'it', 'its', 'be'}

# Rows where the automatic rule is undecided or misfires: (realidx) -> (option, phrase quoted from the explanations).
REVIEWED = {
    '0_speech_pathologist': ('D', 'reduction in pressure'),
    '1_speech_pathologist': ('B', 'higher pillow during sleep'),
    '2_speech_pathologist': ('D', 'limit caffeine consumption'),
    '4_speech_pathologist': ('B', 'existence and intensity of a voice tremor'),
    '11_speech_pathologist': ('B', 'picture description task for preschoolers'),
    '20_speech_pathologist': ('A', 'within the frontal association area'),
    '22_speech_pathologist': ('B', 'ensuring proper humidity levels is the correct one'),
    '29_speech_pathologist': ('D', 'persist even in'),
    '32_speech_pathologist': ('D', 'plotting changes in amplitude over time'),
    '33_speech_pathologist': ('A', 'align with the distinct needs and choices of the users'),
    '37_speech_pathologist': ('A', 'cranial nerve V (CN 5), primarily governs'),
    '40_speech_pathologist': ('D', 'can be managed with the yawn-sigh technique'),
    '42_speech_pathologist': ('D', 'stop the harmful vocal habits'),
    '60_speech_pathologist': ('D', 'the GCI is often greater than the GOI'),
    '67_speech_pathologist': ('D', 'invaluable in emergencies'),
    '71_speech_pathologist': ('B', 'delve into their feelings and perceptions'),
    '74_speech_pathologist': ('A', 'articulate newly acquired language structures'),
    '75_speech_pathologist': ('B', 'Pitch refers to the aspect of sound'),
    '76_speech_pathologist': ('D', 'reduced pauses between words'),
    '84_speech_pathologist': ('C', 'one-time vocal strain event'),
    '85_speech_pathologist': ('D', 'could indicate that speech therapy might be beneficial'),
    '87_speech_pathologist': ('B', "acknowledging the guardian's feelings"),
    '88_speech_pathologist': ('D', 'within the last 3 months, has a higher chance'),
    '91_speech_pathologist': ('C', "known as 'facilitation through presenting pre-stimulus.'"),
    '100_speech_pathologist': ('A', 'stuporous condition'),
    '103_speech_pathologist': ('D', 'Cerebral infarction, commonly referred to as an ischemic stroke'),
    '111_speech_pathologist': ('D', "incorporating both of a child's languages"),
    '117_speech_pathologist': ('D', 'following bilateral damage to the vocal folds'),
    '119_speech_pathologist': ('A', "Broca's aphasia and apraxia of speech are considered ideal for MIT"),
    '121_speech_pathologist': ('D', 'the thalamus plays a crucial role'),
    '125_speech_pathologist': ('D', 'most consistent with an embolic cerebral infarction'),
    '127_speech_pathologist': ('B', 'Anterograde amnesia refers to a memory impairment'),
    # Explanations contradict the key, but no upstream row supports a change: flagged only.
    '3_speech_pathologist': ('A', "subtest that focuses on 'Judging grammatical errors'"),
    '18_speech_pathologist': ('D', 'Conductive hearing loss occurs'),
    '30_speech_pathologist': ('A', 'The Mendelsohn maneuver is a swallowing exercise'),
    '56_speech_pathologist': ('C', 'A pacing board is an assistive tool'),
    # Automatic rule misfires here; the explanations support the released key.
    '47_speech_pathologist': ('D', 'tilts toward the unaffected side'),
    '32_clinical_laboratory_scientist': ('B', "'CV' refers to the Coefficient of Variation"),
    '252_clinical_laboratory_scientist': ('A', 'commonly occurs during severe bacterial infections'),
    '84_occupational_therapist': ('A', 'A FIM score of 4 denotes that the person needs minimal assistance'),
}


def tok(s):
    return set(re.findall(r'[a-z0-9]+', s.lower())) - STOP


def auto_support(row):
    ex = (row[5] + ' ' + row[6]).lower()
    sc = {k: (o.lower().strip(' .') in ex, len(tok(o) & tok(ex)) / max(1, len(tok(o)))) for k, o in zip('ABCD', row[1:5])}
    a, b = sorted(sc, key=lambda k: sc[k], reverse=True)[:2]
    if sc[a][0] and not sc[b][0]:
        return a
    if sc[a][0] == sc[b][0] and sc[a][1] - sc[b][1] >= 0.25:
        return a
    return None


def main(tsv_dir, apply=False):
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'medexqa')
    items = {}
    for line in open(os.path.join(here, 'test.jsonl'), encoding='utf-8'):
        r = json.loads(line)
        items[r['question'].strip()] = r
    corrections, stats = {}, Counter()
    print('realidx | released key | upstream key | next-row upstream key | supported | evidence')
    for s in SUBSETS:
        rows = list(csv.reader(open(os.path.join(tsv_dir, f'{s}_test.tsv'), encoding='utf-8'), delimiter='\t'))
        for i, row in enumerate(rows):
            item = items.get(row[0].strip())
            if item is None:
                continue  # source row 0 is not in the released test split
            stats[s, 'checked'] += 1
            rid, key = item['realidx'], item['answer_idx']
            nxt = rows[i + 1][7] if i + 1 < len(rows) else '-'
            if rid in REVIEWED:
                sup, phrase = REVIEWED[rid]
                assert phrase.lower() in (row[5] + ' ' + row[6]).lower(), (rid, phrase)
                ev = f'reviewed: "{phrase}"'
            elif s == 'speech_pathologist':
                sup, ev = auto_support(row), 'option text in explanations'
            else:
                sup, ev = None, ''
            if sup is None or sup == key:
                stats[s, 'undecided' if sup is None else 'key supported'] += 1
            elif sup == nxt and s == 'speech_pathologist':
                corrections[rid] = sup
                stats[s, 'changed'] += 1
                print(f'{rid} | {key} | {row[7]} | {nxt} | {sup} | {ev}')
            else:
                stats[s, 'flagged'] += 1
                print(f'FLAGGED (unchanged) {rid} | {key} | {row[7]} | {nxt} | {sup} | {ev}')
    print()
    for s in SUBSETS:
        print(s, {k: stats[s, k] for k in ('checked', 'key supported', 'changed', 'flagged', 'undecided')})
    print('total corrections:', len(corrections))
    if apply:
        for name in ('test.jsonl', 'test_hard.jsonl'):
            path = os.path.join(here, name)
            out = []
            for line in open(path, encoding='utf-8'):
                r = json.loads(line)
                if r['realidx'] in corrections:
                    r['answer_idx'] = corrections[r['realidx']]
                    r['answer'] = r['options'][r['answer_idx']]
                out.append(json.dumps(r))
            with open(path, 'w', encoding='utf-8', newline='\n') as f:
                f.write('\n'.join(out) + '\n')
    return corrections


if __name__ == '__main__':
    main(sys.argv[1], '--apply' in sys.argv)
