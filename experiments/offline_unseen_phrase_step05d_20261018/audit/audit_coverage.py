"""Independent Unicode arithmetic and observed-label coverage audit."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
from audit_sources import read,binding,check_binding,S3,S5C
HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def encode(text):
    result=[]
    for char in text:
        if char==" ":result.append(68);continue
        code=ord(char)-0xAC00;assert 0<=code<11172
        lead,rem=divmod(code,588);vowel,tail=divmod(rem,28)
        result.extend([lead+1,vowel+20])
        if tail:result.append(tail+40)
    return result

def coverage(values,seen):
    seen=set(seen);counts=Counter(values);observed=set(counts)&seen
    total=sum(counts.values());n=sum(counts[k] for k in observed)
    return {"occurrences":total,"seen_occurrences":n,"occurrence_coverage":n/total if total else None,"unique":len(counts),"seen_unique":len(observed),"unique_coverage":len(observed)/len(counts) if counts else None,"unseen":[{"unit":k,"occurrences":counts[k]} for k in sorted(set(counts)-seen)]}

def units(record):
    text=record["text"]
    return {"tokens_including_space":encode(text),"hangul_syllables_excluding_spaces":[c for c in text if 0xAC00<=ord(c)<=0xD7A3],"eojeols_exact_space_delimited":text.split()}

def inspect():
    path=BASE/"analysis/coverage_analysis.json";value=read(path)
    for source in value["sources"]+[value["analysis_code"]]:check_binding(source)
    source=read(S3/"results/cache_handoff.json")["records"]
    train=[r for r in source if r["kind"]=="phrase" and r["split"]=="train"]
    dev=[r for r in source if r["kind"]=="phrase" and r["split"]=="dev"]
    assert len(train)==12 and len(dev)==3
    train_saved={r["clip_id"]:r for r in read(S5C/"runs/seed42/evaluations/update_1680/evaluation_summary.json")["rows"] if r["condition"]=="normal"}
    dev_saved={r["clip_id"]:r for r in read(BASE/"runs/seed42/evaluation_summary.json")["rows"] if r["condition"]=="normal"}
    for record in train+dev:
        saved=(train_saved if record["split"]=="train" else dev_saved)[record["clip_id"]]
        assert encode(record["text"])==saved["reference_label_ids"]
    kinds=list(units(train[0]));train_units={k:[x for r in train for x in units(r)[k]] for k in kinds}
    overall={"vocabulary_representable_all":True}
    for kind in kinds:overall[kind]=coverage([x for r in dev for x in units(r)[kind]],train_units[kind])
    assert not overall["tokens_including_space"]["unseen"]
    assert value["overall_dev3"]==overall
    byid={r["clip_id"]:r for r in dev}
    for row in value["per_dev_clip"]:
        record=byid[row["clip_id"]];same=[r for r in train if r["speaker_id"]==record["speaker_id"]]
        assert row["speaker_id"]==record["speaker_id"] and row["reference_text"]==record["text"]
        assert row["vocabulary_representable"] is True and row["same_speaker_train_clip_count"]==len(same)==1
        for kind in kinds:assert row[kind]==coverage(units(record)[kind],train_units[kind])
        assert row["same_speaker_observed_token_coverage"]==coverage(encode(record["text"]),[t for r in same for t in encode(r["text"])])
        assert row["same_speaker_observed_eojeol_coverage"]==coverage(record["text"].split(),[w for r in same for w in r["text"].split()])
    assert value["vocabulary"]=={"total_size_including_blank":69,"reference_blank_allowed":False,"nonblank_symbol_inventory":68,"observed_train_unique_tokens":len(set(train_units["tokens_including_space"])),"observed_train_token_occurrences":len(train_units["tokens_including_space"])}
    assert value["train_counts"]=={"clips":12,"unique_phrase_texts":len({r["text"] for r in train}),"speakers":len({r["speaker_id"] for r in train}),"hangul_syllables":len(train_units["hangul_syllables_excluding_spaces"]),"unique_hangul_syllables":len(set(train_units["hangul_syllables_excluding_spaces"])),"eojeols":len(train_units["eojeols_exact_space_delimited"]),"unique_eojeols":len(set(train_units["eojeols_exact_space_delimited"]))}
    ps=defaultdict(set);sp=defaultdict(set)
    for r in train:ps[r["text"]].add(r["speaker_id"]);sp[r["speaker_id"]].add(r["text"])
    assert value["train_phrase_speaker_distribution"]==[{"text":t,"speaker_count":len(s),"speakers":sorted(s)} for t,s in sorted(ps.items())]
    assert value["train_speaker_phrase_distribution"]==[{"speaker_id":s,"unique_phrase_count":len(t),"texts":sorted(t)} for s,t in sorted(sp.items())]
    return {"schema":"offline-stage5d-independent-coverage-audit-v1","created_utc":datetime.now(timezone.utc).isoformat(),"status":"passed","coverage":binding(path),"independent_Hangul_arithmetic_label_roundtrips":15,"train":value["train_counts"],"vocabulary":value["vocabulary"],"dev_once_only":overall,"meaning":"Observed label membership only, not recognition coverage or a proved cause of failure.","producer_imports":0,"new_network_forward_calls":0,"optimizer_updates":0,"pixel_arrays_read":0,"checkpoint_tensor_loads":0}

if __name__=="__main__":
    result=inspect();output=HERE/"coverage_audit.json";assert not output.exists()
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"status":result["status"],"unicode_roundtrips":15,"dev_counts":{k:[v["seen_occurrences"],v["occurrences"]] for k,v in result["dev_once_only"].items() if isinstance(v,dict)}}))
