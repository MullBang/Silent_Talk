"""Derive the separately bounded addition runner without editing executed code."""
from __future__ import annotations
import ast
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent


class Style(ast.NodeTransformer):
    """Add explicit generic types and contextual rethrow to local functions."""
    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Preserve computations while adding repository-required diagnostics."""
        try:
            self.generic_visit(node)
            for arg in node.args.posonlyargs+node.args.args+node.args.kwonlyargs:
                if arg.annotation is None:
                    arg.annotation = ast.Name(id="Any", ctx=ast.Load())
            if node.returns is None:
                node.returns = ast.Name(id="Any", ctx=ast.Load())
            if not ast.get_docstring(node):
                node.body.insert(0, ast.Expr(value=ast.Constant(value=f"Execute {node.name} with explicit scope checks.")))
            doc, rest = node.body[0], node.body[1:]
            handler = ast.parse(f"try:\n    pass\nexcept Exception as exc:\n    exc.add_note('stage5j addition {node.name}')\n    raise").body[0].handlers[0]
            node.body = [doc, ast.Try(body=rest, handlers=[handler], orelse=[], finalbody=[])]
            return node
        except Exception as exc:
            exc.add_note("addition builder style")
            raise


def main() -> None:
    """Copy verified arithmetic and replace only declared data/schedule scopes."""
    try:
        original = (BASE.parent/"training/runner.py").read_text(encoding="utf-8")
        original = original.replace("import traceback\n", "import traceback\nfrom typing import Any\n")
        original = original.replace("stage5j_frozen_5i", "stage5j_added_frozen_5i")
        original = original.replace("offline-stage5j-budget-extension-plan-v1", "offline-stage5j-added-recordings-plan-v1")
        original = original.replace('require(bound(plan["old_data_plan"]) == bound(old_plan["data_plan"]), "Changed continuation data")',
            'require(bound(plan["old_data_plan"]) == bound(old_plan["data_plan"]), "Changed old continuation data")\n    bound(plan["data_plan"])')
        original = original.replace('data = source.Dataset(plan["old_data_plan"])', 'data = Dataset(plan)')
        original = original.replace('data.batch(tasks[42][0]["clip_ids"], "B26")', 'data.batch(tasks[42][0]["clip_ids"], "D_added")')
        original = original.replace('data.batch(tasks[42][1]["clip_ids"], "B26")', 'data.batch(tasks[42][1]["clip_ids"], "D_added")')
        original = original.replace('data.batch(maximum["clip_ids"], "B26")', 'data.batch(maximum["clip_ids"], "D_added")')
        original = original.replace('data.batch(task["clip_ids"], "B26")', 'data.batch(task["clip_ids"], "D_added")')
        original = original.replace('"seen_in_optimizer": sid in data.arms["B26"]', '"seen_in_optimizer": sid in data.arms["D_added"]')
        original = original.replace('"optimizer_ids": data.arms["B26"]', '"optimizer_ids": data.arms["D_added"]')
        original = original.replace('"B26_extended"', '"D_added"').replace('f"B26_extended_seed{seed}"', 'f"D_added_seed{seed}"')
        original = original.replace('require(len(rows) == 128, "Missing evaluation rows")', 'require(len(rows) == 4*len(data.rows), "Missing evaluation rows")')
        original = original.replace('"epoch": task["epoch"],', '"epoch": task["epoch"], "clip_epochs": task["clip_epochs"],')
        tree = ast.parse(original)
        replacements = ast.parse(REPLACEMENTS)
        names = {n.name for n in replacements.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        out = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                continue
            if isinstance(node, ast.FunctionDef) and node.name == "load_checkpoint":
                out.extend(replacements.body)
            out.append(node)
        tree.body = out
        tree = Style().visit(tree)
        ast.fix_missing_locations(tree)
        text = ast.unparse(tree)+"\n"
        text = text.replace('Bounded full-state continuation of frozen 5i B26, with unchanged inputs.',
                            'Bounded full-state continuation with separately screened new whole utterances.')
        with (BASE/"runner.py").open("x", encoding="utf-8") as stream:
            stream.write(text)
    except Exception as exc:
        exc.add_note("addition builder main")
        raise


REPLACEMENTS = '''
class Dataset(source.Dataset):
    """Keep frozen old32 and add only exact separately gated source rows."""
    def __init__(self, plan):
        self.original = source.Dataset(plan["old_data_plan"])
        self.plan = read(bound(plan["data_plan"]))
        require(self.plan["schema"] == "offline-stage5j-added-recording-data-v1", "Wrong addition data")
        self.rows = {r["clip_id"]: r for r in self.plan["records"]}
        old_rows = self.original.rows
        require(all(self.rows.get(sid) == row for sid,row in old_rows.items()), "Old source rows changed")
        self.new_ids = self.plan["new_clip_ids"]
        require(1 <= len(self.new_ids) <= 4 and len(self.new_ids) == len(set(self.new_ids)), "Wrong new source count")
        require(set(self.rows) == set(old_rows)|set(self.new_ids) and not set(old_rows)&set(self.new_ids), "Source scope changed")
        self.arms = dict(self.original.arms)
        self.arms["D_added"] = sorted(self.original.arms["B26"] + self.new_ids)
        require(self.arms == self.plan["arms"], "Arm allowlist changed")
        self.cohorts = dict(self.original.cohorts)
        self.cohorts["train_new_short"] = list(self.new_ids)
        require(self.cohorts == self.plan["cohorts"], "Cohort composition changed")
        self.overrides = {}
        self.dev = self.original.dev
        self.training = self.arms["D_added"]
        for binding in self.plan["source_bindings"]:
            bound(binding)
        gate = read(bound(self.plan["acceptance"]))
        require(gate["selected_clip_ids"] == self.new_ids and gate["human_approvals_added"] == 0, "Acceptance scope differs")
        selected = {r["clip_id"]:r for r in gate["selected"]}
        held = set(self.plan["held_parent_ids"])
        require(held == set(self.original.plan["held_parent_ids"]), "Held6 scope changed")
        old_recordings = {r["recording_group"] for r in old_rows.values()}
        old_parents = {r["parent_id"] for r in old_rows.values()}
        dev_texts = {old_rows[s]["text"] for s in self.dev}
        old_meta = [read(bound({"path":r["metadata_path"],"sha256":r["metadata_sha256"]})) for r in old_rows.values()]
        old_video = {m["source_members"]["video"]["sha256"] for m in old_meta}
        old_label = {m["source_members"]["label"]["sha256"] for m in old_meta}
        self.items = dict(self.original.items)
        new_recordings = set()
        for sid in self.new_ids:
            row = self.rows[sid]
            decision = selected[sid]
            require(decision["automatic_gate_passed"] and decision["technical_gate_passed"] and decision["spatial_sample_gate_passed"], "Ungated new source")
            require(row["human_approvals_added"] == 0 and row["provisional"] is True and row["training_ready"] is False,
                    "New source approval promoted")
            require(row["kind"] == "full_sentence" and row["split"] == "train" and row["technical_QC_passed"] and
                    row["display_neighbor_intersections"] == [] and row["boundary_followup_flag"] is False, "Source temporal/technical gate failed")
            require(row["recording_group"] not in old_recordings|new_recordings and row["parent_id"] not in old_parents|held and
                    row["text"] not in dev_texts, "New source recording/parent/text collision")
            new_recordings.add(row["recording_group"])
            meta = read(bound({"path":row["metadata_path"],"sha256":row["metadata_sha256"]}))
            require(meta["source_members"]["video"]["sha256"] not in old_video and meta["source_members"]["label"]["sha256"] not in old_label,
                    "New source original hash overlaps old sources")
            require(meta["new_internal_cut"] is False and meta["human_transcript_approved"] is False and
                    meta["human_visual_boundary_approved"] is False and meta["new_original_pixels_human_approved"] is False,
                    "New source is recut or human approval changed")
            self.items[sid] = self._load(row)

def expected_continuation(data, seed):
    """Flatten freshly shuffled expanded epochs; use exactly3640clip exposures."""
    ids = sorted(data.arms["D_added"])
    rng = random.Random(seed)
    stream = []
    epoch = 0
    while len(stream) < 3640:
        epoch += 1
        order = list(ids)
        rng.shuffle(order)
        stream.extend((sid,epoch) for sid in order)
    tasks = []
    for start in range(0,3640,2):
        pair = [x[0] for x in stream[start:start+2]]
        epochs = [x[1] for x in stream[start:start+2]]
        lengths = [data.rows[s]["T"] for s in pair]
        tasks.append({"update":1821+len(tasks),"epoch":max(epochs),"clip_epochs":epochs,"clip_ids":pair,
                      "lengths":lengths,"lr":.0001,"true_frames":sum(lengths),"dense_frames":2*max(lengths)})
    return tasks

def schedule(plan,data,seed):
    """Reject any expanded pair order, length, LR, or budget change."""
    generated = expected_continuation(data,seed)
    tasks = read(bound(plan["schedules"][str(seed)]))["updates"]
    require(tasks == generated, "Expanded continuation schedule differs")
    require(len(tasks)==1820 and all(t["lr"]==.0001 for t in tasks), "Wrong continuation budget/LR")
    return tasks
'''


if __name__ == "__main__":
    main()
